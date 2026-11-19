# BS-RoFormer with the spatial head and stem exchange used by gilliaan's
# CenterWide v4 model (BandSplitSpatialFormer). Copied from PhantomSplit
# (https://github.com/gilliaangilliaan/PhantomSplit, Apache-2.0, commit
# c7ecbf160982e1c4ede970f753b0304bf008277f).

from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from msst.models.bs_roformer.attend import Attend
from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(Module):
    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                 rotary_embed=rotary_embed, flash=flash_attn)

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class StemExchange(Module):
    def __init__(self, dim, freqs_per_bands, hidden=48, band_dim=16, short=(3, 5), long=(3, 21)):
        super().__init__()
        self.short = short
        self.long = long
        self.band_proj = nn.Linear(dim, band_dim)
        band_index = torch.repeat_interleave(torch.arange(len(freqs_per_bands)), torch.tensor(freqs_per_bands))
        self.register_buffer('band_index', band_index, persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(37 + 9 + band_dim, hidden, (5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, (5, 3), padding=(2, 1)),
            nn.GELU(),
        )
        self.route_head = nn.Conv2d(hidden, 14, 1)
        with torch.no_grad():
            self.route_head.weight.zero_()
            self.route_head.bias.zero_()

    @staticmethod
    def _smooth(a, size):
        kf, kt = size
        return F.avg_pool2d(a[:, None], (kf, kt), stride=1, padding=(kf // 2, kt // 2),
                            count_include_pad=False)[:, 0]

    def _relation(self, a, b, floor, size):
        cross = a * b.conj()
        pa = self._smooth(a.abs() ** 2, size)
        pb = self._smooth(b.abs() ** 2, size)
        den = torch.sqrt(pa * pb + floor)
        return self._smooth(cross.real, size) / den, self._smooth(cross.imag, size) / den

    def _copy(self, a, b, floor, size):
        cross = a * b.conj()
        pb = self._smooth(b.abs() ** 2, size)
        return torch.complex(self._smooth(cross.real, size), self._smooth(cross.imag, size)) / (pb + floor)

    def _measure(self, mix, a_m, a_s, b_m, b_s, r_m, r_s):
        short, long = self.short, self.long
        xl, xr = mix[:, 0], mix[:, 1]
        level = (xl.abs() + xr.abs()).mean(dim=(-2, -1), keepdim=True) + 1e-8
        p_floor = (level * 1e-3) ** 2
        pp_floor = p_floor ** 2

        p_l = self._smooth(xl.abs() ** 2, short)
        p_r = self._smooth(xr.abs() ** 2, short)
        rho_re, rho_im = self._relation(xl, xr, pp_floor, short)
        coh = torch.sqrt(rho_re ** 2 + rho_im ** 2 + 1e-8).clamp(max=1.0)
        ild_floor = (level * 1e-2) ** 2
        ild = (0.5 * torch.log((p_l + ild_floor) / (p_r + ild_floor))).clamp(-4, 4) / 4
        p_m = self._smooth(((xl + xr) / 2).abs() ** 2, short)
        p_s = self._smooth(((xl - xr) / 2).abs() ** 2, short)
        mix_side_share = p_s / (p_m + p_s + p_floor)
        loglev = torch.log1p((xl.abs() + xr.abs()) / level)

        p_am = self._smooth(a_m.abs() ** 2, short)
        p_as = self._smooth(a_s.abs() ** 2, short)
        p_bm = self._smooth(b_m.abs() ** 2, short)
        p_bs = self._smooth(b_s.abs() ** 2, short)
        p_rm = self._smooth(r_m.abs() ** 2, short)
        p_rs = self._smooth(r_s.abs() ** 2, short)
        a_side_share = p_as / (p_am + p_as + p_floor)
        b_side_share = p_bs / (p_bm + p_bs + p_floor)
        a_share = (p_am + p_as) / (p_am + p_as + p_bm + p_bs + p_floor)
        rest_share = (p_rm + p_rs) / (p_m + p_s + p_floor)
        rest_side_share = p_rs / (p_rm + p_rs + p_floor)

        feats = [
            rho_re, rho_im, coh, ild, mix_side_share, loglev,
            a_side_share, b_side_share, a_share,
            *self._relation(a_m, b_m, pp_floor, short),
            *self._relation(a_m, a_s, pp_floor, short),
            *self._relation(b_m, b_s, pp_floor, short),
            *self._relation(a_s, b_s, pp_floor, short),
            rest_share, rest_side_share,
            *self._relation(r_m, a_m, pp_floor, short),
            *self._relation(r_m, b_m, pp_floor, short),
            *self._relation(r_s, a_s, pp_floor, short),
            *self._relation(r_s, b_s, pp_floor, short),
            *self._relation(xl, xr, pp_floor, long),
            *self._relation(a_m, b_m, pp_floor, long),
            *self._relation(a_m, a_s, pp_floor, long),
            *self._relation(b_m, b_s, pp_floor, long),
            *self._relation(a_s, b_s, pp_floor, long),
        ]
        feats = torch.nan_to_num(torch.stack(feats, dim=1))

        copies = (
            self._copy(a_m, b_m, p_floor, short), self._copy(a_m, b_m, p_floor, long),
            self._copy(b_m, a_m, p_floor, short), self._copy(b_m, a_m, p_floor, long),
        )
        m_floor = torch.sqrt(p_floor)
        lift_a = (torch.sqrt(a_m.abs() ** 2 + a_s.abs() ** 2) / (a_m.abs() + m_floor)).clamp(1.0, 4.0)
        lift_b = (torch.sqrt(b_m.abs() ** 2 + b_s.abs() ** 2) / (b_m.abs() + m_floor)).clamp(1.0, 4.0)
        return feats, copies, (lift_a, lift_b)

    @staticmethod
    def _position(n, device):
        u = torch.arange(n, device=device, dtype=torch.float32) / max(n - 1, 1)
        feats = [u * 2 - 1]
        for octave in range(4):
            angle = 2 * torch.pi * (2 ** octave) * u
            feats.append(torch.sin(angle))
            feats.append(torch.cos(angle))
        return torch.stack(feats)

    @staticmethod
    def _gate(z):
        return torch.sigmoid(z - 5.0)

    def forward(self, tokens, mix, stem_a, stem_b):
        b, _, n_f, n_t = mix.shape
        with torch.autocast(device_type=mix.device.type, enabled=False):
            mix = mix.to(torch.complex64)
            stem_a = stem_a.to(torch.complex64)
            stem_b = stem_b.to(torch.complex64)
            a_m = (stem_a[:, 0] + stem_a[:, 1]) / 2
            a_s = (stem_a[:, 0] - stem_a[:, 1]) / 2
            b_m = (stem_b[:, 0] + stem_b[:, 1]) / 2
            b_s = (stem_b[:, 0] - stem_b[:, 1]) / 2
            rest = mix - stem_a - stem_b
            r_m = (rest[:, 0] + rest[:, 1]) / 2
            r_s = (rest[:, 0] - rest[:, 1]) / 2
            feats, copies, lifts = self._measure(mix, a_m.detach(), a_s.detach(), b_m.detach(), b_s.detach(),
                                                 r_m.detach(), r_s.detach())
            position = self._position(n_f, mix.device)[None, :, :, None].expand(b, -1, n_f, n_t)

        band = self.band_proj(tokens)
        band = band[:, :, self.band_index].permute(0, 3, 2, 1)
        z = self.route_head(self.net(torch.cat([feats.to(band.dtype), position.to(band.dtype), band], dim=1))).float()

        side_ab = self._gate(z[:, 0])
        side_ba = self._gate(z[:, 1])
        mid_ab = self._gate(z[:, 2])
        mid_ba = self._gate(z[:, 3])
        copy_ab = self._gate(z[:, 4])
        copy_ba = self._gate(z[:, 5])
        blend_ab = torch.sigmoid(z[:, 6])
        blend_ba = torch.sigmoid(z[:, 7])
        rest_side = 1.0 - self._gate(z[:, 8])
        rest_side_split = torch.sigmoid(z[:, 9])
        rest_mid = 1.0 - self._gate(z[:, 10])
        rest_mid_split = torch.sigmoid(z[:, 11])
        keep_a = torch.sigmoid(z[:, 12])
        keep_b = torch.sigmoid(z[:, 13])

        a_copy_short, a_copy_long, b_copy_short, b_copy_long = copies
        moved_copy_ab = copy_ab * (blend_ab * a_copy_short + (1.0 - blend_ab) * a_copy_long) * b_m
        moved_copy_ba = copy_ba * (blend_ba * b_copy_short + (1.0 - blend_ba) * b_copy_long) * a_m
        moved_mid_ab = moved_copy_ab + mid_ab * (a_m - moved_copy_ab)
        moved_mid_ba = moved_copy_ba + mid_ba * (b_m - moved_copy_ba)
        lift_a, lift_b = lifts
        level_a = side_ab * keep_a * (lift_a - 1.0) * a_m
        level_b = side_ba * keep_b * (lift_b - 1.0) * b_m
        given_ab = side_ab * (1.0 - keep_a) * a_s
        given_ba = side_ba * (1.0 - keep_b) * b_s

        new_a_m = a_m + level_a - moved_mid_ab + moved_mid_ba + rest_mid * rest_mid_split * r_m
        new_b_m = b_m + level_b - moved_mid_ba + moved_mid_ab + rest_mid * (1.0 - rest_mid_split) * r_m
        new_a_s = a_s - side_ab * a_s + given_ba + rest_side * rest_side_split * r_s
        new_b_s = b_s - side_ba * b_s + given_ab + rest_side * (1.0 - rest_side_split) * r_s

        stem_a = torch.stack([new_a_m + new_a_s, new_a_m - new_a_s], dim=1)
        stem_b = torch.stack([new_b_m + new_b_s, new_b_m - new_b_s], dim=1)
        return stem_a, stem_b


class SpatialHead(Module):
    def __init__(self, dim, freqs_per_bands, num_stems, hidden=64, depth=2, gain=1.0, band_dim=16,
                 short=(3, 5), long=(3, 21)):
        super().__init__()
        assert depth >= 2, 'spatial_head_depth must be at least 2'
        self.gain = gain
        self.num_stems = num_stems
        self.short = short
        self.long = long
        self.band_proj = nn.Linear(dim, band_dim)
        band_index = torch.repeat_interleave(torch.arange(len(freqs_per_bands)), torch.tensor(freqs_per_bands))
        self.register_buffer('band_index', band_index, persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(17 + 4 * num_stems + 9 + band_dim, hidden, (5, 3), padding=(2, 1)),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, (5, 3), padding=(2, 1)),
            nn.GELU(),
        )
        self.extra = ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, (5, 3), padding=(2, 1)),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 1),
            )
            for _ in range(depth - 2)
        ])
        self.out = nn.Conv2d(hidden, 8 * num_stems, 1)
        with torch.no_grad():
            for block in self.extra:
                block[2].weight.zero_()
                block[2].bias.zero_()
            self.out.weight.zero_()
            self.out.bias.zero_()

    @staticmethod
    def _smooth(a, size):
        kf, kt = size
        return F.avg_pool2d(a, (kf, kt), stride=1, padding=(kf // 2, kt // 2), count_include_pad=False)

    def _relation(self, a, b, floor, size):
        cross = a * b.conj()
        pa = self._smooth(a.abs() ** 2, size)
        pb = self._smooth(b.abs() ** 2, size)
        den = torch.sqrt(pa * pb + floor)
        return self._smooth(cross.real, size) / den, self._smooth(cross.imag, size) / den

    def _evidence(self, xl, xr, m, s, a, d):
        short, long = self.short, self.long
        al, ar = xl.abs(), xr.abs()
        level = (al + ar).mean(dim=(-2, -1), keepdim=True) + 1e-8
        p_floor = (level * 1e-3) ** 2
        pp_floor = p_floor ** 2
        amp_floor = level * 1e-2

        p_l = self._smooth(al ** 2, short)
        p_r = self._smooth(ar ** 2, short)
        rho_re, rho_im = self._relation(xl, xr, pp_floor, short)
        coh = torch.sqrt(rho_re ** 2 + rho_im ** 2 + 1e-8).clamp(max=1.0)
        ild = (0.5 * torch.log((p_l + amp_floor ** 2) / (p_r + amp_floor ** 2))).clamp(-4, 4) / 4
        pm_now = m.abs() ** 2
        ps_now = s.abs() ** 2
        p_m = self._smooth(pm_now, short)
        p_s = self._smooth(ps_now, short)
        side_share = p_s / (p_m + p_s + p_floor)
        loglev = torch.log1p((al + ar) / level)

        rho_re_long, rho_im_long = self._relation(xl, xr, pp_floor, long)
        p_m_long = self._smooth(pm_now, long)
        p_s_long = self._smooth(ps_now, long)
        side_share_long = p_s_long / (p_m_long + p_s_long + p_floor)
        ms_re, ms_im = self._relation(m, s, pp_floor, short)
        ms_re_long, ms_im_long = self._relation(m, s, pp_floor, long)

        side_now = ps_now / (pm_now + ps_now + p_floor)
        ild_now = torch.log((al + amp_floor) / (ar + amp_floor)).clamp(-4, 4) / 4
        cross_now = xl * xr.conj()
        den_now = al * ar + amp_floor ** 2
        cos_now = cross_now.real / den_now
        sin_now = cross_now.imag / den_now

        stem_mid = a * m + d * s
        stem_side = d * m + a * s
        p_stem_mid = self._smooth(stem_mid.abs() ** 2, short)
        p_stem_side = self._smooth(stem_side.abs() ** 2, short)
        stem_side_share = p_stem_side / (p_stem_mid + p_stem_side + p_floor)
        stem_share = ((p_stem_mid + p_stem_side) / (p_m + p_s + p_floor)).clamp(max=4.0)

        feats = torch.cat([
            rho_re, rho_im, coh, ild, side_share, loglev,
            rho_re_long, rho_im_long, side_share_long,
            ms_re, ms_im, ms_re_long, ms_im_long,
            side_now, ild_now, cos_now, sin_now,
            a.abs().clamp(max=4.0), d.abs().clamp(max=4.0), stem_side_share, stem_share,
        ], dim=1)
        return torch.nan_to_num(feats)

    def forward(self, tokens, mix, mask):
        b, _, n_f, n_t = mix.shape
        n = self.num_stems
        with torch.autocast(device_type=mix.device.type, enabled=False):
            mix = mix.to(torch.complex64)
            mask = torch.view_as_complex(mask.float().contiguous())
            xl, xr = mix[:, 0:1], mix[:, 1:2]
            m = (xl + xr) / 2
            s = (xl - xr) / 2
            a = (mask[:, :, 0] + mask[:, :, 1]) / 2
            d = (mask[:, :, 0] - mask[:, :, 1]) / 2
            feats = self._evidence(xl, xr, m, s, a.detach(), d.detach())
            position = StemExchange._position(n_f, mix.device)[None, :, :, None].expand(b, -1, n_f, n_t)
            lift = (torch.sqrt(a.abs() ** 2 + d.abs() ** 2) / (a.abs() + 1e-3)).clamp(1.0, 4.0).detach()

        band = self.band_proj(tokens)
        band = band[:, :, self.band_index].permute(0, 3, 2, 1)
        h = self.net(torch.cat([feats.to(band.dtype), position.to(band.dtype), band], dim=1))
        for block in self.extra:
            h = h + block(h)
        z = self.out(h).float() * self.gain
        z = z.reshape(b, n, 8, n_f, n_t)

        keep_made = 1.0 - torch.tanh(z[:, :, 0])
        keep_taken = 1.0 - torch.tanh(z[:, :, 1])
        add_mid_from_mid = torch.complex(z[:, :, 2], z[:, :, 3])
        add_mid_from_side = torch.complex(z[:, :, 4], z[:, :, 5])
        add_side_from_side = torch.complex(z[:, :, 6], z[:, :, 7])

        level = 1.0 + (1.0 - keep_made).clamp(min=0.0) * (lift - 1.0)
        mid = (a * level + add_mid_from_mid) * m + (d + add_mid_from_side) * s
        side = keep_made * d * m + (keep_taken * a + add_side_from_side) * s
        return torch.stack([mid + side, mid - side], dim=2)


class BSRoformer(Module):
    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            stem_exchange=False,
            exchange_channels=48,
            exchange_passes=2,
            spatial_head=False,
            spatial_head_channels=None,
            spatial_head_depth=None,
            spatial_head_gain=1.0,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        self.exchange = None
        if stem_exchange:
            assert stereo and num_stems == 2, 'stem_exchange needs stereo: true and exactly two stems'
            self.exchange = ModuleList([
                StemExchange(dim, freqs_per_bands, exchange_channels) for _ in range(exchange_passes)
            ])

        self.spatial_head = None
        if spatial_head:
            assert stereo, 'spatial_head needs stereo: true'
            if spatial_head_channels is None:
                spatial_head_channels = dim // 2
            if spatial_head_depth is None:
                spatial_head_depth = max(2, depth // 3)
            self.spatial_head = SpatialHead(dim, freqs_per_bands, num_stems, hidden=spatial_head_channels,
                                            depth=spatial_head_depth, gain=spatial_head_gain)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs)
            )
            tran_modules.append(
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)


        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        device = raw_audio.device

        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'


        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(raw_audio.cpu() if x_is_mps else raw_audio, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=True).to(device)

        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)


        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block

                x, ft_ps = pack([x], 'b * d')
                if self.use_torch_checkpoint:
                    x = checkpoint(linear_transformer, x, use_reentrant=False)
                else:
                    x = linear_transformer(x)
                x, = unpack(x, ft_ps, 'b * d')
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in self.mask_estimators], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)


        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')


        stft_repr = torch.view_as_complex(stft_repr)

        mix_repr = stft_repr

        sep = None
        if self.spatial_head is not None:
            mix_c = rearrange(mix_repr[:, 0], 'b (f s) t -> b s f t', s=self.audio_channels)
            mask_c = rearrange(mask, 'b n (f s) t c -> b n s f t c', s=self.audio_channels)
            if self.use_torch_checkpoint:
                sep = checkpoint(self.spatial_head, x, mix_c, mask_c, use_reentrant=False)
            else:
                sep = self.spatial_head(x, mix_c, mask_c)
        else:
            mask = torch.view_as_complex(mask)
            stft_repr = stft_repr * mask


        if self.exchange is not None:
            if sep is None:
                sep = rearrange(stft_repr, 'b n (f s) t -> b n s f t', s=self.audio_channels)
            mix_c = rearrange(mix_repr[:, 0], 'b (f s) t -> b s f t', s=self.audio_channels)
            stem_a, stem_b = sep[:, 0], sep[:, 1]
            for exchange in self.exchange:
                if self.use_torch_checkpoint:
                    stem_a, stem_b = checkpoint(exchange, x, mix_c, stem_a, stem_b, use_reentrant=False)
                else:
                    stem_a, stem_b = exchange(x, mix_c, stem_a, stem_b)
            stft_repr = rearrange(torch.stack([stem_a, stem_b], dim=1), 'b n s f t -> (b n s) f t')
        elif sep is not None:
            stft_repr = rearrange(sep, 'b n s f t -> (b n s) f t')
        else:
            stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        try:
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=raw_audio.shape[-1])
        except:
            recon_audio = torch.istft(stft_repr.cpu() if x_is_mps else stft_repr, **self.stft_kwargs, window=stft_window.cpu() if x_is_mps else stft_window, return_complex=False, length=raw_audio.shape[-1]).to(device)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')


        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
