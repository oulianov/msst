# coding: utf-8
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from ml_collections import ConfigDict
from tqdm.auto import tqdm


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class MDXNetONNX(nn.Module):
    """MDX-Net ONNX separator compatible with UVR-style two-stem configs."""

    stereo = True

    def __init__(self, config: ConfigDict):
        super().__init__()
        self.config = config
        self.session = None
        self.input_name = "input"
        self.output_name = "output"
        self.device = torch.device("cpu")

        model_config = config.model
        inference_config = config.inference
        training_config = config.training

        self.dim_f = int(_config_value(model_config, "dim_f"))
        self.dim_t = int(_config_value(model_config, "dim_t"))
        self.n_fft = int(_config_value(model_config, "n_fft"))
        self.hop_length = int(_config_value(model_config, "hop_length"))
        self.compensation = float(_config_value(model_config, "compensation", 1.0))
        self.normalization_threshold = float(
            _config_value(inference_config, "normalization_threshold", 0.9)
        )
        self.enable_denoise = bool(_config_value(inference_config, "denoise", False))

        instruments = [
            str(instrument)
            for instrument in _config_value(training_config, "instruments", [])
        ]
        primary_stem = str(
            _config_value(
                model_config,
                "primary_stem",
                _config_value(training_config, "target_instrument", ""),
            )
            or (instruments[0] if instruments else "other")
        )
        self.primary_stem = primary_stem
        self.secondary_stem = next(
            (instrument for instrument in instruments if instrument != primary_stem),
            f"no_{primary_stem}",
        )

        self.trim = self.n_fft // 2
        self.chunk_size = self.hop_length * (self.dim_t - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        if self.gen_size <= 0:
            raise ValueError(
                "MDX-Net ONNX config produces a non-positive generation size: "
                f"chunk_size={self.chunk_size}, trim={self.trim}"
            )

    def to(self, *args: Any, **kwargs: Any) -> "MDXNetONNX":
        super().to(*args, **kwargs)
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        return self

    def load_onnx_model(self, model_path: str | Path) -> "MDXNetONNX":
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        available_providers = ort.get_available_providers()
        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available_providers
        ] or available_providers

        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers,
            sess_options=session_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        return self

    def demix(
        self,
        mix: np.ndarray | torch.Tensor,
        batch_size: int = 1,
        pbar: bool = False,
    ) -> dict[str, np.ndarray]:
        if self.session is None:
            raise RuntimeError("MDX-Net ONNX model has not been loaded.")

        if isinstance(mix, torch.Tensor):
            mix_array = mix.detach().cpu().numpy()
        else:
            mix_array = np.asarray(mix)

        mix_array = mix_array.astype(np.float32, copy=False)
        if mix_array.ndim != 2:
            raise ValueError(
                f"Expected mix shape (channels, samples), got {mix_array.shape}"
            )
        if mix_array.shape[0] == 1:
            mix_array = np.repeat(mix_array, repeats=2, axis=0)
        elif mix_array.shape[0] > 2:
            mix_array = mix_array[:2]
        elif mix_array.shape[0] != 2:
            raise ValueError(
                f"Expected one or two input channels, got {mix_array.shape[0]}"
            )

        peak = float(np.abs(mix_array).max(initial=0.0))
        normalized_mix = mix_array.copy()
        normalization_gain = 1.0
        if peak > self.normalization_threshold > 0:
            normalization_gain = self.normalization_threshold / peak
            normalized_mix *= normalization_gain

        primary = self._demix_primary(normalized_mix, max(1, int(batch_size)), pbar)
        if normalization_gain != 1.0:
            primary /= normalization_gain

        primary = primary[:, : mix_array.shape[1]]
        secondary = mix_array[:, : primary.shape[1]] - primary * self.compensation
        return {
            self.primary_stem: primary.astype(np.float32, copy=False),
            self.secondary_stem: secondary.astype(np.float32, copy=False),
        }

    def _demix_primary(
        self,
        mix: np.ndarray,
        batch_size: int,
        pbar: bool,
    ) -> np.ndarray:
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        overlap = 1.0 - (
            1.0 / max(1, int(_config_value(self.config.inference, "num_overlap", 2)))
        )
        step = max(1, int((1.0 - overlap) * self.chunk_size))
        pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
        mixture = np.concatenate(
            (
                np.zeros((2, self.trim), dtype=np.float32),
                mix,
                np.zeros((2, pad), dtype=np.float32),
            ),
            axis=1,
        )

        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        chunk_ranges = list(range(0, mixture.shape[-1], step))
        iterator = (
            tqdm(chunk_ranges, desc="Processing MDX-Net chunks", leave=False)
            if pbar and should_print
            else chunk_ranges
        )

        batch_chunks: list[np.ndarray] = []
        batch_windows: list[np.ndarray | None] = []
        batch_locations: list[tuple[int, int]] = []

        for start in iterator:
            end = min(start + self.chunk_size, mixture.shape[-1])
            chunk_size_actual = end - start
            mix_part = mixture[:, start:end]
            if end != start + self.chunk_size:
                mix_part = np.concatenate(
                    (
                        mix_part,
                        np.zeros((2, start + self.chunk_size - end), dtype=np.float32),
                    ),
                    axis=-1,
                )

            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual).astype(np.float32)
                window = np.tile(window[None, None, :], (1, 2, 1))

            batch_chunks.append(mix_part)
            batch_windows.append(window)
            batch_locations.append((start, chunk_size_actual))

            if len(batch_chunks) >= batch_size:
                self._flush_batch(batch_chunks, batch_windows, batch_locations, result, divider)
                batch_chunks.clear()
                batch_windows.clear()
                batch_locations.clear()

        if batch_chunks:
            self._flush_batch(batch_chunks, batch_windows, batch_locations, result, divider)

        with np.errstate(divide="ignore", invalid="ignore"):
            primary = result / divider
        np.nan_to_num(primary, copy=False, nan=0.0)
        primary = primary[:, :, self.trim : -self.trim]
        return primary[0, :, : mix.shape[-1]]

    def _flush_batch(
        self,
        batch_chunks: list[np.ndarray],
        batch_windows: list[np.ndarray | None],
        batch_locations: list[tuple[int, int]],
        result: np.ndarray,
        divider: np.ndarray,
    ) -> None:
        mix_batch = torch.tensor(np.stack(batch_chunks), dtype=torch.float32).to(
            self.device
        )
        predicted_batch = self._run_model(mix_batch)

        for index, (start, chunk_size_actual) in enumerate(batch_locations):
            predicted = predicted_batch[index : index + 1, ..., :chunk_size_actual]
            window = batch_windows[index]
            if window is not None:
                predicted *= window
                divider[..., start : start + chunk_size_actual] += window
            else:
                divider[..., start : start + chunk_size_actual] += 1
            result[..., start : start + chunk_size_actual] += predicted

    def _run_model(self, mix: torch.Tensor) -> np.ndarray:
        spek = self._stft(mix)
        spek[:, :, :3, :] *= 0

        if self.enable_denoise:
            spec_pred_neg = self._run_session(-spek)
            spec_pred_pos = self._run_session(spek)
            spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
        else:
            spec_pred = self._run_session(spek)

        prediction = torch.tensor(spec_pred, dtype=torch.float32).to(self.device)
        return self._istft(prediction).cpu().detach().numpy()

    def _run_session(self, spek: torch.Tensor) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("MDX-Net ONNX model has not been loaded.")
        return self.session.run(
            [self.output_name],
            {self.input_name: spek.detach().cpu().numpy()},
        )[0]

    def _stft(self, input_tensor: torch.Tensor) -> torch.Tensor:
        stft_window = torch.hann_window(self.n_fft, periodic=True).to(
            input_tensor.device
        )
        batch_dimensions = input_tensor.shape[:-2]
        channel_dim, time_dim = input_tensor.shape[-2:]
        stft_output = torch.stft(
            input_tensor.reshape([-1, time_dim]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
            return_complex=True,
        )
        real_imag = torch.view_as_real(stft_output).permute([0, 3, 1, 2])
        return real_imag.reshape(
            [*batch_dimensions, channel_dim, 2, -1, real_imag.shape[-1]]
        ).reshape([*batch_dimensions, channel_dim * 2, -1, real_imag.shape[-1]])[
            ..., : self.dim_f, :
        ]

    def _istft(self, input_tensor: torch.Tensor) -> torch.Tensor:
        stft_window = torch.hann_window(self.n_fft, periodic=True).to(
            input_tensor.device
        )
        batch_dimensions = input_tensor.shape[:-3]
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]
        num_freq_bins = self.n_fft // 2 + 1
        if freq_dim < num_freq_bins:
            input_tensor = torch.cat(
                (
                    input_tensor,
                    torch.zeros(
                        [*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim],
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                    ),
                ),
                dim=-2,
            )

        complex_tensor = torch.view_as_complex(
            input_tensor.reshape(
                [*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim]
            )
            .reshape([-1, 2, num_freq_bins, time_dim])
            .permute([0, 2, 3, 1])
            .contiguous()
        )
        istft_result = torch.istft(
            complex_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            center=True,
        )
        return istft_result.reshape([*batch_dimensions, 2, -1])
