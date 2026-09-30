"""Exercise the fused CUDA attention boundary without loading checkpoints."""

import unittest

import torch
from msst.models.bs_roformer.attend import Attend
from torch.nn import functional as F


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class AttentionBatchingTests(unittest.TestCase):
    def test_large_batches_preserve_order_and_values(self) -> None:
        attention = Attend(flash=True).cuda().eval()
        generator = torch.Generator(device="cuda").manual_seed(6000)
        for batch_size in (65535, 65536, 166528):
            with self.subTest(batch_size=batch_size), torch.inference_mode():
                tensors = [
                    torch.randn(
                        (batch_size, 1, 4, 64),
                        generator=generator,
                        device="cuda",
                        dtype=torch.float16,
                    )
                    for _ in range(3)
                ]
                actual = attention(*tensors)
                self.assertTrue(bool(torch.isfinite(actual).all()))
                # Compare across the split boundary and at the end, not just
                # the first chunk, against independent float32 math attention.
                for region in (slice(65530, 65536), slice(-16, None)):
                    q, k, v = (tensor[region].float() for tensor in tensors)
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=True,
                        enable_mem_efficient=False,
                        enable_cudnn=False,
                    ):
                        expected = F.scaled_dot_product_attention(q, k, v)
                    torch.testing.assert_close(
                        actual[region].float(), expected, atol=0.002, rtol=0.002
                    )

    def test_large_batch_preserves_gradients(self) -> None:
        attention = Attend(flash=True).cuda().train()
        tensors = [
            torch.randn(
                (65536, 1, 2, 16),
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        attention(*tensors).float().square().mean().backward()
        for tensor in tensors:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
