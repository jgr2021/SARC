from __future__ import annotations

import unittest

import torch

from sarc import SARC


def make_model() -> SARC:
    transfer = torch.ones(257, 4, dtype=torch.complex64)
    return SARC(transfer)


class TestSARC(unittest.TestCase):
    def test_parameter_count(self) -> None:
        self.assertEqual(make_model().parameter_count(), 46_834)

    def test_unit_response_and_lossless_residual(self) -> None:
        torch.manual_seed(0)
        model = make_model().eval()
        mixture = torch.randn(1, 4, 257, 5, dtype=torch.complex64)
        with torch.no_grad():
            output = model.spatial_model(mixture)
        self.assertLess(torch.max(torch.abs(output["constraint"] - 1)).item(), 1e-5)
        reconstructed = output["mean"][:, None] + output["residual"]
        self.assertTrue(
            torch.allclose(reconstructed, output["aligned_observation"], atol=1e-6)
        )

    def test_correction_is_complex(self) -> None:
        model = make_model().eval()
        mixture = torch.randn(1, 4, 257, 3, dtype=torch.complex64)
        with torch.no_grad():
            output = model(mixture)
        self.assertTrue(torch.is_complex(output["correction"]))
        self.assertEqual(output["mean"].shape, (1, 257, 3))


if __name__ == "__main__":
    unittest.main()
