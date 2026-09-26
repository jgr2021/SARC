"""Public-release checks; no private audio or calibration files are required."""
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from sarc import SARC, STFTConfig, load_sarc, make_window, stft, istft

ROOT = Path(__file__).resolve().parents[1]


class TestRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_architecture(self):
        model = SARC(torch.ones(257, 4, dtype=torch.complex64))
        spatial, correction = model.spatial_model, model.score_model
        self.assertEqual(spatial.input_projection.weight.shape, (32, 16, 1, 1))
        self.assertEqual(spatial.gru.hidden_size, 48)
        self.assertEqual(spatial.parameter_count(), 18768)
        self.assertEqual(sum(p.numel() for p in correction.parameters()), 28066)
        self.assertEqual(correction.input_projection.weight.shape, (32, 18, 1, 1))
        self.assertEqual(correction.frequency_gru.hidden_size, 24)
        self.assertTrue(correction.frequency_gru.bidirectional)
        self.assertEqual(correction.frequency_projection.weight.shape, (32, 48))
        self.assertEqual(correction.temporal_gru.hidden_size, 48)
        self.assertEqual(correction.head.weight.shape, (2, 48))

    def test_all_frozen_checkpoints_and_outputs(self):
        expected = json.loads((ROOT / 'results/table1_sarc_verification.json').read_text())
        torch.manual_seed(927)
        mixture = torch.randn(1, 4, 257, 5, dtype=torch.complex64) * .03
        for name, digest in expected['checkpoint_sha256'].items():
            with self.subTest(checkpoint=name):
                path = ROOT / 'checkpoints' / name
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                model, _ = load_sarc(path)
                with torch.no_grad():
                    output = model(mixture)
                    quiet = model(torch.zeros_like(mixture))
                self.assertTrue(all(torch.isfinite(t).all() for t in output.values()))
                self.assertTrue(all(torch.isfinite(t).all() for t in quiet.values()))
                bound = 4 * output['spatial_mean'].abs().clamp_min(1e-8)
                self.assertTrue((output['correction'].real.abs() <= bound + 1e-7).all())
                self.assertTrue((output['correction'].imag.abs() <= bound + 1e-7).all())

    def test_invalid_rtf(self):
        for value in (0., float('nan'), float('inf')):
            transfer = torch.ones(257, 4, dtype=torch.complex64)
            transfer[0, 1] = value
            with self.assertRaisesRegex(ValueError, 'finite and nonzero'):
                SARC(transfer)

    def test_stft_validation_and_roundtrip(self):
        config = STFTConfig()
        window = make_window(config, torch.device('cpu'))
        audio = torch.randn(4, 1600) * .03
        spec = stft(audio, window, config)
        self.assertTrue(torch.allclose(istft(spec, window, config, 1600), audio, atol=1e-6))
        for bad in (torch.zeros(4, 0), torch.zeros(4, 256), torch.full((4, 1600), float('nan'))):
            with self.assertRaises(ValueError):
                stft(bad, window, config)

    def test_zero_batch_size_rejected(self):
        spec = importlib.util.spec_from_file_location('sarc_train_cli', ROOT / 'scripts/train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = ['train.py', '--train', 'train', '--validation', 'val', '--rtf', 'rtf.npy',
                '--output', 'unused.pt', '--batch-size', '0']
        with patch('sys.argv', args), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                module.parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
