"""Regression checks for the Table 1 training recipe and archived loss values."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from sarc import SARC, SARCSpatial, STFTConfig, make_window, stft
from sarc.losses import spatial_objective, correction_objective, joint_objective
from sarc.training import experiment_stages, initialize_stage, optimize_stage, set_seed


class TestTable1Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = STFTConfig()
        cls.window = make_window(cls.config, torch.device("cpu"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def transfer(self):
        a = torch.ones(257, 4, dtype=torch.complex64)
        a[:, 1:] = torch.polar(torch.full((257, 3), .8),
                               torch.linspace(0, 1, 257)[:, None].expand(-1, 3))
        return a

    def batch(self, amplitude=.03):
        torch.manual_seed(25)
        x = stft(torch.randn(1, 4, 1600) * amplitude, self.window, self.config)
        s = stft(torch.randn(1, 1, 1600) * amplitude, self.window, self.config)[:, 0]
        return {"mixture": x, "target": s}

    def test_experimental_schedule(self):
        stages = experiment_stages(1)
        self.assertEqual([s.seed for s in stages], [20260816, 20260813, 20260817])
        self.assertEqual([s.batch_size for s in stages], [3, 4, 3])
        self.assertEqual([s.updates for s in stages], [600, 800, 600])
        self.assertEqual([s.learning_rate for s in stages], [2e-3, 2e-3, 5e-4])
        for index in (2, 3):
            self.assertEqual([s.seed for s in experiment_stages(index)],
                             [s.seed + (index - 1) * 10000 for s in stages])

    def test_stage1_excludes_corrector(self):
        model = initialize_stage(experiment_stages()[0], self.transfer())
        self.assertIsInstance(model, SARCSpatial)
        self.assertFalse(hasattr(model, "score_model"))
        batch = self.batch()
        with self.assertRaisesRegex(ValueError, "stage 1"):
            spatial_objective(SARC(self.transfer())(batch["mixture"]), batch["target"],
                              self.window, self.config, 1600)

    def test_stage2_freezes_and_stage3_unfreezes(self):
        stages = experiment_stages()
        spatial = initialize_stage(stages[0], self.transfer())
        correction = initialize_stage(stages[1], self.transfer(), spatial.state_dict())
        batch = self.batch()
        correction.train()
        loss = correction_objective(correction(batch["mixture"]), batch["target"],
                                    self.window, self.config, 1600)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in correction.spatial_model.parameters()))
        self.assertTrue(any(p.grad is not None for p in correction.score_model.parameters()))
        self.assertFalse(correction.spatial_model.training)
        joint = initialize_stage(stages[2], self.transfer(), correction.state_dict())
        self.assertTrue(all(p.requires_grad for p in joint.parameters()))

    def test_archived_loss_reference_values(self):
        # Recorded from the original Table 1 training functions, not from the
        # paper's abbreviated equations. The quiet case exercises the floors.
        references = {
            .03: [2.000455856323242, 2.3519344329833984, 2.6492512226104736],
            1e-7: [1.7947181463241577, 1.8401861190795898, 2.039431571960449],
        }
        for amplitude, values in references.items():
            batch = self.batch(amplitude)
            for stage, reference in zip(experiment_stages(), values):
                with self.subTest(amplitude=amplitude, stage=stage.name):
                    set_seed(stage.seed)
                    model = (SARCSpatial(self.transfer()) if stage.name == "spatial"
                             else SARC(self.transfer(), freeze_spatial=stage.name == "correction"))
                    objective = {"spatial": spatial_objective, "correction": correction_objective,
                                 "joint": joint_objective}[stage.name]
                    extra = {} if stage.name == "correction" else {
                        "mixture": batch["mixture"], "transfer": self.transfer()}
                    loss = objective(model(batch["mixture"]), batch["target"], self.window,
                                     self.config, 1600, **extra)
                    self.assertAlmostEqual(float(loss.detach()), reference, delta=5e-6)

    def test_best_validation_state_is_restored(self):
        batch = self.batch()

        class Provider:
            segment_samples = 1600

            def sample_batch(self, batch_size):
                return batch

        stage = replace(experiment_stages()[0], updates=2, batch_size=1)
        model = initialize_stage(stage, self.transfer())
        with patch("sarc.training.validate", side_effect=[10.0, 1.0]):
            best = optimize_stage(model, stage, Provider(), Provider(), torch.device("cpu"),
                                  self.window, self.config, validation_every=1, validation_batches=1)
        self.assertEqual(best["step"], 1)
        self.assertEqual(best["best_validation_delta_si_sdr"], 10.0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in best["model_state"].items()))


if __name__ == "__main__":
    unittest.main()
