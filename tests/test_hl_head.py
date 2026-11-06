"""HL distribution head: soft labels, expectation, config, routing and migration."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config
from models import MISA
from solver import Solver
import distribution
from test_private_selection import config, batch


def hl_config(**kw):
    return config(variant='default', prediction_head='distribution', pretrained_emb=None, **kw)


def item_batch(labels=None):
    """Solver-order batch: (t, v, a, y, lengths, bert, types, mask)."""
    t, v, a, lengths, bs, bt, bm = batch()
    y = torch.randn(len(lengths), 1) if labels is None else labels
    return (t, v, a, y, lengths, bs, bt, bm)


def loader(labels=None):
    return DataLoader([item_batch(labels)], batch_size=None)


class HLHeadTests(unittest.TestCase):
    def test_preset_defaults_and_invalid_config(self):
        self.assertEqual(config(variant='misa_hl').prediction_head, 'distribution')
        self.assertEqual(config(variant='default').prediction_head, 'scalar')
        self.assertEqual(hl_config().num_classes, 1)  # scalar trunk preserved
        self.assertEqual(hl_config(hl_bins=11).hl_bins, 11)
        base = dict(data='mosi', use_cmd_sim=True, variant='default',
                    use_bert=False, num_classes=1, prediction_head='distribution')
        for kw in ({'hl_bins': 1}, {'hl_sigma': 0}, {'hl_sigma': -1},
                   {'hl_min': 1.0, 'hl_max': 1.0}, {'hl_reg_weight': -1}):
            with self.assertRaises(ValueError):
                Config(**dict(base, **kw))
        with self.assertRaises(ValueError):
            Config(**dict(base, data='ur_funny'))

    def test_soft_targets_and_expectation(self):
        c = hl_config(hl_bins=21)
        targets = distribution.soft_targets(torch.tensor([-3.0, 0.0, 3.0, 0.3]), c)
        torch.testing.assert_close(targets.sum(-1), torch.ones(4))
        self.assertTrue((targets >= 0).all())
        # Label at a bin centre peaks at that bin; the extreme label peaks at an edge bin.
        self.assertEqual(int(targets[1].argmax()), 10)
        self.assertIn(int(targets[0].argmax()), (0,))
        self.assertIn(int(targets[2].argmax()), (c.hl_bins - 1,))

        centers = distribution.bin_centers(c)
        logits = torch.zeros(5, c.hl_bins)
        pred = distribution.expectation(logits, c)
        torch.testing.assert_close(pred, torch.full((5,), centers.mean()))
        onehot = torch.full((1, c.hl_bins), -50.0)
        onehot[0, 3] = 50.0
        torch.testing.assert_close(distribution.expectation(onehot, c), centers[3:4])

    def test_hl_task_loss_is_finite_and_gradient_flows(self):
        c = hl_config()
        logits = torch.randn(6, c.hl_bins, requires_grad=True)
        labels = torch.linspace(-3, 3, 6)
        loss = distribution.hl_task_loss(logits, labels, c)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        reg = distribution.hl_task_loss(logits, labels, hl_config(hl_reg_weight=0.5))
        self.assertGreater(reg.item(), 0)

    def test_model_head_shapes_and_scalar_unchanged(self):
        t, v, a, lengths, bs, bt, bm = batch()
        hl = MISA(hl_config()).eval()
        self.assertIsNotNone(hl.hl_projection)
        self.assertTrue(any(k.startswith('hl_projection.') for k in hl.state_dict()))
        self.assertEqual(hl(*batch()).shape, (len(lengths), hl_config().hl_bins))

        scalar = MISA(config(variant='default')).eval()
        self.assertIsNone(scalar.hl_projection)
        self.assertFalse(any(k.startswith('hl_projection.') for k in scalar.state_dict()))
        self.assertEqual(scalar(*batch()).shape, (len(lengths), 1))

    def test_distribution_route_freezes_scalar_layer_and_only_updates_new_head(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(7)
            c = hl_config(optimizer='Adam', learning_rate=1e-3, n_epoch=1, patience=1)
            model = MISA(c)
            solver = Solver(c, None, None, None, None, None, model=model)
            solver.checkpoint_dir = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                solver.build(cuda=False)
            frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
            self.assertEqual(frozen, {'fusion.fusion_layer_3.weight',
                                      'fusion.fusion_layer_3.bias'})
            optimized = {id(p) for g in solver.optimizer.param_groups for p in g['params']}
            self.assertNotIn(id(model.fusion[3].weight), optimized)
            self.assertIn(id(model.hl_projection.weight), optimized)
            # From-scratch: no warm-start key is ever loaded from a scalar checkpoint.
            self.assertNotIn('hl_migrate_scalar', vars(c))

    def test_solver_trains_hl_and_selects_on_dev_mae(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(7)
            c = hl_config(optimizer='Adam', learning_rate=1e-3, n_epoch=2, patience=2)
            model = MISA(c)
            data = loader()
            solver = Solver(c, None, None, data, data, data, model=model)
            solver.checkpoint_dir = Path(directory)
            solver.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            solver.calc_metrics = lambda *args: 0.
            with patch('solver.to_gpu', side_effect=lambda x: x), \
                    contextlib.redirect_stdout(io.StringIO()) as log:
                solver.train()
            self.assertIn('Prediction head: distribution', log.getvalue())
            state = torch.load(Path(directory) / 'model_smoke.std', weights_only=True)
            self.assertIn('hl_projection.weight', state)
            metadata = json.loads((Path(directory) / 'config_smoke.json').read_text())
            self.assertEqual(metadata['prediction_head'], 'distribution')

            # Dev/eval loss for the HL route is a finite MAE over converted scores.
            eval_loss, _ = solver.eval(mode='dev')
            self.assertTrue(torch.isfinite(torch.tensor(eval_loss)).item())

    def test_hl_training_updates_distribution_head_only_from_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(9)
            c = hl_config(optimizer='Adam', learning_rate=1e-3, n_epoch=1, patience=1)
            model = MISA(c)
            scalar_w = model.fusion[3].weight.detach().clone()
            scalar_b = model.fusion[3].bias.detach().clone()
            hl_w = model.hl_projection.weight.detach().clone()
            data = loader()
            solver = Solver(c, None, None, data, data, data, model=model)
            solver.checkpoint_dir = Path(directory)
            solver.calc_metrics = lambda *args: 0.
            with contextlib.redirect_stdout(io.StringIO()):
                solver.build(cuda=False)
            with patch('solver.to_gpu', side_effect=lambda x: x), \
                    contextlib.redirect_stdout(io.StringIO()):
                solver.train()
            # Frozen scalar layer is byte-identical; only the new head trains.
            torch.testing.assert_close(model.fusion[3].weight, scalar_w, rtol=0, atol=0)
            torch.testing.assert_close(model.fusion[3].bias, scalar_b, rtol=0, atol=0)
            self.assertFalse(torch.allclose(model.hl_projection.weight, hl_w))


if __name__ == '__main__':
    unittest.main()
