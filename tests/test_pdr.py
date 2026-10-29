"""PDR config compatibility, independent DiffLoss gradients, real training route."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config, get_config
from solver import Solver
from utils import DiffLoss
from test_sentiment_centroids import TinySharedModel

class PDRTests(unittest.TestCase):
    def config(self, **kw):
        return Config(**dict(dict(data='mosi', use_cmd_sim=True), **kw))

    def test_presets_defaults_overrides_and_cli_names(self):
        for variant, pp in [('default', .3), ('misa_pdr', .1), ('misa_pdr_zero', 0.)]:
            c = self.config(variant=variant)
            self.assertAlmostEqual(c.diff_sp_weight, .3)
            self.assertAlmostEqual(c.diff_pp_weight, pp)
        c = self.config(variant='misa_pdr', diff_weight=.6)
        self.assertAlmostEqual(c.diff_sp_weight, .6)
        self.assertAlmostEqual(c.diff_pp_weight, .2)
        c = self.config(variant='misa_pdr', diff_sp_weight=.2, diff_pp_weight=0)
        self.assertEqual((c.diff_sp_weight, c.diff_pp_weight), (.2, 0))
        with patch.object(sys, 'argv', ['train', '--variant', 'misa_pdr', '--data', 'mosei', '--seed', '337']), contextlib.redirect_stdout(io.StringIO()):
            c = get_config()
        self.assertTrue(c.name.startswith('mosei_misa_pdr_s337_'))
        self.assertEqual(self.config(name='explicit').name, 'explicit')

    def test_invalid_weights_and_preset_conflicts(self):
        for key in ('diff_weight', 'diff_sp_weight', 'diff_pp_weight'):
            for value in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    self.config(**{key: value})
        for kw in ({'shared_align_loss': 'atp'}, {'use_cmd_sim': False},
                   {'cu_weight': .1}, {'sent_weight': .1}, {'rnc_weight': .1}):
            with self.assertRaises(ValueError):
                self.config(variant='misa_pdr', **kw)

    def test_pair_definitions_legacy_and_zero_pp_gradients(self):
        torch.manual_seed(7)
        tensors = [torch.randn(8, 4, dtype=torch.double, requires_grad=True) for _ in range(6)]
        st, sv, sa, pt, pv, pa = tensors
        model = SimpleNamespace(**dict(zip(('utt_shared_t', 'utt_shared_v', 'utt_shared_a',
                                           'utt_private_t', 'utt_private_v', 'utt_private_a'), tensors)))
        solver = Solver(self.config(), None, None, None, None, None, model=model)
        solver.loss_diff = DiffLoss()
        d = solver.loss_diff
        terms = [d(pt, st), d(pv, sv), d(pa, sa), d(pa, pt), d(pa, pv), d(pt, pv)]
        expected = sum(terms)
        legacy = solver.get_diff_loss()
        torch.testing.assert_close(legacy, expected)
        for pp_weight in (.3, .1, 0.):
            sp, pp = solver.get_diff_loss(return_parts=True)
            actual = .3 * sp + pp_weight * pp
            reference = .3 * sum(terms[:3]) + pp_weight * sum(terms[3:])
            torch.testing.assert_close(actual, reference)
            ga = torch.autograd.grad(actual, tensors, retain_graph=True)
            gr = torch.autograd.grad(reference, tensors, retain_graph=True)
            for a, r in zip(ga, gr):
                torch.testing.assert_close(a, r)

    def test_real_train_step_uses_absolute_weights_and_records_config(self):
        for variant, weights in [('default', (.3, .3)), ('misa_pdr', (.3, .1)), ('misa_pdr_zero', (.3, 0.))]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(10)
                model = TinySharedModel()
                c = self.config(variant=variant, sim_weight=0., recon_weight=0.,
                                patience=1, n_epoch=1, clip=1e9, name='smoke')
                x = [torch.rand(4, 4) for _ in range(3)]
                y = torch.randn(4, 1)
                aux = [torch.ones(4, 1, dtype=torch.long) for _ in range(4)]
                loader = DataLoader(TensorDataset(*x, y, *aux), batch_size=4)
                solver = Solver(c, None, None, loader, loader, loader, model=model)
                solver.checkpoint_dir = Path(directory)
                solver.optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                initial = model.shared.weight.detach().clone()
                pred = model(*x)
                expected_loss = (pred-y).square().mean() + weights[0]*model.shared.weight.square().sum() + weights[1]*model.shared.weight.sum()
                expected_grad, = torch.autograd.grad(expected_loss, model.shared.weight)
                def diff(return_parts=False):
                    sp, pp = model.shared.weight.square().sum(), model.shared.weight.sum()
                    return (sp, pp) if return_parts else sp + pp
                solver.get_diff_loss = diff
                solver.get_similarity_loss = lambda: model.shared.weight.sum()*0
                solver.get_recon_loss = lambda: model.shared.weight.sum()*0
                solver.calc_metrics = lambda *args: 0.
                with patch('solver.to_gpu', side_effect=lambda t:t), contextlib.redirect_stdout(io.StringIO()) as output:
                    solver.train()
                torch.testing.assert_close(model.shared.weight, initial - .01*expected_grad)
                self.assertIn('DiffLoss weighted:', output.getvalue())
                meta = json.loads((Path(directory)/'config_smoke.json').read_text())
                self.assertEqual(meta['variant'], variant)
                self.assertAlmostEqual(meta['diff_pp_weight'], weights[1])

if __name__ == '__main__':
    unittest.main()
