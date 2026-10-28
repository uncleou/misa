"""RNC formula/gradient checks, real solver routing and global-batch equivalence."""

import contextlib
import io
import math
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config, get_config
from shared_alignment import (
    shared_centroids, gather_centroids_and_labels, rnc_centroid_loss,
    sentiment_centroid_loss, centroid_uniformity,
)
from test_sentiment_centroids import TinySharedModel


def reference_rnc(points, labels, tau, distance):
    """Explicit ordered-pair definition, independent of the sorted implementation."""
    terms = []
    targets = labels.detach().reshape(-1)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            candidates = [k for k in range(len(points))
                          if k != i and abs(targets[k] - targets[i]) >= abs(targets[j] - targets[i])]
            scores = []
            for k in candidates:
                delta = points[i] - points[k]
                d = delta.norm() if distance == 'l2' else delta.square().sum()
                scores.append(-d / tau)
            delta = points[i] - points[j]
            numerator = -(delta.norm() if distance == 'l2' else delta.square().sum()) / tau
            terms.append(torch.logsumexp(torch.stack(scores), dim=0) - numerator)
    return torch.stack(terms).mean()


def distributed_rnc_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        for distance in ('l2', 'l2sq'):
            torch.manual_seed(47)
            inputs = [torch.rand(5, 4, dtype=torch.double) for _ in range(3)]
            labels = torch.tensor([0., -1., 1., 0., 2.], dtype=torch.double)
            weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
            full = shared_centroids(*(x @ weight for x in inputs), normalize_centroid=False)
            expected = rnc_centroid_loss(full, labels, tau=0.3, distance=distance)
            expected_gradient, = torch.autograd.grad(expected, weight)
            indices = slice(0, 2) if rank == 0 else slice(2, 5)
            local = shared_centroids(*(x[indices] @ weight for x in inputs), normalize_centroid=False)
            centroids, targets = gather_centroids_and_labels(local, labels[indices, None])
            torch.testing.assert_close(centroids, full)
            torch.testing.assert_close(targets, labels)
            loss = rnc_centroid_loss(centroids, targets, tau=0.3, distance=distance)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            dist.all_reduce(weight.grad)
            weight.grad.div_(2)  # Real solver averages parameter gradients.
            torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-8, atol=1e-10)
    finally:
        dist.destroy_process_group()


class RNCCentroidTests(unittest.TestCase):
    def test_explicit_reference_including_ties_and_gradients(self):
        torch.manual_seed(23)
        for distance in ('l2', 'l2sq'):
            for targets in ([0., -1., 1., 0., 2.], [0.] * 5, [-2., -.3, .2, 1.1, 3.]):
                with self.subTest(distance=distance, targets=targets):
                    points = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
                    labels = torch.tensor(targets, dtype=torch.double)
                    expected = reference_rnc(points, labels, 0.7, distance)
                    actual = rnc_centroid_loss(points, labels, tau=0.7, distance=distance)
                    torch.testing.assert_close(actual, expected)
                    expected_grad, = torch.autograd.grad(expected, points)
                    actual_grad, = torch.autograd.grad(actual, points)
                    torch.testing.assert_close(actual_grad, expected_grad)
        # Equal distances/labels give uniform competition, not zero loss.
        self.assertAlmostEqual(rnc_centroid_loss(torch.eye(4), torch.zeros(4)).item(),
                               math.log(3), places=6)

    def test_permutations_and_label_scale_invariance(self):
        torch.manual_seed(11)
        points = torch.randn(5, 3, dtype=torch.double)
        labels = torch.tensor([0., -1., 1., 0., 2.], dtype=torch.double)
        order = torch.tensor([4, 1, 3, 0, 2])
        for distance in ('l2', 'l2sq'):
            actual = rnc_centroid_loss(points, labels, distance=distance)
            torch.testing.assert_close(actual, rnc_centroid_loss(points[order], labels[order], distance=distance))
            torch.testing.assert_close(actual, rnc_centroid_loss(points, 2 * labels + 5, distance=distance))

    def test_gradcheck_and_all_modality_gradients(self):
        for distance in ('l2', 'l2sq'):
            torch.manual_seed(9)
            states = [torch.rand(4, 5, dtype=torch.double, requires_grad=True) for _ in range(3)]
            originals = [state.detach().clone() for state in states]
            labels = torch.tensor([-2., 0., 0.2, 2.], dtype=torch.double, requires_grad=True)
            points = shared_centroids(*states, normalize_centroid=False)
            rnc_centroid_loss(points, labels, tau=0.2, distance=distance).backward()
            self.assertIsNone(labels.grad)
            for state, original in zip(states, originals):
                torch.testing.assert_close(state, original)
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertGreater(state.grad.abs().sum().item(), 0)
            points = torch.rand(4, 2, dtype=torch.double, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(
                lambda p: rnc_centroid_loss(p, labels, tau=0.2, distance=distance), (points,)))

    def test_small_batches_coincident_points_and_small_tau(self):
        for distance in ('l2', 'l2sq'):
            for count in (0, 1, 2):
                points = torch.ones(count, 3, requires_grad=True)
                loss = rnc_centroid_loss(points, torch.zeros(count), distance=distance)
                self.assertEqual(loss.item(), 0.)
                loss.backward()
                torch.testing.assert_close(points.grad, torch.zeros_like(points))
            for data in ([[0., 0.], [0., 0.], [0., 0.], [0., 0.]],
                         [[0., 0.], [0., 0.], [0.5, 0.], [0., 1.]]):
                points = torch.tensor(data, requires_grad=True)
                loss = rnc_centroid_loss(points, torch.tensor([-3., 0., 0., 3.]),
                                         tau=1e-4, distance=distance)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(points.grad).all())

    def test_configuration_and_input_errors(self):
        for args in ([], ['--cu_weight', '0.01'], ['--sent_weight', '0.01']):
            with patch.object(sys, 'argv', ['test'] + args), contextlib.redirect_stdout(io.StringIO()):
                config = get_config()
            self.assertEqual(config.rnc_weight, 0.)
            self.assertEqual(config.rnc_tau, 2.)
            self.assertEqual(config.rnc_distance, 'l2')
        with patch.object(sys, 'argv', ['test', '--shared_align_loss', 'atp', '--atp_variant', 'all',
                                       '--rnc_weight', '0.01', '--rnc_tau', '0.5', '--rnc_distance', 'l2sq']), \
             contextlib.redirect_stdout(io.StringIO()):
            config = get_config()
        self.assertEqual((config.rnc_weight, config.rnc_tau, config.rnc_distance), (0.01, 0.5, 'l2sq'))
        for kwargs in ({'rnc_weight': -1}, {'rnc_weight': math.nan}, {'rnc_weight': math.inf},
                       {'rnc_tau': 0}, {'rnc_tau': math.inf}, {'rnc_tau': math.nan},
                       {'rnc_distance': 'cosine'}, {'rnc_weight': 0.1, 'cu_weight': 0.1},
                       {'rnc_weight': 0.1, 'sent_weight': 0.1}):
            with self.assertRaises(ValueError):
                Config(data='mosi', use_cmd_sim=True, **kwargs)
        with self.assertRaises(ValueError):
            Config(data='ur_funny', use_cmd_sim=True, rnc_weight=0.1)
        for kwargs in ({'tau': 0}, {'tau': math.inf}, {'distance': 'cosine'}):
            with self.assertRaises(ValueError):
                rnc_centroid_loss(torch.eye(3), torch.arange(3), **kwargs)
        with self.assertRaises(ValueError):
            rnc_centroid_loss(torch.ones(3, 2), torch.ones(2))
        with self.assertRaises(ValueError):
            rnc_centroid_loss(torch.ones(3), torch.ones(3))

    def test_real_solver_routes_loss_weighting_and_evaluation(self):
        from solver import Solver
        for mode in ('off', 'cu', 'sent', 'rnc_l2', 'rnc_l2sq'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(12)
                model = TinySharedModel()
                initial = model.shared.weight.detach().clone()
                x = [torch.rand(4, 4) for _ in range(3)]
                labels = torch.tensor([[-2.], [-0.1], [0.5], [2.]])
                auxiliary = [torch.ones(4, 1, dtype=torch.long) for _ in range(4)]
                loader = DataLoader(TensorDataset(*x, labels, *auxiliary), batch_size=4)
                config = Config(data='mosi', use_cmd_sim=True, shared_align_loss='atp',
                                atp_variant='all', cu_weight=0.01 if mode == 'cu' else 0.,
                                sent_weight=0.01 if mode == 'sent' else 0.,
                                rnc_weight=0.07 if mode.startswith('rnc') else 0., rnc_tau=0.3,
                                rnc_distance='l2sq' if mode == 'rnc_l2sq' else 'l2',
                                sim_weight=1., diff_weight=0.3, recon_weight=1.,
                                patience=1, n_epoch=1, clip=1., name='smoke')
                solver = Solver(config, None, None, loader, loader, loader, model=model)
                solver.checkpoint_dir = Path(directory)
                solver.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                solver.get_diff_loss = lambda: model.shared.weight.sum() * 0
                solver.get_recon_loss = lambda: model.shared.weight.sum() * 0
                solver.calc_metrics = lambda *args: 0.
                captured = []

                def record_rnc(points, targets, **kwargs):
                    loss = rnc_centroid_loss(points, targets, **kwargs)
                    loss.retain_grad()
                    captured.append(loss)
                    self.assertFalse(targets.requires_grad)
                    return loss

                with patch('solver.to_gpu', side_effect=lambda t: t), \
                     patch('solver.rnc_centroid_loss', side_effect=record_rnc) as rnc, \
                     patch('solver.sentiment_centroid_loss', wraps=sentiment_centroid_loss) as sent, \
                     patch('solver.centroid_uniformity', wraps=centroid_uniformity) as cu, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    solver.train()
                active = mode.startswith('rnc')
                self.assertEqual(rnc.call_count, int(active))  # Neither dev nor test uses RNC.
                self.assertEqual(sent.call_count, int(mode == 'sent'))
                self.assertEqual(cu.call_count, int(mode == 'cu'))
                self.assertFalse(torch.equal(model.shared.weight, initial))
                self.assertTrue((Path(directory) / 'model_smoke.std').is_file())
                if active:
                    self.assertAlmostEqual(captured[0].grad.item(), config.rnc_weight, places=6)
                    self.assertIn('distance=' + config.rnc_distance, output.getvalue())
                    self.assertIn('centroid_nn_label_gap=', output.getvalue())
                    self.assertIn('weighted:', output.getvalue())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_global_value_and_parameter_gradients_both_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_rnc_worker,
                     args=(Path(directory, 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
