"""Numerical and distributed checks; no training data or BERT download required."""

import math
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import Config
from shared_alignment import shared_centroids, gather_centroids, centroid_uniformity


def distributed_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(17)
        inputs = [torch.randn(5, 4, dtype=torch.double) for _ in range(3)]
        weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        full_centroids = shared_centroids(*(x @ weight for x in inputs))
        expected = centroid_uniformity(full_centroids)
        expected_gradient, = torch.autograd.grad(expected, weight)

        # Unequal local sizes also exercise padding removal and cross-rank pairs.
        indices = slice(0, 2) if rank == 0 else slice(2, 5)
        local_centroids = shared_centroids(*(x[indices] @ weight for x in inputs))
        loss = centroid_uniformity(gather_centroids(local_centroids))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(2)  # Same averaging as Solver.train.
        torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-8, atol=1e-10)
    finally:
        dist.destroy_process_group()


class SharedAlignmentTests(unittest.TestCase):
    def test_uniformity_known_geometry_and_no_self_pairs(self):
        # Two orthogonal unit vectors: only one pair, squared distance = 2.
        points = torch.eye(2, dtype=torch.double, requires_grad=True)
        loss = centroid_uniformity(points, tau=2)
        self.assertAlmostEqual(loss.item(), -4.0)
        loss.backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        collapsed = torch.ones(3, 2, dtype=torch.double)
        collapsed = F.normalize(collapsed, dim=-1).requires_grad_()
        zero = centroid_uniformity(collapsed)
        self.assertAlmostEqual(zero.item(), 0)
        zero.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())

    def test_centroids_preserve_states_and_reach_all_modalities(self):
        torch.manual_seed(4)
        states = [torch.rand(4, 5, dtype=torch.double, requires_grad=True) for _ in range(3)]
        originals = [state.detach().clone() for state in states]
        centroids = shared_centroids(*states)
        torch.testing.assert_close(centroids.norm(dim=-1), torch.ones(4, dtype=torch.double))
        self.assertIs(gather_centroids(centroids), centroids)
        centroid_uniformity(centroids).backward()
        for state, original in zip(states, originals):
            torch.testing.assert_close(state, original)
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertGreater(state.grad.abs().sum().item(), 0)

    def test_singleton_and_large_tau_are_finite(self):
        point = torch.ones(1, 3, requires_grad=True)
        loss = centroid_uniformity(point)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        torch.testing.assert_close(point.grad, torch.zeros_like(point))
        self.assertTrue(torch.isfinite(centroid_uniformity(torch.eye(3), tau=1000)))

    def test_config_defaults_and_validation(self):
        config = Config(data='mosi', use_cmd_sim=True)
        self.assertEqual(config.shared_align_loss, 'cmd')
        self.assertEqual(config.cu_weight, 0)
        self.assertEqual(config.cu_tau, 2)
        for kwargs in ({'cu_weight': -1}, {'cu_weight': math.nan},
                       {'cu_tau': 0}, {'cu_tau': math.inf}):
            with self.assertRaises(ValueError):
                Config(data='mosi', use_cmd_sim=True, **kwargs)

    def test_existing_atp_variants_and_cmd_dispatch(self):
        from solver import Solver
        from utils import CMD
        states = (torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]),
                  torch.tensor([[-1., 0.]]))
        model = SimpleNamespace(utt_shared_t=states[0], utt_shared_a=states[1],
                                utt_shared_v=states[2])
        config = SimpleNamespace(use_cmd_sim=True, shared_align_loss='atp', atp_variant='text')
        solver = Solver(config, None, None, None, None, None, model=model)
        self.assertAlmostEqual(solver.get_similarity_loss().item(), 3.0)
        config.atp_variant = 'all'
        self.assertAlmostEqual(solver.get_similarity_loss().item(), 8 / 3, places=6)
        config.shared_align_loss = 'cmd'
        solver.loss_cmd = CMD()
        torch.testing.assert_close(solver.get_similarity_loss(), solver.get_cmd_loss())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_global_cu_matches_single_process_value_and_parameter_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, 'rendezvous').as_uri()
            mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
