"""Checks for label-conditioned neighborhoods, integration and global gradients."""

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
    shared_centroids, gather_centroids_and_labels, sentiment_centroid_loss,
    sentiment_centroid_stats,
)


def distributed_sentiment_worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(41)
        inputs = [torch.rand(5, 4, dtype=torch.double) for _ in range(3)]
        labels = torch.tensor([2.5, -1., 0., 2., -2.5], dtype=torch.double)
        weight = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
        full = shared_centroids(*(x @ weight for x in inputs), normalize_centroid=False)
        expected = sentiment_centroid_loss(full, labels, sigma=0.7, tau=0.3)
        expected_gradient, = torch.autograd.grad(expected, weight)
        indices = slice(0, 2) if rank == 0 else slice(2, 5)
        local = shared_centroids(*(x[indices] @ weight for x in inputs), normalize_centroid=False)
        centroids, gathered_labels = gather_centroids_and_labels(local, labels[indices, None])
        torch.testing.assert_close(centroids, full)
        torch.testing.assert_close(gathered_labels, labels)
        assert not gathered_labels.requires_grad
        loss = sentiment_centroid_loss(centroids, gathered_labels, sigma=0.7, tau=0.3)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        dist.all_reduce(weight.grad)
        weight.grad.div_(2)  # Solver.train averages parameter gradients.
        torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-8, atol=1e-10)
    finally:
        dist.destroy_process_group()


class TinySharedModel(torch.nn.Module):
    """Small trainable stand-in for exercising the real solver loop offline."""
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 3)
        self.head = torch.nn.Linear(9, 1)

    def forward(self, text, video, audio, *unused):
        self.utt_shared_t = self.shared(text).sigmoid()
        self.utt_shared_v = self.shared(video).sigmoid()
        self.utt_shared_a = self.shared(audio).sigmoid()
        return self.head(torch.cat((self.utt_shared_t, self.utt_shared_v, self.utt_shared_a), dim=-1))


class SentimentCentroidTests(unittest.TestCase):
    def test_centroid_mean_is_not_renormalized(self):
        states = [torch.eye(3)[i:i+1] * (i + 1) for i in range(3)]
        raw = shared_centroids(*states, normalize_centroid=False)
        torch.testing.assert_close(raw, torch.full((1, 3), 1 / 3))
        self.assertAlmostEqual(raw.norm().item(), 1 / math.sqrt(3), places=6)
        # Existing CU still defaults to a unit centroid.
        self.assertAlmostEqual(shared_centroids(*states).norm().item(), 1, places=6)

    def test_document_kl_against_scalar_reference(self):
        points = torch.tensor([[0., 0.], [0.3, 0.], [0., 0.8]], dtype=torch.double)
        labels = torch.tensor([2.5, 2., -2.5], dtype=torch.double)
        sigma, tau = 0.7, 0.3
        expected = 0.
        for i in range(3):
            others = [j for j in range(3) if i != j]
            q = [math.exp(-abs(labels[i].item() - labels[j].item()) / sigma) for j in others]
            p = [math.exp(-sum((points[i, k].item() - points[j, k].item()) ** 2
                              for k in range(2)) / tau) for j in others]
            q, p = [x / sum(q) for x in q], [x / sum(p) for x in p]
            expected += sum(a * math.log(a / b) for a, b in zip(q, p)) / 3
        self.assertAlmostEqual(sentiment_centroid_loss(points, labels, sigma, tau).item(), expected)
        # Uniform targets from equal labels: zero KL for equal off-diagonal distances.
        self.assertAlmostEqual(sentiment_centroid_loss(torch.eye(3), torch.zeros(3)).item(), 0.)

    def test_correct_neighborhood_beats_label_permutation(self):
        points = torch.tensor([[0., 0.], [0.05, 0.], [1., 0.]])
        labels = torch.tensor([2.5, 2., -2.5])
        good = sentiment_centroid_loss(points, labels, sigma=0.5, tau=0.2)
        bad = sentiment_centroid_loss(points, labels[[0, 2, 1]], sigma=0.5, tau=0.2)
        self.assertLess(good.item(), bad.item())
        gap, _, _ = sentiment_centroid_stats(points, labels)
        self.assertAlmostEqual(gap, (0.5 + 0.5 + 4.5) / 3, places=6)

    def test_gradients_reach_modalities_not_labels(self):
        torch.manual_seed(9)
        states = [torch.rand(4, 5, dtype=torch.double, requires_grad=True) for _ in range(3)]
        originals = [state.detach().clone() for state in states]
        labels = torch.tensor([-2., 0., 0.2, 2.], dtype=torch.double, requires_grad=True)
        centroids, targets = gather_centroids_and_labels(
            shared_centroids(*states, normalize_centroid=False), labels)
        self.assertFalse(targets.requires_grad)
        loss = sentiment_centroid_loss(centroids, targets, sigma=0.5, tau=0.2)
        loss.backward()
        self.assertIsNone(labels.grad)
        for state, original in zip(states, originals):
            torch.testing.assert_close(state, original)
            self.assertTrue(torch.isfinite(state.grad).all())
            self.assertGreater(state.grad.abs().sum().item(), 0)
        points = torch.rand(3, 2, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda p: sentiment_centroid_loss(p, labels[:3], sigma=0.5, tau=0.2), (points,)))

    def test_small_batches_collapse_and_sharp_distributions(self):
        for count in (0, 1, 2):
            points = torch.ones(count, 3, requires_grad=True)
            loss = sentiment_centroid_loss(points, torch.zeros(count))
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            torch.testing.assert_close(points.grad, torch.zeros_like(points))
        for points in (torch.ones(3, 3), torch.eye(3)):
            points.requires_grad_()
            loss = sentiment_centroid_loss(points, torch.tensor([-3., 0., 3.]), sigma=1e-4, tau=1e-4)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(points.grad).all())

    def test_configuration_and_old_cli_compatibility(self):
        for args, expected in (([], 0.), (['--cu_weight', '0.01'], 0.),
                               (['--sent_weight', '0.01', '--sent_sigma', '0.5'], 0.01)):
            with patch.object(sys, 'argv', ['test'] + args), contextlib.redirect_stdout(io.StringIO()):
                config = get_config()
            self.assertEqual(config.sent_weight, expected)
        for kwargs in ({'sent_weight': -1}, {'sent_weight': math.nan},
                       {'sent_sigma': 0}, {'sent_sigma': math.inf},
                       {'sent_tau': -1}, {'sent_tau': math.nan},
                       {'sent_weight': 0.1, 'cu_weight': 0.1}):
            with self.assertRaises(ValueError):
                Config(data='mosi', use_cmd_sim=True, **kwargs)
        with self.assertRaises(ValueError):
            Config(data='ur_funny', use_cmd_sim=True, sent_weight=0.1)
        with self.assertRaises(ValueError):
            sentiment_centroid_loss(torch.ones(3, 2), torch.ones(2))

    def test_real_solver_training_and_evaluation_routes(self):
        from solver import Solver
        for mode in ('off', 'cu', 'sent'):
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
                                sent_weight=0.01 if mode == 'sent' else 0., sent_sigma=0.5,
                                sent_tau=0.2, sim_weight=1., diff_weight=0.3, recon_weight=1.,
                                patience=1, n_epoch=1, clip=1., name='smoke')
                solver = Solver(config, None, None, loader, loader, loader, model=model)
                solver.checkpoint_dir = Path(directory)
                solver.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                # No private/reconstruction branches in the tiny stand-in model.
                solver.get_diff_loss = lambda: model.shared.weight.sum() * 0
                solver.get_recon_loss = lambda: model.shared.weight.sum() * 0
                solver.calc_metrics = lambda *args: 0.
                from shared_alignment import centroid_uniformity
                with patch('solver.to_gpu', side_effect=lambda x: x), \
                     patch('solver.sentiment_centroid_loss', wraps=sentiment_centroid_loss) as sent, \
                     patch('solver.centroid_uniformity', wraps=centroid_uniformity) as cu, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    solver.train()  # Real forward, backward, checkpoint, dev and test evaluation.
                self.assertEqual(sent.call_count, int(mode == 'sent'))
                self.assertEqual(cu.call_count, int(mode == 'cu'))
                self.assertTrue((Path(directory) / 'model_smoke.std').is_file())
                self.assertFalse(torch.equal(model.shared.weight, initial))
                if mode == 'sent':
                    self.assertIn('centroid_nn_label_gap=', output.getvalue())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo required')
    def test_distributed_pairing_and_gradient_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_sentiment_worker,
                     args=(Path(directory, 'rendezvous').as_uri(),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
