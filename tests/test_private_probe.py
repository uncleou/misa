"""Offline checks for the fixed CMD probe protocol; no dataset/BERT download."""

import contextlib
import importlib
import io
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import probe_private as probe
from config import get_config


class FakeTokenizer:
    def encode_plus(self, text, max_length, **kwargs):
        return dict(input_ids=[1] * max_length, token_type_ids=[0] * max_length,
                    attention_mask=[1] * max_length)


class FakeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 768)
        self.dropout = nn.Dropout(0.8)

    def forward(self, input_ids, **kwargs):
        return (self.dropout(self.embedding(input_ids)),)


def synthetic_split(count, seed):
    rng = torch.Generator().manual_seed(seed)
    split = {key: torch.rand(count, 128, generator=rng) for key in probe.FEATURES}
    split['labels'] = 2 * split['utt_private_a'][:, 0] - 1
    split['sample_ids'] = [str(i) for i in range(count)]
    return split


class PrivateProbeTests(unittest.TestCase):
    def test_groups_and_parameter_budget(self):
        split = synthetic_split(7, 1)
        expected = {'S': (384, 49409), 'S_T': (512, 49345), 'S_A': (512, 49345),
                    'S_V': (512, 49345), 'S_T_A_V': (768, 49281)}
        for group, (dimension, count) in expected.items():
            x = probe.group_input(split, group)
            self.assertEqual(x.shape, (7, dimension))
            torch.testing.assert_close(x[:, :384], torch.cat([split[k] for k in probe.SHARED], 1))
            self.assertEqual(sum(p.numel() for p in probe.make_head(group).parameters()), count)
        torch.testing.assert_close(probe.group_input(split, 'S_A')[:, -128:], split['utt_private_a'])

    def test_metrics_weighting_zero_labels_and_undefined_correlation(self):
        # Asymmetric class support catches reversed arguments to weighted F1.
        y = np.array([-1., -1., -1., 1., 0.])
        pred = np.array([-1., 1., 1., 1., -1.])
        metrics = probe.regression_metrics(y, pred)
        self.assertAlmostEqual(metrics['mse'], 9 / 5)
        self.assertAlmostEqual(metrics['mae'], 1.0)
        self.assertEqual(metrics['acc2_nonzero'], 0.5)
        self.assertEqual(metrics['acc2_with_zero'], 0.4)
        self.assertAlmostEqual(metrics['f1_nonzero'],
                               f1_score(y[:4] > 0, pred[:4] > 0, average='weighted'))
        empty = probe.regression_metrics([0., 0.], [1., 1.])
        self.assertIsNone(empty['corr'])
        self.assertIsNone(empty['f1_nonzero'])
        self.assertIsNone(probe.regression_metrics([1.], [2.])['corr'])

    def test_train_restore_reproducibility_and_artifacts(self):
        torch.set_num_threads(1)
        cache = dict(train=synthetic_split(23, 3), dev=synthetic_split(9, 4))
        settings = dict(probe.SETTINGS, max_epochs=8, patience=3, batch_size=7)
        row, state, prediction, history = probe.train_head(
            cache['train'], cache['dev'], 'S_A', 336, 'cpu', settings)
        self.assertAlmostEqual(row['mse'], min(h['dev_mse'] for h in history))
        self.assertEqual(row['best_epoch'], min(history, key=lambda h: h['dev_mse'])['epoch'])
        head = probe.make_head('S_A')
        head.load_state_dict(state)
        restored = probe.predict(head, probe.group_input(cache['dev'], 'S_A'), 'cpu', batch_size=4)
        torch.testing.assert_close(restored, prediction)
        repeat = probe.train_head(cache['train'], cache['dev'], 'S_A', 336, 'cpu', settings)
        torch.testing.assert_close(repeat[2], prediction, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            rows, summary = probe.run_probes(cache, output, 'cpu', seeds=(336, 337), settings=settings)
            self.assertEqual(len(rows), 10)
            self.assertEqual(len(summary), 5)
            self.assertEqual(summary[0]['delta_mae_vs_S_mean'], 0)
            baseline = {r['seed']: r['mae'] for r in rows if r['group'] == 'S'}
            for r in rows:
                self.assertAlmostEqual(r['delta_mae_vs_S'], baseline[r['seed']] - r['mae'])
            self.assertEqual(len(list(output.glob('*_dev_predictions.csv'))), 10)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['version'], probe.VERSION)

    def test_real_misa_extraction_sorted_ids_freezing_and_no_test_read(self):
        from transformers import BertTokenizer
        # Patch downloads while exercising the actual collate and MISA forward paths.
        with patch.object(BertTokenizer, 'from_pretrained', return_value=FakeTokenizer()):
            data_loader = importlib.import_module('data_loader')
        from models import MISA
        from transformers import BertConfig, BertModel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for index, length in enumerate((2, 5, 3)):
                samples.append(((np.ones(length, dtype=np.int64),
                                 np.ones((length, 3), dtype=np.float32) * index,
                                 np.ones((length, 2), dtype=np.float32) * index,
                                 ['word'] * length),
                                np.array([[float(index)]], dtype=np.float32), 'id' + str(index)))
            with open(root / 'train.pkl', 'wb') as stream:
                pickle.dump(samples, stream)
            torch.save((torch.zeros(2, 300), {'<pad>': 0, 'word': 1}), root / 'embedding_and_mapping.pt')
            with patch.object(sys, 'argv', ['test']), contextlib.redirect_stdout(io.StringIO()):
                config = get_config(parse=False, data='mosi', mode='train', batch_size=3)
            config.dataset_dir = root
            # dev/test.pkl deliberately absent: cached train-only loading must succeed.
            with patch.object(data_loader, 'bert_tokenizer', FakeTokenizer()):
                loader = data_loader.get_loader(config, shuffle=False, diagnostic=True)
                with patch.object(BertConfig, 'from_pretrained', return_value=BertConfig()), \
                        patch.object(BertModel, 'from_pretrained', return_value=FakeBert()):
                    model = MISA(config)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                cache = probe.extract_split(model, loader, 'cpu')
                repeat = probe.extract_split(model, loader, 'cpu')
            self.assertEqual(cache['sample_ids'], ['1:id1', '2:id2', '0:id0'])
            torch.testing.assert_close(cache['labels'], torch.tensor([1., 2., 0.]))
            for key in probe.FEATURES:
                torch.testing.assert_close(cache[key], repeat[key], rtol=0, atol=0)
                self.assertFalse(cache[key].requires_grad)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertTrue(all(not m.training for m in model.modules()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(before[key], value, rtol=0, atol=0)
            config.mode = 'test'
            with self.assertRaises(ValueError):
                data_loader.get_loader(config, diagnostic=True)

    def test_invalid_features_and_cli_version(self):
        split = synthetic_split(3, 5)
        split['sample_ids'][1] = split['sample_ids'][0]
        with self.assertRaises(ValueError):
            probe.validate_split(split)
        split = synthetic_split(3, 5)
        split['utt_shared_t'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            probe.validate_split(split)
        args = probe.parse_args(['--checkpoint', 'future.std'])
        self.assertEqual(args.version, 'cmd_private_probe_v1')
        self.assertEqual(probe.SEEDS, (336, 337, 338, 339, 340))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(['--checkpoint', 'future.std', '--extract_batch_size', '0'])

    def test_main_end_to_end_with_local_checkpoint(self):
        from transformers import BertTokenizer, BertConfig, BertModel
        with patch.object(BertTokenizer, 'from_pretrained', return_value=FakeTokenizer()):
            data_loader = importlib.import_module('data_loader')
        from models import MISA
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            samples = [((np.ones(2, dtype=np.int64), np.ones((2, 3), dtype=np.float32),
                         np.ones((2, 2), dtype=np.float32), ['a', 'b']),
                        np.array([[label]], dtype=np.float32), str(i))
                       for i, label in enumerate((-1., 0., 1.))]
            for split in ('train', 'dev'):
                with open(root / (split + '.pkl'), 'wb') as stream:
                    pickle.dump(samples, stream)
            torch.save((None, {'word': 1}), root / 'embedding_and_mapping.pt')

            def local_config(**kwargs):
                config = get_config(**kwargs)
                config.dataset_dir = root
                config.visual_size, config.acoustic_size = 3, 2
                return config

            with patch.object(sys, 'argv', ['test']), \
                    patch('config.get_config', side_effect=local_config), \
                    patch.object(data_loader, 'bert_tokenizer', FakeTokenizer()), \
                    patch.object(BertConfig, 'from_pretrained', return_value=BertConfig()), \
                    patch.object(BertModel, 'from_pretrained', side_effect=lambda *a, **k: FakeBert()), \
                    patch.dict(probe.SETTINGS, max_epochs=2, patience=1):
                config = local_config(parse=False, data='mosi', mode='train')
                checkpoint = root / 'cmd.std'
                torch.save(MISA(config).state_dict(), checkpoint)
                before = probe.sha256(checkpoint)
                output = root / 'output'
                args = ['--checkpoint', str(checkpoint), '--output', str(output), '--device', 'cpu']
                probe.main(args)
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(manifest['samples'], {'train': 3, 'dev': 3})
                self.assertEqual(manifest['checkpoint_sha256'], before)
                self.assertEqual(probe.sha256(checkpoint), before)
                cache = torch.load(output / 'features.pt', weights_only=True)
                self.assertEqual(set(cache), {'train', 'dev', 'metadata'})
                self.assertEqual(len(list(output.glob('*_dev_predictions.csv'))), 25)
                with self.assertRaises(FileExistsError):
                    probe.main(args)


if __name__ == '__main__':
    unittest.main()
