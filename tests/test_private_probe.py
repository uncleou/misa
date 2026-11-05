"""Offline checks for the masked private-probe protocol; no dataset/BERT download."""

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
import probe_utils
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
    split = {key: torch.rand(count, 8, generator=rng) for key in probe.FEATURE_ORDER}
    split['labels'] = 2 * split['utt_private_a'][:, 0] - 1
    split['sample_ids'] = [str(i) for i in range(count)]
    return split


class PrivateProbeTests(unittest.TestCase):
    def test_fixed_masks_and_shared_prefix(self):
        split = synthetic_split(7, 1)
        standardizer = probe_utils.fit_standardizer(split)
        full = probe_utils.group_input(split, 'shared', standardizer)
        self.assertEqual(full.shape, (7, 48))
        # shared group zeros the three private blocks, keeps the shared blocks
        self.assertTrue(torch.count_nonzero(full[:, 24:]) == 0)
        standardized = (probe_utils.stack_features(split) - standardizer[0]) / standardizer[1]
        torch.testing.assert_close(full[:, :24], standardized[:, :24])
        for group, pattern in probe_utils.GROUPS.items():
            x = probe_utils.group_input(split, group, standardizer)
            self.assertEqual(x.shape, (7, 48))
            active = torch.tensor(pattern).repeat_interleave(8).bool()
            torch.testing.assert_close(x[:, :24], standardized[:, :24])
            self.assertTrue(torch.count_nonzero(x[:, ~active]) == 0)
        # 'shared_all' equals the masked full standardized input
        torch.testing.assert_close(probe_utils.group_input(split, 'shared_all', standardizer),
                                   standardized)

    def test_standardizer_fits_train_only(self):
        train = synthetic_split(5, 2)
        dev = synthetic_split(4, 3)
        mean, std = probe_utils.fit_standardizer(train)
        torch.testing.assert_close(mean, probe_utils.stack_features(train).mean(0))
        # Validation input uses the training statistics, not its own.
        x_dev = probe_utils.group_input(dev, 'shared_all', (mean, std))
        torch.testing.assert_close(x_dev, (probe_utils.stack_features(dev) - mean) / std)
        # Constant training dimension must not divide by zero.
        train['utt_shared_t'][:, 0] = 3.0
        mean, std = probe_utils.fit_standardizer(train)
        self.assertTrue(torch.isfinite(std).all() and (std > 0).all())

    def test_head_parameter_budget_is_uniform(self):
        # Same head size for every group: 6d -> 64 -> 1.
        for dimension in (8, 128):
            head = probe_utils.make_head(dimension)
            self.assertEqual(sum(p.numel() for p in head.parameters()),
                             6 * dimension * 64 + 64 + 64 + 1)

    def test_metrics_weighting_zero_labels_and_undefined_correlation(self):
        # Asymmetric class support catches reversed arguments to weighted F1.
        y = np.array([-1., -1., -1., 1., 0.])
        pred = np.array([-1., 1., 1., 1., -1.])
        metrics = probe_utils.regression_metrics(y, pred)
        self.assertAlmostEqual(metrics['mse'], 9 / 5)
        self.assertAlmostEqual(metrics['mae'], 1.0)
        self.assertAlmostEqual(metrics['acc3'], np.mean(np.round(y) == np.round(pred)))
        self.assertEqual(metrics['acc2_nonzero'], 0.5)
        self.assertEqual(metrics['acc2_with_zero'], 0.4)
        self.assertAlmostEqual(metrics['f1_nonzero'],
                               f1_score(y[:4] > 0, pred[:4] > 0, average='weighted'))
        empty = probe_utils.regression_metrics([0., 0.], [1., 1.])
        self.assertIsNone(empty['corr'])
        self.assertIsNone(empty['f1_nonzero'])
        self.assertIsNone(probe_utils.regression_metrics([1.], [2.])['corr'])

    def test_train_restore_reproducibility_and_artifacts(self):
        torch.set_num_threads(1)
        cache = dict(train=synthetic_split(23, 3), dev=synthetic_split(9, 4))
        standardizer = probe_utils.fit_standardizer(cache['train'])
        settings = dict(probe_utils.SETTINGS, max_epochs=8, patience=3, batch_size=7)
        row, state, prediction, history = probe_utils.train_head(
            cache['train'], cache['dev'], 'shared_pa', 11, 'cpu', standardizer, settings)
        self.assertAlmostEqual(row['mse'], min(h['dev_mse'] for h in history))
        self.assertEqual(row['best_epoch'], min(history, key=lambda h: h['dev_mse'])['epoch'])
        head = probe_utils.make_head(probe_utils.feature_dim(cache['train']))
        head.load_state_dict(state)
        restored = probe_utils.predict(
            head, probe_utils.group_input(cache['dev'], 'shared_pa', standardizer), 'cpu', 4)
        torch.testing.assert_close(restored, prediction)
        repeat = probe_utils.train_head(
            cache['train'], cache['dev'], 'shared_pa', 11, 'cpu', standardizer, settings)
        torch.testing.assert_close(repeat[2], prediction, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)
            groups = ('shared', 'shared_pt')
            rows, summary = probe_utils.run_probes(
                cache, output, 'cpu', standardizer, groups=groups, seeds=(11, 22),
                settings=settings)
            self.assertEqual(len(rows), 4)
            self.assertEqual([s['group'] for s in summary], list(groups))
            baseline = {r['seed']: r['mae'] for r in rows if r['group'] == 'shared'}
            for r in rows:
                self.assertAlmostEqual(r['delta_mae_vs_shared'], baseline[r['seed']] - r['mae'])
            self.assertEqual(len(list(output.glob('*_dev_predictions.csv'))), 4)
            payload = json.loads((output / 'summary.json').read_text())
            self.assertEqual(payload['version'], probe_utils.VERSION)
            self.assertEqual(payload['groups'], list(groups))
            self.assertEqual(payload['seeds'], [11, 22])

    def test_architecture_restore_from_saved_config(self):
        from config import activation_dict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model_mosei_cmd_s336.std'
            checkpoint.write_bytes(b'weights')
            (root / 'config_mosei_cmd_s336.json').write_text(json.dumps(dict(
                data='mosei', variant='misa_scps_v2', private_selection='shared_context',
                use_bert=True, use_cmd_sim=True, shared_align_loss='cmd', hidden_size=128,
                scps_hidden=64, scps_init_std=0.01, gate_lr_mult=3.0, activation='relu')))
            options, source = probe.restore_architecture(checkpoint, activation_dict)
            self.assertEqual(source, str(root / 'config_mosei_cmd_s336.json'))
            self.assertEqual(options['variant'], 'misa_scps_v2')
            self.assertEqual(options['scps_init_std'], 0.01)
            self.assertEqual(options['activation'], 'relu')
            # No config file: fixed CMD defaults.
            bare = root / 'cmd.std'
            bare.write_bytes(b'weights')
            options, source = probe.restore_architecture(bare, activation_dict)
            self.assertEqual(source, 'defaults')
            self.assertEqual(options['shared_align_loss'], 'cmd')
            self.assertEqual(options['hidden_size'], 128)

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
            for key in probe.FEATURE_ORDER:
                torch.testing.assert_close(cache[key], repeat[key], rtol=0, atol=0)
                self.assertFalse(cache[key].requires_grad)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertTrue(all(not m.training for m in model.modules()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(before[key], value, rtol=0, atol=0)
            config.mode = 'test'
            with self.assertRaises(ValueError):
                data_loader.get_loader(config, diagnostic=True)

    def test_invalid_features_and_cli(self):
        split = synthetic_split(3, 5)
        split['sample_ids'][1] = split['sample_ids'][0]
        with self.assertRaises(ValueError):
            probe_utils.validate_split(split)
        split = synthetic_split(3, 5)
        split['utt_shared_t'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            probe_utils.validate_split(split)
        with self.assertRaises(ValueError):
            probe_utils.validate_split(split, hidden_size=99)
        args = probe.parse_args(['--checkpoint', 'future.std'])
        self.assertEqual(args.version, probe_utils.VERSION)
        self.assertEqual(args.groups, list(probe_utils.GROUPS))
        self.assertEqual(args.probe_seeds, list(probe_utils.DEFAULT_SEEDS))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(['--checkpoint', 'future.std', '--extract_batch_size', '0'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.parse_args(['--checkpoint', 'future.std', '--groups', 'shared', 'shared'])

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
                    patch.dict(probe_utils.SETTINGS, max_epochs=2, patience=1):
                config = local_config(parse=False, data='mosei', mode='train')
                checkpoint = root / 'cmd.std'
                torch.save(MISA(config).state_dict(), checkpoint)
                before = probe.sha256(checkpoint)
                output = root / 'output'
                args = ['--checkpoint', str(checkpoint), '--output_dir', str(output),
                        '--device', 'cpu', '--data', 'mosei',
                        '--groups', 'shared', 'shared_pt', '--probe_seeds', '11']
                probe.main(args)
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(manifest['samples'], {'train': 3, 'dev': 3})
                self.assertEqual(manifest['checkpoint_sha256'], before)
                self.assertEqual(manifest['standardization'],
                                 'train_mean_std_fit_on_train_only')
                self.assertEqual(probe.sha256(checkpoint), before)
                cache = torch.load(output / 'features.pt', weights_only=True)
                self.assertEqual(set(cache), {'train', 'dev', 'metadata'})
                self.assertEqual(len(list(output.glob('*_dev_predictions.csv'))), 2)
                with self.assertRaises(FileExistsError):
                    probe.main(args)


if __name__ == '__main__':
    unittest.main()
