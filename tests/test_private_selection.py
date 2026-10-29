"""SCPS routing, gradient boundaries, legacy compatibility and saved architecture."""
import contextlib
import copy
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
from private_selection import PrivateSelection
from models import MISA
from solver import Solver
from utils import DiffLoss, MSE
from evaluate import build_eval_config


def config(**kw):
    args = dict(data='mosi', mode='train', variant='misa_scps', use_bert=False,
                use_cmd_sim=True, embedding_size=4, visual_size=3, acoustic_size=2,
                hidden_size=4, num_classes=1, dropout=0., rnncell='lstm',
                activation='relu', word2id={'pad':0, 'x':1, 'y':2},
                sim_weight=1., recon_weight=1., patience=1, n_epoch=1, clip=1., name='smoke')
    args.update(kw)
    return Config(**args)


def batch():
    return (torch.ones(3, 4, dtype=torch.long), torch.randn(3, 4, 3),
            torch.randn(3, 4, 2), torch.tensor([3,3,2,1]),
            torch.ones(4,3,dtype=torch.long), torch.zeros(4,3,dtype=torch.long),
            torch.ones(4,3,dtype=torch.long))


class PrivateSelectionTests(unittest.TestCase):
    def test_config_and_invalid_combinations(self):
        c = config()
        self.assertEqual(c.private_selection, 'shared_context')
        self.assertEqual((c.diff_sp_weight, c.diff_pp_weight), (.3,.3))
        self.assertEqual(config(variant='default').private_selection, 'none')
        for kw in ({'diff_pp_weight':.1}, {'shared_align_loss':'atp'}, {'rnc_weight':.1},
                   {'scps_hidden':0}, {'scps_init_gate':1}, {'scps_init_gate':float('nan')},
                   {'variant':'misa_pdr','private_selection':'shared_context'}):
            with self.assertRaises(ValueError):
                config(**kw)

    def test_gate_initialization_and_gradient_boundary(self):
        torch.manual_seed(2)
        selector = PrivateSelection(8)
        shared = {m:torch.randn(5,8,requires_grad=True) for m in 'tav'}
        private = {m:torch.randn(5,8,requires_grad=True) for m in 'tav'}
        selected, gates = selector(shared,private)
        for m in 'tav':
            torch.testing.assert_close(selected[m], gates[m]*private[m])
            self.assertLess((gates[m]-.9).abs().max().item(), .01)
        sum(x.square().mean() for x in selected.values()).backward()
        self.assertTrue(all(x.grad is None for x in shared.values()))
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum()>0 for x in private.values()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in selector.parameters()))
        self.assertGreater(selector.context[0].weight.grad.abs().sum().item(), 0.)

    def test_real_misa_routes_selected_tokens_but_raw_reconstruction_and_diff(self):
        torch.manual_seed(3)
        model = MISA(config()).eval()
        captured = []
        hook = model.transformer_encoder.register_forward_pre_hook(lambda module, inp: captured.append(inp[0]))
        out = model(*batch())
        hook.remove()
        for i,m in enumerate('tva'):
            torch.testing.assert_close(captured[0][i], getattr(model,'utt_selected_private_'+m))
            torch.testing.assert_close(captured[0][i+3], getattr(model,'utt_shared_'+m))
            raw = getattr(model,'utt_private_'+m)
            shared = getattr(model,'utt_shared_'+m)
            torch.testing.assert_close(getattr(model,'utt_'+m), raw+shared)
            torch.testing.assert_close(getattr(model,'utt_'+m+'_recon'), getattr(model,'recon_'+m)(raw+shared))
            shared.retain_grad()
        solver = Solver(config(),None,None,None,None,None,model=model)
        solver.loss_diff = DiffLoss()
        before = solver.get_diff_loss()
        for m in 'tav':
            setattr(model,'utt_selected_private_'+m, torch.zeros_like(getattr(model,'utt_private_'+m)))
        torch.testing.assert_close(before, solver.get_diff_loss())
        out.square().mean().backward()
        for m in 'tav':
            self.assertGreater(getattr(model,'utt_shared_'+m).grad.abs().sum().item(),0)
        self.assertGreater(model.private_selector.context[0].weight.grad.abs().sum().item(),0)

    def test_legacy_states_and_ungated_ablation_are_identical(self):
        torch.manual_seed(4)
        legacy = MISA(config(variant='default')).eval()
        torch.manual_seed(4)
        ablation = MISA(config(private_selection='none')).eval()
        self.assertFalse(any(k.startswith('private_selector.') for k in legacy.state_dict()))
        ablation.load_state_dict(legacy.state_dict(), strict=True)
        inputs=batch()
        torch.testing.assert_close(legacy(*inputs), ablation(*inputs), rtol=0,atol=0)
        selected = MISA(config()).eval()
        with self.assertRaises(RuntimeError):
            selected.load_state_dict(legacy.state_dict(), strict=True)
        copied = MISA(config()).eval()
        copied.load_state_dict(selected.state_dict(),strict=True)
        torch.testing.assert_close(selected(*inputs),copied(*inputs),rtol=0,atol=0)

    def test_private_only_ablation_has_no_context(self):
        selector=PrivateSelection(4, mode='private_only')
        private={m:torch.randn(3,4,requires_grad=True) for m in 'tav'}
        selected,gates=selector({},private)
        self.assertIsNone(selector.context)
        sum(v.sum() for v in selected.values()).backward()
        self.assertTrue(all(v.grad is not None for v in private.values()))

    def test_saved_eval_config_and_conflicts(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoints';path.mkdir()
            c=config(scps_hidden=11)
            (path/'config_smoke.json').write_text(json.dumps(vars(c),default=str))
            args=SimpleNamespace(name='smoke',data=None,variant=None,private_selection=None,
                                 shared_align_loss=None,scps_hidden=None)
            with contextlib.redirect_stdout(io.StringIO()):
                restored=build_eval_config(args,root)
            self.assertEqual(restored.scps_hidden,11)
            self.assertEqual(restored.variant,'misa_scps')
            self.assertFalse(restored.use_bert)
            self.assertEqual(restored.activation,c.activation)
            args.variant='default'
            with self.assertRaises(ValueError):build_eval_config(args,root)

    def test_real_training_saves_and_restores_scps(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(5)
            c=config()
            model=MISA(c)
            # Existing Solver expects batches in T,V,A,Y,lengths,BERT... order.
            t,v,a,lengths,bs,bt,bm=batch()
            item=(t,v,a,torch.randn(4,1),lengths,bs,bt,bm)
            loader=DataLoader([item],batch_size=None)
            solver=Solver(c,None,None,loader,loader,loader,model=model)
            solver.optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            solver.checkpoint_dir=Path(directory)
            solver.calc_metrics=lambda *args:0.
            initial=model.private_selector.gates['a'][-1].weight.detach().clone()
            with patch('solver.to_gpu',side_effect=lambda t:t),contextlib.redirect_stdout(io.StringIO()) as log:
                solver.train()
            self.assertIn('SCPS gates',log.getvalue())
            self.assertFalse(torch.equal(initial,model.private_selector.gates['a'][-1].weight))
            state=torch.load(Path(directory)/'model_smoke.std',weights_only=True)
            self.assertTrue(any(k.startswith('private_selector.context.') for k in state))
            metadata=json.loads((Path(directory)/'config_smoke.json').read_text())
            self.assertEqual(metadata['private_selection'],'shared_context')

if __name__=='__main__':unittest.main()
