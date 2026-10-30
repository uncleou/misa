"""V2 presets, optimizer coverage, fixed-batch diagnostics and training isolation."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from models import MISA
from solver import Solver
from private_selection import PrivateSelection
from scps_diagnostics import optimizer_groups, SCPSDiagnostics
from test_private_selection import config, batch
from evaluate import build_eval_config
from types import SimpleNamespace


def loader():
    t,v,a,l,bs,bt,bm=batch()
    return DataLoader([(t,v,a,torch.randn(4,1),l,bs,bt,bm)],batch_size=None)


class SCPSV2Tests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        old=config()
        new=config(variant='misa_scps_v2')
        self.assertEqual((old.scps_init_std,old.gate_lr_mult,old.scps_diagnostics),(.001,1,False))
        self.assertEqual((new.scps_init_std,new.gate_lr_mult,new.scps_diagnostics),(.01,3,True))
        self.assertEqual(new.private_selection,'shared_context')
        self.assertEqual(config(variant='misa_scps_v2',gate_lr_mult=1).gate_lr_mult,1)
        for kw in ({'gate_lr_mult':0},{'scps_init_std':float('nan')},{'scps_init_std':-1}):
            with self.assertRaises(ValueError):config(**kw)
        torch.manual_seed(2);a=PrivateSelection(8,init_std=.001)
        torch.manual_seed(2);b=PrivateSelection(8,init_std=.01)
        torch.testing.assert_close(b.gates['t'][-1].weight,10*a.gates['t'][-1].weight)
        torch.testing.assert_close(a.gates['t'][-1].bias,b.gates['t'][-1].bias)

    def test_optimizer_coverage_scheduler_and_reload(self):
        model=MISA(config(variant='misa_scps_v2'))
        model.private_t[0].weight.requires_grad=False
        groups=optimizer_groups(model,1e-4,3)
        all_ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(all_ids),len(set(all_ids)))
        self.assertEqual(set(all_ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({id(p) for p in groups[1]['params']},
                         {id(p) for p in model.private_selector.parameters() if p.requires_grad})
        opt=torch.optim.Adam(groups,lr=1e-4)
        schedule=torch.optim.lr_scheduler.ExponentialLR(opt,.5)
        opt.step();schedule.step()
        self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],3)
        state=copy.deepcopy(opt.state_dict())
        opt2=torch.optim.Adam(optimizer_groups(model,1e-4,3),lr=1e-4)
        opt2.load_state_dict(state)
        self.assertEqual([g['lr'] for g in opt.param_groups],[g['lr'] for g in opt2.param_groups])
        c=config(variant='misa_scps_v2',optimizer='Adam',learning_rate=1e-4,pretrained_emb=None)
        solver=Solver(c,None,None,None,None,None,model=model)
        with contextlib.redirect_stdout(io.StringIO()):solver.build(cuda=False)
        self.assertEqual(len(solver.optimizer.param_groups),2)

    def test_fixed_batch_metrics_rng_and_mode_restoration(self):
        with tempfile.TemporaryDirectory() as root:
            model=MISA(config(variant='misa_scps_v2')).train()
            data=loader()
            opt=torch.optim.Adam(optimizer_groups(model,1e-4,3),lr=1e-4)
            rng=torch.get_rng_state().clone()
            with contextlib.redirect_stdout(io.StringIO()):
                diag=SCPSDiagnostics(model,data,root,'x')
                initial=diag.report(0,opt)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(model.training)
            self.assertIsNone(model.private_selector.override_gate)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertEqual(initial['gates']['t']['mean_abs_change'],0)
            diag.start_epoch()
            with torch.no_grad():model.private_selector.gates['t'][-1].bias.add_(.2)
            with contextlib.redirect_stdout(io.StringIO()):current=diag.report(1,opt)
            self.assertGreater(current['gates']['t']['mean_abs_change'],0)
            self.assertGreater(current['prediction_delta_mean'],0)
            self.assertGreater(current['modules']['t_last']['relative_parameter_update'],0)
            self.assertEqual(current['modules']['context']['relative_parameter_update'],0)
            self.assertEqual(len((Path(root)/'scps_diagnostics_x.jsonl').read_text().splitlines()),2)
            self.assertTrue((Path(root)/'scps_fixed_dev_x.pt').exists())
            with self.assertRaises(FileExistsError):SCPSDiagnostics(model,data,root,'x')

    def test_diagnostics_do_not_change_training_and_eval_reloads_v2(self):
        results=[]
        torch.manual_seed(3)
        initial=MISA(config(variant='misa_scps_v2')).state_dict()
        data=loader()
        for enabled in (False,True):
            with tempfile.TemporaryDirectory() as root:
                c=config(variant='misa_scps_v2',scps_diagnostics=enabled)
                model=MISA(c);model.load_state_dict(initial)
                solver=Solver(c,None,None,data,data,data,model=model)
                solver.checkpoint_dir=Path(root)/'checkpoints'
                solver.optimizer=torch.optim.Adam(optimizer_groups(model,1e-4,3),lr=1e-4)
                solver.calc_metrics=lambda *args:0.
                torch.manual_seed(88)
                with patch('solver.to_gpu',side_effect=lambda x:x),contextlib.redirect_stdout(io.StringIO()):solver.train()
                results.append({k:v.clone() for k,v in model.state_dict().items()})
                if enabled:
                    rows=[json.loads(x) for x in (solver.checkpoint_dir/'scps_diagnostics_smoke.jsonl').read_text().splitlines()]
                    self.assertEqual(len(rows),2)
                    self.assertGreater(rows[1]['modules']['t_last']['grad_norm_mean'],0)
                    self.assertGreater(rows[1]['modules']['context']['grad_nonzero_steps'],0)
                    args=SimpleNamespace(name='smoke')
                    with contextlib.redirect_stdout(io.StringIO()):restored=build_eval_config(args,root)
                    self.assertEqual(restored.scps_init_std,.01)
                    self.assertEqual(restored.gate_lr_mult,3)
        for k in results[0]:torch.testing.assert_close(results[0][k],results[1][k],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
