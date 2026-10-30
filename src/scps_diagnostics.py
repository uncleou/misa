"""Read-only fixed-dev diagnostics plus per-epoch optimizer measurements.

Only rank 0 invokes diagnostics. No labels enter the selector and no diagnostic
loss is backpropagated. RNG and module train/eval states are restored.
"""
import json
from contextlib import contextmanager
from pathlib import Path
import torch


def optimizer_groups(model, base_lr, multiplier):
    selector = getattr(model, 'private_selector', None)
    if selector is None or multiplier == 1:
        return [p for p in model.parameters() if p.requires_grad]
    selected = [p for p in selector.parameters() if p.requires_grad]
    ids = {id(p) for p in selected}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return [{'params': base, 'lr': base_lr, 'group_name': 'backbone'},
            {'params': selected, 'lr': base_lr * multiplier, 'group_name': 'selector'}]


@contextmanager
def preserve_state(model):
    modes = [(m, m.training) for m in model.modules()]
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    override = model.private_selector.override_gate
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            yield
        finally:
            model.private_selector.override_gate = override
            for module, training in modes:
                module.training = training


def flat(parameters):
    return torch.cat([p.detach().reshape(-1).cpu() for p in parameters])


class SCPSDiagnostics:
    def __init__(self, model, dev_loader, directory, run_name):
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f'scps_diagnostics_{run_name}.jsonl'
        if self.path.exists():
            raise FileExistsError(f'Diagnostic run already exists: {self.path}; choose a new --name')
        # Cache the exact first dev batch, including sorted labels; do not touch test data.
        with preserve_state(model):
            try:
                batch = next(iter(dev_loader))
            except StopIteration:
                raise ValueError('SCPS diagnostics require a nonempty dev loader')
        self.batch = tuple(x.detach().cpu().clone() for x in batch)
        torch.save(self.batch, self.directory / f'scps_fixed_dev_{run_name}.pt')
        selector = model.private_selector
        self.parts = {}
        if selector.context is not None:
            self.parts['context'] = list(selector.context.parameters())
        for m in ('t', 'a', 'v'):
            self.parts[f'{m}_first'] = list(selector.gates[m][0].parameters())
            self.parts[f'{m}_last'] = list(selector.gates[m][-1].parameters())
            self.parts[f'{m}_norm'] = list(selector.norms[m].parameters())
        self.initial_gates = None
        self.start_epoch()

    def start_epoch(self):
        self.before = {k: flat(v) for k, v in self.parts.items()}
        self.grad_sums = {k: 0. for k in self.parts}
        self.grad_nonzero = {k: 0 for k in self.parts}
        self.grad_present = {k: 0 for k in self.parts}
        self.steps = 0

    def observe_gradients(self):
        # Called after distributed reduction, before clipping/optimizer.step.
        self.steps += 1
        for key, params in self.parts.items():
            grads = [p.grad.detach() for p in params if p.grad is not None]
            if grads:
                norm = torch.stack([g.square().sum() for g in grads]).sum().sqrt().item()
                self.grad_sums[key] += norm
                self.grad_present[key] += 1
                self.grad_nonzero[key] += int(norm > 0)

    @torch.no_grad()
    def report(self, epoch, optimizer):
        device = next(self.model.parameters()).device
        t, v, a, y, lengths, bs, bt, bm = [x.to(device) for x in self.batch]
        args = (t, v, a, lengths, bs, bt, bm)
        with preserve_state(self.model):
            self.model.private_selector.override_gate = None
            pred = self.model(*args).detach().cpu().reshape(-1)
            gates = {m: g.detach().cpu().clone() for m, g in self.model.private_gates.items()}
            if self.initial_gates is None:
                self.initial_gates = {m: g.clone() for m, g in gates.items()}
            self.model.private_selector.override_gate = .9
            fixed = self.model(*args).detach().cpu().reshape(-1)
        labels = y.cpu().reshape(-1)
        record = {'epoch': epoch, 'split': 'dev_first_batch', 'samples': len(labels),
                  'learning_rates': [g['lr'] for g in optimizer.param_groups],
                  'learned_mae': (pred-labels).abs().mean().item(),
                  'fixed_09_mae': (fixed-labels).abs().mean().item(),
                  'prediction_delta_mean': (pred-fixed).abs().mean().item(),
                  'prediction_delta_max': (pred-fixed).abs().max().item(),
                  'gates': {}, 'modules': {}}
        for m, g in gates.items():
            delta = (g-self.initial_gates[m]).abs()
            record['gates'][m] = dict(mean=g.mean().item(),
                mean_abs_change=delta.mean().item(), max_abs_change=delta.max().item(),
                mean_abs_from_09=(g-.9).abs().mean().item(),
                sample_std=g.std(dim=0, unbiased=False).mean().item())
        for key, params in self.parts.items():
            old = self.before[key]
            change = (flat(params)-old).norm().item()
            record['modules'][key] = dict(
                grad_norm_mean=self.grad_sums[key]/max(self.steps,1),
                grad_present_steps=self.grad_present[key],
                grad_nonzero_steps=self.grad_nonzero[key], steps=self.steps,
                parameter_delta_norm=change,
                relative_parameter_update=change/max(old.norm().item(),1e-12))
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        print('SCPS fixed-dev diagnostics: ' + json.dumps(record, ensure_ascii=False, allow_nan=False))
        return record
