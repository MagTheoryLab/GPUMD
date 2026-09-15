#!/usr/bin/env python3
"""GPU runtime checks for group-selected Zeeman coupling; standard library only."""
import json
import tempfile
from pathlib import Path
import validate_spin_sib_runtime as sib

CB = 5.7883818060e-5
FIELD = [12.0, -7.0, 31.0]

def grouped(model):
    lines = model.splitlines()
    lines[1] = lines[1].replace('spin:R:3', 'spin:R:3:group:I:2')
    for i in range(2, len(lines)):
        lines[i] += f' 0 {(i-2)%2}'
    return '\n'.join(lines)+'\n'

def frame(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return [[float(x) for x in line.split()[1:]] for line in lines[-n:]]

def main():
    original_run = sib.run_case
    with tempfile.TemporaryDirectory(prefix='gpumd-mfield-') as tmp:
        root = Path(tmp)
        model = grouped(sib.base_model())
        def static(name, command):
            case, result = original_run(root, name,
                'potential nep.txt\n'+command+'\nensemble nve\ntime_step 0\n'
                'dump_xyz 1 state.xyz precision double mass spin mforce potential\nrun 1\n', model)
            assert result.returncode == 0, result.stdout+result.stderr
            return frame(case/'state.xyz')
        baseline = static('baseline', '')
        applied = static('field', 'add_mfield 1 1 12 -7 31')
        for i, (before, after) in enumerate(zip(baseline, applied)):
            h = [CB*b if i%2 else 0.0 for b in FIELD]
            for k in range(3):
                assert abs((after[7+k]-before[7+k])-h[k]) < 2e-12
            expected = -sum(before[4+k]*h[k] for k in range(3))
            assert abs(after[10]-before[10]-expected) < 2e-12
        cleared = static('cleared', 'add_mfield 1 1 12 -7 31\nadd_mfield 0 0 0 0 0')
        assert cleared == baseline
        negative = ['add_mfield 0 0 1 2', 'add_mfield -1 0 1 2 3',
                    'add_mfield 0 99 1 2 3', 'add_mfield 0 0 nan 0 0',
                    'add_mfield 0 0 1 2 3 block 0 1 0 1 0 1']
        for i, command in enumerate(negative):
            _, result = original_run(root, f'negative{i}', 'potential nep.txt\n'+command+'\n', model)
            assert result.returncode != 0, command
        # Independent Cayley oracle samples the zero-field model, then adds the
        # analytic external derivative. Only the trajectory gets the command.
        original_evaluate = sib.evaluate
        def evaluate(root, name, model_text):
            result = original_evaluate(root, name, model_text)
            result['mforce'] = [[f[k]+CB*FIELD[k] for k in range(3)] for f in result['mforce']]
            return result
        def run_case(root, name, run_input, model_text=None):
            if name == 'sib':
                run_input = run_input.replace('potential nep.txt\n', 'potential nep.txt\nadd_mfield 0 0 12 -7 31\n')
            return original_run(root, name, run_input, grouped(model_text or sib.base_model()))
        sib.evaluate = evaluate
        sib.run_case = run_case
        errors = sib.one_step_oracle(root)
        print(json.dumps({'group_energy_force': 'passed', 'zero_field': 'passed',
                          'parser_negatives': len(negative), 'sib_errors': errors}))

if __name__ == '__main__':
    main()
