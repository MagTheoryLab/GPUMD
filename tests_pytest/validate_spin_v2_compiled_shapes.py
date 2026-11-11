#!/usr/bin/env python3
"""Compare generic and compiled version2 at high spin order and high angular rank."""
import argparse
import json
import random
import re
from pathlib import Path
from validate_spin_v2 import FIXTURE, INPUT, execute, training_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    template = (FIXTURE/'c3l2s2.nep').read_text().split('@ parameters')[0]
    cases = json.loads((FIXTURE/'c3l2s2.json').read_text())['cases']
    xyz = ''.join(training_xyz(case) for case in cases.values())
    report = {}
    # Fixed independent model dimensions: structural=30, response=14/15.
    for C,L,S,response,projection in [(2,1,4,14,62),(1,4,1,15,35)]:
        dim = 30+response
        count = 3*(32*dim+64)+1+810+projection
        header = template
        for key,value in [('spin_compress',C),('spin_l_max',L),('spin_order',S),
                          ('spin_projection_size',projection)]:
            header = re.sub(rf'(?m)^{key} .*$', f'{key} {value}', header)
        rng = random.Random(3401)
        model = header + f'@ parameters {count}\n'
        model += ''.join(f'{rng.uniform(-.1,.1):.9g}\n' for _ in range(count))
        model += f'@ q_scaler {dim}\n' + '1\n'*dim
        dirs = []
        for jit in (0,1):
            directory = args.workdir/f'c{C}l{L}s{S}-jit{jit}'
            text = INPUT.format(C=C,L=L,S=S) + f'prediction 1\noutput_descriptor 1\nnep_compile {jit}\n'
            execute(directory, text, xyz, model)
            if jit:
                assert 'Spin layout JIT ready:' in (directory/'stdout.log').read_text()
            dirs.append(directory)
        errors = {}
        for file in ('energy_train.out','force_train.out','mforce_train.out',
                     'virial_train.out','descriptor.out'):
            a,b = [[float(x) for x in (d/file).read_text().split()] for d in dirs]
            assert len(a)==len(b)
            error = max(abs(x-y) for x,y in zip(a,b))
            assert error < 1e-4, (C,L,S,file,error)
            errors[file] = error
        report[f'c{C}l{L}s{S}'] = errors
    (args.workdir/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
