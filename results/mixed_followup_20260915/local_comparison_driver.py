#!/usr/bin/env python3
"""Frozen, matched-bit follow-up to the near-baseline mixed recipe.

Prepare a run directory, then execute its snapshot with `run RUN_DIR`.
Uses the existing search worker and PRISM hooks; no surrogate evaluations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

SITES = ('attn_c_attn', 'attn_c_proj', 'mlp_c_fc', 'mlp_c_proj', 'lm_head')
IMAGE = 'localhost/big-data-lab-team/fuzzy-pytorch:sr-epoch'


def config(mp, ap=6, mixed=True):
    bits = (8, ap, 8, mp, 11)
    return ','.join(f'{s}={t}:{"sr" if mixed and s.endswith("c_proj") else "rn"}'
                    for s, t in zip(SITES, bits))


def prepare(args):
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    shutil.copytree(source, root / 'LLM',
                    ignore=shutil.ignore_patterns('__pycache__', 'pre-*'))
    shutil.copytree(source.parent / 'python', root / 'python',
                    ignore=shutil.ignore_patterns('__pycache__'))
    image_id = subprocess.check_output(
        ['podman', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], text=True).strip()
    jobs = [dict(name='reference', config=','.join(f'{s}=24:rn' for s in SITES), seed=1),
            dict(name='incumbent_rn', config=config(9, mixed=False), seed=1)]
    for mp in args.mlp_bits:
        jobs.append(dict(name=f'mlp{mp}_rn', config=config(mp, mixed=False), seed=1))
        for seed in range(101, 106):
            jobs.append(dict(name=f'mlp{mp}_mixed_seed{seed}', config=config(mp), seed=seed))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for directory in ('LLM', 'python') for p in (root / directory).rglob('*')
              if p.is_file()}
    plan = dict(image=image_id, image_tag=IMAGE, cache=str(args.cache.resolve()),
                context=256, tokens=1024, threads=1, concurrency=3,
                mlp_bits=args.mlp_bits, jobs=jobs, hashes=hashes,
                purpose='Exploratory screen; fresh confirmation seeds required after selection.',
                created=time.time())
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {len(jobs)} runs in {root}', flush=True)


def result(root, name):
    p = root / 'outcomes' / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def evaluate(root, plan, job):
    previous = result(root, job['name'])
    if previous and previous['status'] == 'complete':
        return previous
    cmd = ['podman', 'run', '--rm', '--network=none', '--cpus=1', '--memory=3g',
           '-v', f'{plan["cache"]}:/hf_cache:ro',
           '-v', f'{root / "LLM"}:/workspace:ro',
           '-v', f'{root / "python"}:/pkg:ro', '-w', '/workspace']
    env = dict(PYTHONPATH='/pkg:/workspace', HF_HOME='/hf_cache', HF_HUB_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               VFC_BACKENDS=f'libinterflop_prism.so --seed={job["seed"]} --mode=rn')
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    cmd += [plan['image'], 'python3', '-u', 'search_precision_config.py', '--worker',
            '--site', job['config'], '--context_length', str(plan['context']),
            '--max_tokens', str(plan['tokens']), '--default_mode', 'rn']
    start = time.time()
    print(f'START {job["name"]}', flush=True)
    with (root / 'logs' / f'{job["name"]}.log').open('w') as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    logtext = (root / 'logs' / f'{job["name"]}.log').read_text()
    matches = re.findall(r'Result ->[^\n]*Perplexity:\s*([\deE.+-]+)', logtext)
    ppl = float(matches[-1]) if matches else None
    success = completed.returncode == 0 and ppl is not None and math.isfinite(ppl) and ppl > 0
    record = dict(**job, status='complete' if success else 'failed', ppl=ppl,
                  returncode=completed.returncode, elapsed=time.time() - start, command=cmd)
    (root / 'outcomes' / f'{job["name"]}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'{record["status"].upper()} {job["name"]} ppl={ppl}', flush=True)
    return record


def report(root, plan):
    from scipy.stats import t
    rows = []
    for mp in plan['mlp_bits']:
        rn = result(root, f'mlp{mp}_rn')
        mixed = [result(root, f'mlp{mp}_mixed_seed{s}') for s in range(101, 106)]
        if not rn or rn['status'] != 'complete' or any(
                r is None or r['status'] != 'complete' for r in mixed):
            rows.append(dict(mlp_bits=mp, status='incomplete'))
            continue
        values = [r['ppl'] for r in mixed]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        gap = rn['ppl'] - mean
        half = float(t.ppf(.975, 4)) * sd / math.sqrt(5)
        rows.append(dict(mlp_bits=mp, status='complete', rn=rn['ppl'], mixed_mean=mean,
                         mixed_sd=sd, rn_minus_mixed=gap, exploratory_ci=[gap-half, gap+half],
                         relative_reduction=gap/rn['ppl'],
                         note='Selection screen only; intervals are not adjusted for search.'))
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def run(root, local_controls=False):
    root = root.resolve()
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / 'plan.json').read_text())
    # Never execute changed source under a frozen plan.
    for relative, expected in plan['hashes'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Snapshot changed: {relative}')
    for d in ('logs', 'outcomes'):
        (root / d).mkdir(exist_ok=True)
    # Cross-machine low-precision agreement is advisory for a wholly local
    # comparison; each candidate still gets its own matched-bit local RN run.
    with ThreadPoolExecutor(max_workers=2) as pool:
        controls = list(pool.map(lambda job: evaluate(root, plan, job), plan['jobs'][:2]))
    for r, expected in zip(controls, (62.5126, 65.4538)):
        require_agreement = not local_controls or r['name'] == 'reference'
        if r['status'] != 'complete' or (require_agreement and abs(r['ppl'] - expected) > .02):
            (root / 'BLOCKED.json').write_text(json.dumps(dict(
                reason='Full-budget runtime control failed; no search jobs launched.',
                expected=expected, tolerance=.02, observed=r), indent=2) + '\n')
            return 1
    (root / 'control_comparison.json').write_text(json.dumps(dict(
        local_controls=local_controls, controls=controls,
        historical_ppl=[62.5126, 65.4538],
        note='Candidate comparisons use local matched-bit RN controls.'), indent=2) + '\n')
    blocked = root / 'BLOCKED.json'
    if blocked.exists():
        blocked.rename(root / f'previous_block_{time.time_ns()}.json')
    with ThreadPoolExecutor(max_workers=plan['concurrency']) as pool:
        records = list(pool.map(lambda job: evaluate(root, plan, job), plan['jobs'][2:]))
    report(root, plan)
    success = all(r['status'] == 'complete' for r in records)
    (root / ('COMPLETE.json' if success else 'BLOCKED.json')).write_text(
        json.dumps(dict(status='screen_complete' if success else 'failed_runs',
                        finished=time.time()), indent=2) + '\n')
    return 0 if success else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('directory', type=Path)
    prep.add_argument('--cache', type=Path, required=True)
    prep.add_argument('--mlp-bits', type=int, nargs='+', default=[8, 7, 6], choices=range(4, 10))
    for action in ('run', 'report'):
        parser = sub.add_parser(action)
        parser.add_argument('directory', type=Path)
        if action == 'run':
            parser.add_argument('--local-controls', action='store_true',
                                help='Use matched local RN controls; historical low-precision agreement is advisory.')
    args = p.parse_args()
    if args.action == 'prepare':
        if len(set(args.mlp_bits)) != len(args.mlp_bits):
            p.error('Duplicate MLP precisions')
        return prepare(args)
    if args.action == 'run':
        return run(args.directory, local_controls=args.local_controls)
    print(json.dumps(report(args.directory, json.loads(
        (args.directory / 'plan.json').read_text())), indent=2))


if __name__ == '__main__':
    sys.exit(main())
