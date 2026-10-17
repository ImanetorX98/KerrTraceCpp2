#!/usr/bin/env python3
"""Render a fixed set of small frames and hash them, for byte-exact refactor checks.

A refactor that is meant to change no behaviour should leave every hash alone.
Record a baseline with the binary built before the change, then compare:

    python3 tests/render_fingerprint.py build/kerr_tracer base.json
    # ... refactor, rebuild ...
    python3 tests/render_fingerprint.py build/kerr_tracer new.json --compare base.json

Hashes are only comparable between builds on the same machine and compiler
flags (-march=native, Metal driver). Set FAST=1 to skip the slow
semi-analytic case. Rendered PNGs are kept next to the JSON for inspection.
"""
import hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BG = str(ROOT / 'assets/backgrounds/sfondo2.jpg')
BGB = str(ROOT / 'assets/backgrounds/sfondo1.jpg')
RES = ['--custom-res', '96', '54']
BASE = RES + ['--a', '0.9', '--theta', '80', '--r-obs', '40', '--fov', '45', '--bg', BG]

CASES = {
    'ks_default':        BASE,
    'bl':                BASE + ['--bl'],
    'gks':               BASE + ['--gks'],
    'dopri5':            BASE + ['--dopri5'],
    **({} if os.environ.get('FAST') else {'semi_analytic': BASE + ['--bl', '--semi-analytic']}),
    'elliptic_closed':   BASE + ['--elliptic-closed'],
    'bundles':           BASE + ['--bundles'],
    'bundles_nofilter':  BASE + ['--bundles', '--no-bundle-filter'],
    'blackbody':         BASE + ['--disk-blackbody'],
    'stratified':        BASE + ['--disk-stratified'],
    'nasa':              BASE + ['--disk-nasa'],
    'nasa_sym':          BASE + ['--disk-nasa-symmetric'],
    'no_doppler':        BASE + ['--no-doppler'],
    'charge':            BASE + ['--charge', '0.3', '--bl'],
    'lambda':            BASE + ['--lambda', '1e-4', '--bl'],
    'extremal':          BASE + ['--a', '0.998'],
    'retrograde':        BASE + ['--a', '-0.7'],
    'wormhole':          RES + ['--wormhole', '--r-obs', '6', '--fov', '60', '--bg', BG, '--bg-b', BGB],
    'wormhole_tunnel':   RES + ['--wormhole', '--wh-tunnel', '4', '--wh-lensing', '0.5', '--r-obs', '2',
                                '--phi', '180', '--fov', '100', '--bg', BG, '--bg-b', BGB],
    'wormhole_look':     RES + ['--wormhole', '--r-obs', '5', '--wh-look', '45', '--bg', BG, '--bg-b', BGB],
    'metal_gpu':         BASE + ['--metal-kernel', 'single'],
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]

def run_case(binary, name, args, work):
    t0 = time.time()
    r = subprocess.run([binary] + args, capture_output=True, text=True, timeout=600, cwd=ROOT)
    dt = time.time() - t0
    saved = re.findall(r'^Saved: (.+)$', r.stdout, re.MULTILINE)
    backend = re.findall(r'^Backend used: (.+)$', r.stdout, re.MULTILINE)
    res = {'rc': r.returncode, 'time': round(dt, 2), 'backend': backend[-1] if backend else None}
    if saved:
        src = Path(saved[-1])
        dst = work / f'{name}.png'
        shutil.move(str(src), dst)
        res['png'] = sha(dst)
    else:
        res['png'] = None
        res['err'] = (r.stderr[-400:] + r.stdout[-400:])
    return res

def main():
    binary, out = str(Path(sys.argv[1]).resolve()), sys.argv[2]
    work = Path(out).with_suffix('')
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, args in CASES.items():
        results[name] = run_case(binary, name, args, work)
        print(f"{name:18s} {results[name]['png']} {results[name]['backend']} {results[name]['time']}s", flush=True)
    # geo-only / color-only round trip
    kgeo = work / 'roundtrip.kgeo'
    r = subprocess.run([binary] + BASE + ['--geo-only', '--geo-file', str(kgeo)],
                       capture_output=True, text=True, cwd=ROOT)
    results['geo_only'] = {'rc': r.returncode, 'png': sha(kgeo) if kgeo.exists() else None}
    r = subprocess.run([binary, '--color-only', str(kgeo), '--bg', BG, '--disk-nasa'],
                       capture_output=True, text=True, cwd=ROOT)
    saved = re.findall(r'^Saved: (.+)$', r.stdout, re.MULTILINE)
    if saved:
        dst = work / 'color_only.png'; shutil.move(saved[-1], dst)
        results['color_only'] = {'rc': r.returncode, 'png': sha(dst)}
    else:
        results['color_only'] = {'rc': r.returncode, 'png': None, 'err': r.stderr[-400:]}
    print('geo_only', results['geo_only'], 'color_only', results['color_only'].get('png'))
    json.dump(results, open(out, 'w'), indent=1)
    if '--compare' in sys.argv:
        base = json.load(open(sys.argv[sys.argv.index('--compare') + 1]))
        bad = [k for k in base if k in results and base[k].get('png') != results[k].get('png')]
        print('MISMATCH:' if bad else 'ALL MATCH', bad)
        sys.exit(1 if bad else 0)

main()
