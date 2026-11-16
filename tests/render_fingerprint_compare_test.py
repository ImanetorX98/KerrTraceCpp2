#!/usr/bin/env python3
"""The fingerprint comparator must not report a match it has not checked.

Each case below is a way the old comparator said ALL MATCH: a case missing
from one run, two failed renders (png None == None), a non-zero exit code, a
frame that silently moved to another backend. compare() must flag every one.
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'render_fingerprint', Path(__file__).with_name('render_fingerprint.py'))
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

ok = {'rc': 0, 'png': 'aaaa', 'backend': 'cpu'}
base = {'ks_default': dict(ok), 'bl': dict(ok)}

def problems(new, old=base):
    return fp.compare(old, new)

failures = 0
def expect(label, got, want_flagged):
    global failures
    flagged = bool(got)
    if flagged != want_flagged:
        failures += 1
        print(f'[FAIL] {label}: problems={got}')
    else:
        print(f'[ok]   {label}')

expect('identical runs', problems({'ks_default': dict(ok), 'bl': dict(ok)}), False)
expect('different hash', problems({'ks_default': dict(ok), 'bl': {**ok, 'png': 'bbbb'}}), True)
expect('case missing from new run', problems({'ks_default': dict(ok)}), True)
expect('case missing from baseline', problems({**base, 'extra': dict(ok)}), True)
failed = {'rc': 1, 'png': None, 'backend': None}
expect('both renders failed', problems({**base, 'bl': dict(failed)}, {**base, 'bl': dict(failed)}), True)
expect('non-zero exit, same png', problems({**base, 'bl': {**ok, 'rc': 3}}), True)
expect('backend changed', problems({**base, 'bl': {**ok, 'backend': 'gpu-metal'}}), True)
expect('metadata ignored', problems({**base, '__meta__': {'fast': True}},
                                    {**base, '__meta__': {'fast': False}}), False)

sys.exit(1 if failures else 0)
