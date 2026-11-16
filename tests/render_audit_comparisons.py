"""Reproduce the audit's CPU frames using two actual renderer executables.

Usage: python3 tests/render_audit_comparisons.py BEFORE AFTER OUTPUT_DIRECTORY
PNG files are copied verbatim; the NT comparison reuses identical geometry.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

before, after, output = (Path(arg).resolve() for arg in sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
manifest = []
common = ['--custom-res', '640', '360', '--bl', '--disk-blackbody',
          '--disk-out', '12', '--bg', 'assets/backgrounds/black.png']

def render(name, binary, args, animation=False):
    command = [str(binary), *args]
    print('Rendering ' + name, flush=True)
    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (output / (name + '.log')).write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(name + ': see the render log')
    if animation:
        source = output / (name + '-frames') / 'frame_00000.png'
    else:
        source = Path(re.findall(r'^Saved: (.+)$', run.stdout, re.MULTILINE)[-1])
    shutil.copyfile(source, output / (name + '.png'))
    manifest.append(dict(name=name, command=command, source=str(source)))
    (output / 'cpu-commands.json').write_text(json.dumps(manifest, indent=2))

for version, binary in [('before', before), ('after', after)]:
    for name, scene in [
        ('isco', ['--a', '-0.5', '--charge', '0.1', '--theta', '80', '--r-obs', '40', '--fov', '45']),
        ('redshift', ['--a', '0', '--theta', '75', '--r-obs', '10', '--fov', '100']),
    ]:
        label = name + '-' + version
        render(label, binary, common + scene + ['--geo-file', str(output / (label + '.kgeo'))])
    label = 'animation-' + version
    render(label, binary, common + ['--a', '.5', '--theta', '80', '--phi', '15',
        '--r-obs', '40', '--fov', '45', '--anim', '--frames', '1', '--no-encode',
        '--max-steps', '2000', '--frames-dir', str(output / (label + '-frames'))], True)

# The two recolorings below have exactly the same ray hits and redshifts.
geo = output / 'nt-shared.kgeo'
render('nt-geometry', after, common + ['--a', '-0.9', '--theta', '80', '--r-obs', '40',
    '--fov', '45', '--geo-file', str(geo)])
for version, binary in [('before', before), ('after', after)]:
    render('nt-' + version, binary, ['--color-only', str(geo), '--disk-blackbody',
        '--disk-radial-profile', 'physical_nt', '--bg', 'assets/backgrounds/black.png'])
