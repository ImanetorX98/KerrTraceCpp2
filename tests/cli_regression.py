"""Regression checks on the production binary, with temporary outputs only."""
import os
from pathlib import Path
import subprocess
import re
import shutil
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())
failures = []
def check(ok, message):
    print(('[PASS] ' if ok else '[FAIL] ') + message, flush=True)
    if not ok:
        failures.append(message)

with tempfile.TemporaryDirectory(prefix='kerrtrace-cli-') as directory:
    root = Path(directory)
    frames = root / 'frames'
    base = [binary, '--anim', '--frames', '1', '--custom-res', '32', '18',
            '--a', '0.5', '--theta', '80', '--phi', '15', '--r-obs', '40', '--fov', '45',
            '--bl', '--max-steps', '2000', '--frames-dir', str(frames)]
    run = subprocess.run(base + ['--no-encode'], capture_output=True, text=True, timeout=60)
    check(run.returncode == 0 and 'θ=80.0°' in run.stdout and 'φ=15.0°' in run.stdout
          and 'r=40.00' in run.stdout and 'a=0.5000' in run.stdout,
          'animation inherits finite base parameters in the production build')
    check((frames / 'frame_00000.png').exists(), 'animation writes a frame')
    run = subprocess.run([binary, '--custom-res', '32', '18', '--a', '.5', '--theta', '80',
        '--phi', '15', '--r-obs', '40', '--fov', '45', '--bl', '--max-steps', '2000'],
        capture_output=True, text=True, timeout=60)
    saved = re.findall(r'^Saved: (.+)$', run.stdout, re.MULTILINE)
    check(run.returncode == 0 and bool(saved) and Path(saved[-1]).read_bytes()
          == (frames / 'frame_00000.png').read_bytes(),
          'one-frame animation matches a still, including the requested field of view')
    if os.name != 'nt':
        fake = root / 'bin'
        fake.mkdir()
        ffmpeg = fake / 'ffmpeg'
        ffmpeg.write_text('#!/bin/sh\nexit 1\n')
        ffmpeg.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ.get('PATH', ''))
        run = subprocess.run(base + ['--output', str(root / 'failed.mp4')],
                             env=env, capture_output=True, text=True, timeout=60)
        check(run.returncode != 0, 'encoding failure is reported as process failure')
        check((frames / 'frame_00000.png').exists(), 'encoding failure preserves rendered frames')
    real_ffmpeg = shutil.which('ffmpeg')
    if real_ffmpeg:
        # Quotes, spaces and dollars are literal path characters; encoding must
        # not delete unrelated files in a user-supplied frames directory.
        special = root / "frames '$ literal"
        special.mkdir()
        marker = special / 'notes.txt'
        marker.write_text('keep me')
        video = root / "video '$ literal.mp4"
        args = base.copy()
        args[-1] = str(special)
        run = subprocess.run(args + ['--output', str(video)],
                             capture_output=True, text=True, timeout=60)
        check(run.returncode == 0 and video.exists() and video.stat().st_size > 0,
              'real FFmpeg writes a video with literal special characters in paths')
        check(marker.exists() and not (special / 'frame_00000.png').exists(),
              'successful encoding removes only the consumed frames')
    # The CLI must not claim a successful geometry save when the path is invalid.
    run = subprocess.run([binary, '--geo-only', '--geo-file', str(root / 'missing/frame.kgeo'),
                          '--custom-res', '4', '4', '--max-steps', '20'],
                         capture_output=True, text=True, timeout=60)
    check(run.returncode != 0 and 'Geo saved:' not in run.stdout,
          'invalid output path does not produce a success message')

    # A .kgeo records the visibility settings it was traced with; --color-only
    # warns when they differ, and stays quiet when they match.
    geo = root / 'interstellar.kgeo'
    run = subprocess.run([binary, '--custom-res', '32', '18', '--a', '.5', '--max-steps', '2000',
                          '--disk-interstellar', '--geo-only', '--geo-file', str(geo)],
                         capture_output=True, text=True, timeout=60)
    check(run.returncode == 0 and geo.exists(), 'geometry-only trace writes a .kgeo')
    def recolor(*extra):
        return subprocess.run([binary, *extra, '--color-only', str(geo)],
                              capture_output=True, text=True, timeout=60)
    run = recolor('--disk-interstellar')
    check(run.returncode == 0 and 'Warning' not in run.stderr,
          'recolouring with the traced settings gives no warning')
    run = recolor()
    check(run.returncode == 0 and 'traced with palette "interstellar", now "blackbody"' in run.stderr,
          'recolouring with another palette warns that the geometry is the old one')
    run = recolor('--disk-interstellar', '--disk-interstellar-edge-transparency', '0.2')
    check(run.returncode == 0 and 'visibility.interstellar_edge_transparency is 0.2' in run.stderr
          and 'traced with 0.02' in run.stderr,
          'recolouring with another rim transparency names the setting')

sys.exit(bool(failures))
