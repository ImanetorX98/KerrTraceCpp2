"""Metal KS: rays through a transparent Interstellar edge must not run to max_steps.

The KS loop in tracer.metal used `continue` after a pass-through, which skipped
the prev-state update and the horizon/escape checks. The stale prevZ re-detected
the same equator crossing on every step, so those rays only stopped at
max_steps: the frame was ~20x slower and the outer disk rim came out wrong.

Invariant checked here: the image must barely depend on the step cap. Raising
it from 5000 to 500000 changed 677 rim pixels before the fix and 2 after (rays
that genuinely skim the photon orbit), so at most MAX_CHANGED may differ.

usage: metal_passthrough_regression.py <kerr_tracer_metal>
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import struct
import zlib

MAX_CHANGED = 10

ROOT = Path(__file__).resolve().parents[1]
binary = str(Path(sys.argv[1]).resolve())
scene = ['--custom-res', '640', '360', '--a', '0.998', '--theta', '84', '--r-obs', '60',
         '--fov', '50', '--disk-out', '20', '--disk-interstellar', '--ks',
         '--metal-kernel', 'single',
         # A sky is required: without one a stuck ray and an escaping ray are both black.
         '--bg', str(ROOT / 'assets' / 'backgrounds' / 'sfondo4.jpg')]


def png_rgb(data):
    """Decode an 8-bit RGB/RGBA PNG (what stb_image_write produces) to rows of bytes."""
    pos, idat, w, h, ch = 8, b'', 0, 0, 3
    while pos < len(data):
        n, kind = struct.unpack('>I4s', data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
            assert depth == 8 and ctype in (2, 6), 'unsupported PNG'
            ch = 3 if ctype == 2 else 4
        elif kind == b'IDAT':
            idat += body
        pos += 12 + n
    raw, stride, rows, prev = zlib.decompress(idat), w * ch, [], bytearray(w * ch)
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            left = line[i - ch] if i >= ch else 0
            up, ul = prev[i], (prev[i - ch] if i >= ch else 0)
            if f == 1: line[i] = (line[i] + left) & 255
            elif f == 2: line[i] = (line[i] + up) & 255
            elif f == 3: line[i] = (line[i] + ((left + up) >> 1)) & 255
            elif f == 4:
                pa, pb, pc = abs(up - ul), abs(left - ul), abs(left + up - 2 * ul)
                line[i] = (line[i] + (left if pa <= pb and pa <= pc else up if pb <= pc else ul)) & 255
        rows.append(bytes(line)); prev = line
    return w, h, ch, rows


def changed_pixels(a, b):
    wa, ha, ca, ra = png_rgb(a)
    wb, hb, cb, rb = png_rgb(b)
    assert (wa, ha) == (wb, hb)
    return sum(1 for y in range(ha) for x in range(wa)
               if ra[y][x * ca:x * ca + 3] != rb[y][x * cb:x * cb + 3])


def render(max_steps, out_dir):
    run = subprocess.run([binary] + scene + ['--max-steps', str(max_steps)],
                         capture_output=True, text=True, cwd=ROOT, timeout=600)
    saved = re.findall(r'^Saved: (.+)$', run.stdout, re.MULTILINE)
    backend = re.findall(r'^Backend used: (.+)$', run.stdout, re.MULTILINE)
    if run.returncode != 0 or not saved:
        sys.exit(f'[FAIL] render failed (max_steps={max_steps}): {run.stderr[-400:]}')
    if not backend or backend[-1] != 'gpu-metal':
        print(f'[SKIP] Metal not used (backend={backend[-1] if backend else "?"})')
        sys.exit(0)
    target = Path(out_dir) / f'steps_{max_steps}.png'
    Path(saved[-1]).replace(target)
    return target.read_bytes()


with tempfile.TemporaryDirectory(prefix='kerrtrace-passthrough-') as d:
    capped = render(5000, d)
    full = render(500000, d)
    n = changed_pixels(capped, full)
    if n <= MAX_CHANGED:
        print(f'[PASS] {n} pixel(s) depend on the step cap (limit {MAX_CHANGED}): no rays stuck after a pass-through')
        sys.exit(0)
    print(f'[FAIL] {n} pixels change with --max-steps (limit {MAX_CHANGED}): rays run to the step cap')
    sys.exit(1)
