"""kgeo_to_npy must produce a .npy whose records are the .kgeo pixels, byte for byte.

Uses only the standard library (no NumPy): the .npy header is a Python literal.

usage: kgeo_npy_regression.py <kerr_tracer> <kgeo_to_npy>
"""
import ast
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

tracer, converter = (str(Path(p).resolve()) for p in sys.argv[1:3])
failures = []


def check(ok, message):
    print(('[PASS] ' if ok else '[FAIL] ') + message, flush=True)
    if not ok:
        failures.append(message)


with tempfile.TemporaryDirectory(prefix='kerrtrace-npy-') as d:
    kgeo = Path(d) / 'frame.kgeo'
    subprocess.run([tracer, '--custom-res', '48', '27', '--a', '0.9', '--theta', '80',
                    '--r-obs', '40', '--bl', '--bundles', '--max-steps', '20000',
                    '--geo-only', '--geo-file', str(kgeo)],
                   check=True, capture_output=True, timeout=300)
    subprocess.run([converter, str(kgeo), str(Path(d) / 'frame')],
                   check=True, capture_output=True, timeout=60)
    npy = (Path(d) / 'frame.npy').read_bytes()
    meta = json.loads((Path(d) / 'frame.json').read_text())
    raw = kgeo.read_bytes()

    check(npy[:8] == b'\x93NUMPY\x01\x00', '.npy magic and version 1.0')
    hlen = struct.unpack('<H', npy[8:10])[0]
    header = ast.literal_eval(npy[10:10 + hlen].decode('latin1'))
    check((10 + hlen) % 64 == 0, 'data starts on a 64-byte boundary')
    check(header['shape'] == (27, 48) and header['fortran_order'] is False, 'shape is (H, W), C order')

    sizes = {'|u1': 1, '<f4': 4, '<u4': 4, '<f8': 8}
    itemsize = sum(sizes[f[1]] * (f[2][0] if len(f) > 2 else 1) for f in header['descr'])
    names = [f[0] for f in header['descr']]
    check(itemsize == 64, f'record is 64 bytes like GeoPixel (got {itemsize})')
    check(names[:3] == ['outcome', '_pad', 'r'] and 'sky_dph_b' in names and len(names) == 17,
          'all 17 GeoPixel fields named in declaration order')

    pixels_kgeo = raw[8 + 88:]          # magic + version, then KGeoMeta (88 bytes)
    check(npy[10 + hlen:] == pixels_kgeo, 'records byte-identical to the .kgeo pixels')
    W, H = struct.unpack('<II', raw[8:16])
    check((meta['W'], meta['H']) == (W, H) and meta['kgeo_version'] == struct.unpack('<I', raw[4:8])[0],
          'metadata sidecar matches the .kgeo header')

sys.exit(1 if failures else 0)
