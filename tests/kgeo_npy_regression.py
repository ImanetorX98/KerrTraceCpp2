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

    # v5 header: magic, version, byte-order tag, header bytes, record size,
    # meta size, KGeoMeta (W, H first), provenance length + JSON, records.
    version, tag, header_bytes, record_size = struct.unpack('<IIII', raw[4:20])
    check(version == 5 and tag == 0x01020304 and record_size == 64, 'v5 header, little-endian, 64-byte records')
    pixels_kgeo = raw[header_bytes:]
    check(npy[10 + hlen:] == pixels_kgeo, 'records byte-identical to the .kgeo pixels')
    W, H = struct.unpack('<II', raw[24:32])
    check((meta['W'], meta['H']) == (W, H) and meta['kgeo_version'] == version,
          'metadata sidecar matches the .kgeo header')
    prov = meta['provenance']
    check(prov['provenance_complete'] is True and prov['spacetime'] == 'knds'
          and prov['metric.a'] == 0.9 and prov['solver.bundles'] is True
          and prov['solver.chart'] == 'bl' and prov['camera.width'] == 48,
          'provenance names the metric, camera and solver it was traced with')
    check(prov['fields'].startswith('outcome:u1:0:1,_pad:u1:1:3,r:f4:4:4'), 'record schema in the provenance')

sys.exit(1 if failures else 0)
