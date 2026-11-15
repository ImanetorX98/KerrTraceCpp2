// server/binaries.js must pick binaries from the CMake manifests only.
// The layout below is the one that misled a measurement: build/ configured
// with USE_METAL=ON (its Metal binary is kerr_tracer) still holding an old
// kerr_tracer_metal from a previous configuration, and build_cpu/ from a
// toolchain that no longer configures, with binaries but no manifest.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { discoverBinaries, pickBinary } = require('../server/binaries');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kerrtrace-binaries-'));
const T0 = Date.now() / 1000;

function bin(rel, ageSeconds) {
  const p = path.join(root, rel);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, '');
  fs.utimesSync(p, T0 - ageSeconds, T0 - ageSeconds);
  return p;
}
function manifest(dir, binaries) {
  fs.writeFileSync(path.join(root, dir, 'kerrtrace-binaries.json'),
                   JSON.stringify({ version: 'test', binaries }));
}

try {
  const metalMain  = bin('build/kerr_tracer', 20);       // newer than the CPU-only build
  const staleMetal = bin('build/kerr_tracer_metal', 1e6);   // left from an earlier configuration
  const oldCpu     = bin('build_cpu/kerr_tracer', 10);      // newest file, but no manifest
  const oldMetal   = bin('build_cpu/kerr_tracer_metal', 10);
  const cpuOnly    = bin('build_rcpu/kerr_tracer', 50);
  const rcpuMetal  = bin('build_rcpu/kerr_tracer_metal', 200);
  manifest('build', [{ path: metalMain, backends: ['cpu', 'metal'] }]);
  manifest('build_rcpu', [{ path: cpuOnly, backends: ['cpu'] },
                          { path: rcpuMetal, backends: ['cpu', 'metal'] },
                          { path: path.join(root, 'build_rcpu/missing'), backends: ['cuda'] }]);

  const found = discoverBinaries(root);
  const paths = found.map(b => b.path);
  for (const p of [staleMetal, oldCpu, oldMetal]) {
    assert(!paths.includes(p), `unlisted binary discovered: ${p}`);
  }
  assert.strictEqual(found.length, 3, 'missing files must be dropped');

  // CPU: the CPU-only build (it has OpenMP), even though a GPU build is newer.
  assert.strictEqual(pickBinary(found, 'cpu'), cpuOnly);
  // Metal: the newest Metal-capable build.
  assert.strictEqual(pickBinary(found, 'metal'), metalMain);
  fs.utimesSync(rcpuMetal, T0, T0);
  assert.strictEqual(pickBinary(discoverBinaries(root), 'metal'), rcpuMetal);
  assert.strictEqual(pickBinary(found, 'cuda'), null);

  console.log('[PASS] server binary selection follows the build manifests');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
