// ============================================================
//  server/binaries.js — which kerr_tracer runs which backend
//
//  Every CMake build dir writes kerrtrace-binaries.json, listing the
//  executables that configuration actually builds and the backends each one
//  has. File names alone are not enough: build/kerr_tracer is a CPU binary
//  in one configuration and a Metal one in another, and a reconfigured dir
//  keeps the old executable on disk (a stale build/kerr_tracer_metal was once
//  picked up just because it existed). Only manifest entries are considered.
// ============================================================
const fs   = require('fs');
const path = require('path');

const MANIFEST = 'kerrtrace-binaries.json';

function mtimeMs(p) {
  try { return fs.statSync(p).mtimeMs; } catch { return null; }
}

// All binaries listed by build*/kerrtrace-binaries.json under root that exist.
function discoverBinaries(root) {
  let dirs;
  try { dirs = fs.readdirSync(root, { withFileTypes: true }); } catch { return []; }
  const found = [];
  for (const d of dirs) {
    if (!d.isDirectory() || !d.name.startsWith('build')) continue;
    const manifest = path.join(root, d.name, MANIFEST);
    let data;
    try { data = JSON.parse(fs.readFileSync(manifest, 'utf8')); } catch { continue; }
    for (const b of data.binaries || []) {
      const mtime = mtimeMs(b.path);
      if (mtime === null || !Array.isArray(b.backends)) continue;
      found.push({ path: b.path, backends: b.backends, mtime, manifest });
    }
  }
  return found;
}

// Newest binary that has the backend. For the CPU a CPU-only build wins over a
// GPU build: OpenMP is only linked into kerr_tracer when no GPU backend is on.
function pickBinary(binaries, backend) {
  let pool = binaries.filter(b => b.backends.includes(backend));
  if (backend === 'cpu') {
    const pure = pool.filter(b => b.backends.length === 1);
    if (pure.length) pool = pure;
  }
  pool.sort((a, b) => b.mtime - a.mtime);
  return pool.length ? pool[0].path : null;
}

// Newest mtime of the sources compiled into the binaries. The Metal shader is
// left out: the binary loads gpu/metal/tracer.metal from the source tree.
function newestSourceMtime(root) {
  const dirs = [root, path.join(root, 'renderer'), path.join(root, 'gpu', 'metal'),
                path.join(root, 'gpu', 'cuda')];
  let newest = 0;
  for (const dir of dirs) {
    let names;
    try { names = fs.readdirSync(dir); } catch { continue; }
    for (const n of names) {
      if (!/\.(hpp|cpp|mm|cu|cuh)$/.test(n)) continue;
      newest = Math.max(newest, mtimeMs(path.join(dir, n)) || 0);
    }
  }
  return newest;
}

module.exports = { MANIFEST, discoverBinaries, pickBinary, newestSourceMtime };
