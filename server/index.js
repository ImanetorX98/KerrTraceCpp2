// ============================================================
//  server/index.js — KNdS Render API
//  HTTP  :3001  →  REST endpoints
//  WS    :3001/ws  →  real-time render progress
// ============================================================
const express    = require('express');
const cors       = require('cors');
const http       = require('http');
const WebSocket  = require('ws');
const path       = require('path');
const fs         = require('fs');
const { spawn, spawnSync } = require('child_process');

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server, path: '/ws' });

const ROOT         = path.resolve(__dirname, '..');
const BINARY_MAIN  = path.join(ROOT, 'build', 'kerr_tracer');
const BINARY_CPU_ONLY = path.join(ROOT, 'build_cpu', 'kerr_tracer');
const BINARY_METAL_CPU = path.join(ROOT, 'build_cpu', 'kerr_tracer_metal');
const BINARY_METAL_LEGACY = path.join(ROOT, 'build', 'kerr_tracer_metal');
const BINARY_CUDA  = path.join(ROOT, 'build', 'kerr_tracer_cuda');
const OUT_DIR      = path.join(ROOT, 'out');
const THUMB_DIR    = path.join(OUT_DIR, '.thumbs');
const QUEUE_STATE_FILE = path.join(OUT_DIR, '.queue_state.json');
const ASSETS_DIR   = path.join(ROOT, 'assets', 'backgrounds');
const DEFAULT_BACKGROUND = 'sfondo5.jpg';
const NAV_BAKES_DIR = path.join(OUT_DIR, 'nav_bakes');
fs.mkdirSync(NAV_BAKES_DIR, { recursive: true });

function firstExisting(paths) {
  for (const p of paths) {
    if (p && fs.existsSync(p)) return p;
  }
  return null;
}

// Prefer dedicated CPU build when available.
const BINARY_CPU = firstExisting([BINARY_CPU_ONLY, BINARY_MAIN]);
// Prefer Metal binary built in build_cpu (kept most up-to-date in this workflow).
const BINARY_METAL = firstExisting([BINARY_METAL_CPU, BINARY_MAIN, BINARY_METAL_LEGACY]);

function resolveBinary(backend) {
  if (backend === 'metal' && BINARY_METAL) return BINARY_METAL;
  if (backend === 'cuda'  && fs.existsSync(BINARY_CUDA)) return BINARY_CUDA;
  if (BINARY_CPU) return BINARY_CPU;
  return BINARY_MAIN;
}

function availableBackends() {
  const b = [];
  if (BINARY_CPU) b.push('cpu');
  if (BINARY_METAL) b.push('metal');
  if (fs.existsSync(BINARY_CUDA))  b.push('cuda');
  return b;
}

// ── Resolutions ───────────────────────────────────────────────
const RESOLUTIONS = {
  '144p':  { w: 256,  h: 144  },
  '256p':  { w: 454,  h: 256  },
  '480p':  { w: 854,  h: 480  },
  '512p':  { w: 910,  h: 512  },
  '720p':  { w: 1280, h: 720  },
  '1080p': { w: 1920, h: 1080 },
  '2K':    { w: 2560, h: 1440 },
  '4K':    { w: 3840, h: 2160 },
};

const DEFAULT_R_OBS = 60;
const DEFAULT_DISK_OUT = 12;

app.use(cors());
app.use(express.json());
app.use('/renders', express.static(OUT_DIR));
fs.mkdirSync(THUMB_DIR, { recursive: true });

// ── Render queue state ────────────────────────────────────────
let activeJob = null;
let queuedJobs = [];
let recentJobs = [];
let nextJobId = 1;
const MAX_RECENT_JOBS = 80;

// ── Nav bake state ────────────────────────────────────────────
let navBakeJob = null; // { id, bakeDir, frames, renderParams, done, total, cancelled, currentProc }
const QUEUE_STATE_VERSION = 1;

function nowSeconds() {
  return Date.now() / 1000;
}

function nowIso() {
  return new Date().toISOString();
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function readArgValue(args, flag) {
  const idx = Array.isArray(args) ? args.indexOf(flag) : -1;
  if (idx < 0 || idx + 1 >= args.length) return null;
  return args[idx + 1];
}

function derivePixelCount(resolution, args) {
  if (resolution === 'custom') {
    const idx = Array.isArray(args) ? args.indexOf('--custom-res') : -1;
    if (idx >= 0 && idx + 2 < args.length) {
      const w = safeNumber(args[idx + 1], 0);
      const h = safeNumber(args[idx + 2], 0);
      if (w > 0 && h > 0) return Math.floor(w * h);
    }
  }
  const dim = RESOLUTIONS[resolution] || RESOLUTIONS['1080p'];
  return Math.floor(dim.w * dim.h);
}

function deriveCameraSpp(args) {
  const val = safeNumber(readArgValue(args, '--camera-spp'), 1);
  return Math.max(1, Math.floor(val));
}

function attachRenderMetrics(job) {
  if (!job || job.kind !== 'render') return;
  const pixelCount = derivePixelCount(job.resolution, job.args);
  const spp = deriveCameraSpp(job.args);
  const rayCount = pixelCount * spp;
  job.pixelCount = pixelCount;
  job.cameraSpp = spp;
  job.rayCount = rayCount;
  if (!Number.isFinite(job.donePixels)) job.donePixels = 0;
  if (!Number.isFinite(job.doneRays)) job.doneRays = 0;
  if (!Number.isFinite(job.throughputPixPerSec)) job.throughputPixPerSec = 0;
  if (!Number.isFinite(job.throughputRaysPerSec)) job.throughputRaysPerSec = 0;
  if (!Number.isFinite(job.etaSmoothedSec)) job.etaSmoothedSec = 0;
}

function smoothValue(prev, next, alpha = 0.25) {
  if (!Number.isFinite(next) || next < 0) return Number.isFinite(prev) ? prev : 0;
  if (!Number.isFinite(prev) || prev <= 0) return next;
  return alpha * next + (1 - alpha) * prev;
}

function updateDerivedProgress(job, pct, elapsed, etaRaw) {
  if (!job) return;
  const clampedPct = Math.max(0, Math.min(100, safeNumber(pct, 0)));
  const elapsedSafe = Math.max(0, safeNumber(elapsed, 0));
  const etaSafe = Math.max(0, safeNumber(etaRaw, 0));

  job.progressPct = clampedPct;
  job.elapsedSec = elapsedSafe;
  job.etaSec = etaSafe;

  if (!Number.isFinite(job.pixelCount) || job.pixelCount <= 0) return;
  const donePixels = (job.pixelCount * clampedPct) / 100.0;
  const doneRays = (job.rayCount * clampedPct) / 100.0;
  const instPixPerSec = elapsedSafe > 0 ? (donePixels / elapsedSafe) : 0;
  const instRaysPerSec = elapsedSafe > 0 ? (doneRays / elapsedSafe) : 0;

  job.donePixels = donePixels;
  job.doneRays = doneRays;
  job.throughputPixPerSec = smoothValue(job.throughputPixPerSec, instPixPerSec, 0.2);
  job.throughputRaysPerSec = smoothValue(job.throughputRaysPerSec, instRaysPerSec, 0.2);
  job.etaSmoothedSec = smoothValue(job.etaSmoothedSec, etaSafe, 0.3);
}

function cloneTextArray(arr, maxItems) {
  if (!Array.isArray(arr)) return [];
  const cleaned = arr
    .map(v => String(v || '').trim())
    .filter(Boolean);
  if (!Number.isFinite(maxItems) || maxItems <= 0) return cleaned;
  return cleaned.slice(-maxItems);
}

function buildTimestampCompact() {
  return new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 15);
}

function makeGeoFilePath() {
  return path.join(OUT_DIR, `geo_${buildTimestampCompact()}.kgeo`);
}

function rewriteCloneArgs(kind, args) {
  if (!Array.isArray(args)) return [];
  let out = args.slice();
  if (kind === 'render' && !out.includes('--anim')) {
    out = stripArgWithValue(out, '--geo-file');
    out.push('--geo-file', makeGeoFilePath());
  }
  return out;
}

function serializeJobForState(job) {
  if (!job) return null;
  return {
    id: safeNumber(job.id, 0),
    kind: job.kind === 'colorize' ? 'colorize' : 'render',
    status: String(job.status || 'queued'),
    resolution: String(job.resolution || 'unknown'),
    backend: String(job.backend || 'cpu'),
    chart: String(job.chart || 'ks'),
    binary: typeof job.binary === 'string' ? job.binary : '',
    args: Array.isArray(job.args) ? job.args.map(v => String(v)) : [],
    createdAt: job.createdAt || nowIso(),
    startedAt: job.startedAt || null,
    finishedAt: job.finishedAt || null,
    progressPct: safeNumber(job.progressPct, 0),
    elapsedSec: safeNumber(job.elapsedSec, 0),
    etaSec: safeNumber(job.etaSec, 0),
    etaSmoothedSec: safeNumber(job.etaSmoothedSec, 0),
    throughputPixPerSec: safeNumber(job.throughputPixPerSec, 0),
    throughputRaysPerSec: safeNumber(job.throughputRaysPerSec, 0),
    pixelCount: safeNumber(job.pixelCount, 0),
    rayCount: safeNumber(job.rayCount, 0),
    cameraSpp: safeNumber(job.cameraSpp, 1),
    donePixels: safeNumber(job.donePixels, 0),
    doneRays: safeNumber(job.doneRays, 0),
    code: job.code ?? null,
    outputFile: job.outputFile || null,
    previewFile: job.previewFile || null,
    fallbackUsed: !!job.fallbackUsed,
    warnings: cloneTextArray(job.warnings, 40),
    logsTail: cloneTextArray(job.logsTail, 80),
  };
}

function hydrateJobFromState(raw, defaultStatus = 'queued') {
  if (!raw || typeof raw !== 'object') return null;
  const id = safeNumber(raw.id, 0);
  if (!Number.isFinite(id) || id <= 0) return null;
  const kind = raw.kind === 'colorize' ? 'colorize' : 'render';
  const backend = String(raw.backend || 'cpu');
  let binary = typeof raw.binary === 'string' ? raw.binary : '';
  if (!binary || !fs.existsSync(binary)) {
    binary = resolveBinary(backend);
  }
  if (!binary || !fs.existsSync(binary)) return null;

  const job = {
    id: Math.floor(id),
    kind,
    status: String(raw.status || defaultStatus),
    resolution: String(raw.resolution || 'unknown'),
    backend,
    chart: String(raw.chart || 'ks'),
    binary,
    args: Array.isArray(raw.args) ? raw.args.map(v => String(v)) : [],
    createdAt: raw.createdAt || nowIso(),
    startedAt: raw.startedAt || null,
    finishedAt: raw.finishedAt || null,
    progressPct: safeNumber(raw.progressPct, 0),
    elapsedSec: safeNumber(raw.elapsedSec, 0),
    etaSec: safeNumber(raw.etaSec, 0),
    etaSmoothedSec: safeNumber(raw.etaSmoothedSec, 0),
    throughputPixPerSec: safeNumber(raw.throughputPixPerSec, 0),
    throughputRaysPerSec: safeNumber(raw.throughputRaysPerSec, 0),
    pixelCount: safeNumber(raw.pixelCount, 0),
    rayCount: safeNumber(raw.rayCount, 0),
    cameraSpp: Math.max(1, Math.floor(safeNumber(raw.cameraSpp, 1))),
    donePixels: safeNumber(raw.donePixels, 0),
    doneRays: safeNumber(raw.doneRays, 0),
    code: raw.code ?? null,
    outputFile: raw.outputFile || null,
    previewFile: raw.previewFile || null,
    fallbackUsed: !!raw.fallbackUsed,
    warnings: cloneTextArray(raw.warnings, 40),
    logsTail: cloneTextArray(raw.logsTail, 80),
  };
  if (job.kind === 'render' && (!Number.isFinite(job.pixelCount) || job.pixelCount <= 0)) {
    attachRenderMetrics(job);
  }
  return job;
}

function persistQueueState() {
  const payload = {
    version: QUEUE_STATE_VERSION,
    savedAt: nowIso(),
    nextJobId,
    active: activeJob ? serializeJobForState(activeJob) : null,
    queued: queuedJobs.map(j => serializeJobForState(j)).filter(Boolean),
    recent: recentJobs.map(j => serializeJobForState(j)).filter(Boolean),
  };
  try {
    fs.writeFileSync(QUEUE_STATE_FILE, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  } catch (err) {
    console.warn(`[queue] cannot persist state: ${err.message}`);
  }
}

function loadQueueState() {
  if (!fs.existsSync(QUEUE_STATE_FILE)) return;
  try {
    const raw = JSON.parse(fs.readFileSync(QUEUE_STATE_FILE, 'utf8'));
    const queued = Array.isArray(raw?.queued) ? raw.queued.map(v => hydrateJobFromState(v, 'queued')).filter(Boolean) : [];
    const recent = Array.isArray(raw?.recent) ? raw.recent.map(v => hydrateJobFromState(v, 'done')).filter(Boolean) : [];
    const restoredActive = hydrateJobFromState(raw?.active, 'running');
    queuedJobs = queued.map(j => ({ ...j, status: 'queued' }));
    recentJobs = recent.slice(0, MAX_RECENT_JOBS);
    if (restoredActive) {
      restoredActive.status = 'cancelled';
      restoredActive.finishedAt = nowIso();
      restoredActive.code = null;
      restoredActive.warnings = [...(restoredActive.warnings || []), 'Recovered after restart: previous active job marked as cancelled'];
      rememberRecentJob(restoredActive);
    }
    const maxSeenId = [0, ...queuedJobs.map(j => j.id), ...recentJobs.map(j => j.id)].reduce((acc, v) => Math.max(acc, safeNumber(v, 0)), 0);
    const loadedNext = Math.floor(safeNumber(raw?.nextJobId, 1));
    nextJobId = Math.max(maxSeenId + 1, loadedNext, 1);
  } catch (err) {
    console.warn(`[queue] cannot load state file: ${err.message}`);
  }
}

function startProgressHeartbeat(job) {
  if (!job) return;
  job.startedAtSec = nowSeconds();
  job.hasRealProgress = false;
  job.heartbeat = setInterval(() => {
    if (!activeJob || activeJob !== job || job.status !== 'running') return;
    if (job.hasRealProgress) return;
    const elapsed = Math.max(0, nowSeconds() - job.startedAtSec);
    job.elapsedSec = elapsed;
    // Heartbeat progress: elapsed-only update (UI can show indeterminate bar).
    broadcast({
      type: 'progress',
      elapsed,
      etaSmoothed: job.etaSmoothedSec || 0,
      throughputPixPerSec: job.throughputPixPerSec || 0,
      throughputRaysPerSec: job.throughputRaysPerSec || 0,
      pixelCount: job.pixelCount || 0,
      rayCount: job.rayCount || 0,
      donePixels: job.donePixels || 0,
      doneRays: job.doneRays || 0,
      jobId: job.id,
    });
    broadcastQueueSnapshot();
  }, 1000);
}

function stopProgressHeartbeat(job) {
  if (!job || !job.heartbeat) return;
  clearInterval(job.heartbeat);
  job.heartbeat = null;
}

function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach(c => { if (c.readyState === WebSocket.OPEN) c.send(msg); });
}

function tailLines(text, keep = 80) {
  const lines = String(text || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  if (lines.length <= keep) return lines;
  return lines.slice(lines.length - keep);
}

function buildQueueSnapshot() {
  const serialize = (j, queueIndex = null) => ({
    id: j.id,
    kind: j.kind,
    status: j.status,
    resolution: j.resolution || 'unknown',
    backend: j.backend || 'cpu',
    chart: j.chart || 'ks',
    createdAt: j.createdAt,
    startedAt: j.startedAt || null,
    finishedAt: j.finishedAt || null,
    progressPct: Number.isFinite(j.progressPct) ? j.progressPct : 0,
    elapsedSec: Number.isFinite(j.elapsedSec) ? j.elapsedSec : 0,
    etaSec: Number.isFinite(j.etaSec) ? j.etaSec : 0,
    etaSmoothedSec: Number.isFinite(j.etaSmoothedSec) ? j.etaSmoothedSec : 0,
    throughputPixPerSec: Number.isFinite(j.throughputPixPerSec) ? j.throughputPixPerSec : 0,
    throughputRaysPerSec: Number.isFinite(j.throughputRaysPerSec) ? j.throughputRaysPerSec : 0,
    pixelCount: Number.isFinite(j.pixelCount) ? j.pixelCount : 0,
    rayCount: Number.isFinite(j.rayCount) ? j.rayCount : 0,
    cameraSpp: Number.isFinite(j.cameraSpp) ? j.cameraSpp : 1,
    donePixels: Number.isFinite(j.donePixels) ? j.donePixels : 0,
    doneRays: Number.isFinite(j.doneRays) ? j.doneRays : 0,
    code: j.code ?? null,
    outputFile: j.outputFile || null,
    previewFile: j.previewFile || null,
    queueIndex,
    fallbackUsed: !!j.fallbackUsed,
    warnings: Array.isArray(j.warnings) ? j.warnings.slice(-8) : [],
    logsTail: Array.isArray(j.logsTail) ? j.logsTail.slice(-8) : [],
  });

  return {
    active: activeJob ? serialize(activeJob) : null,
    queued: queuedJobs.map((j, idx) => serialize(j, idx + 1)),
    recent: recentJobs.map(j => serialize(j)),
  };
}

function broadcastQueueSnapshot() {
  broadcast({ type: 'queue_state', ...buildQueueSnapshot() });
}

function rememberRecentJob(job) {
  recentJobs.unshift({
    ...job,
    logsTail: Array.isArray(job.logsTail) ? job.logsTail.slice(-80) : [],
    warnings: Array.isArray(job.warnings) ? job.warnings.slice(-40) : [],
  });
  if (recentJobs.length > MAX_RECENT_JOBS) {
    recentJobs = recentJobs.slice(0, MAX_RECENT_JOBS);
  }
  persistQueueState();
}

function extractSavedFileFromStdoutChunk(chunkText) {
  const m = String(chunkText || '').match(/Saved:\s+(.+\.(png|mp4))/i);
  if (!m) return null;
  return path.basename(m[1].trim());
}

function markFallbackFromLine(job, line) {
  const txt = String(line || '').toLowerCase();
  if (!txt) return;
  if (txt.includes('fallback')) {
    job.fallbackUsed = true;
    if (!job.warnings) job.warnings = [];
    job.warnings.push(line.trim());
    job.warnings = job.warnings.slice(-40);
  }
}

function stripArgWithValue(args, flag, valueCount = 1) {
  const skip = Math.max(0, Math.floor(Number(valueCount) || 0));
  const out = [];
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === flag) {
      i += skip;
      continue;
    }
    out.push(args[i]);
  }
  return out;
}

function parseProgressFromChunk(job, chunkText) {
  if (!job || !chunkText) return;
  // Render progress is printed with carriage-return updates; chunks can split
  // in the middle of a progress line. Keep a rolling buffer and parse matches
  // across chunk boundaries.
  const prev = typeof job.stderrCarry === 'string' ? job.stderrCarry : '';
  const merged = `${prev}${String(chunkText)}`;
  job.stderrCarry = merged.slice(-1024);

  const progressRe = /\]\s+(\d+)%\s+([\d.]+)s elapsed,\s+([\d.]+)s ETA/g;
  let m;
  let last = null;
  while ((m = progressRe.exec(merged)) !== null) {
    last = m;
  }
  if (!last) return;

  job.hasRealProgress = true;
  updateDerivedProgress(
    job,
    parseInt(last[1], 10),
    parseFloat(last[2]),
    parseFloat(last[3])
  );
  broadcast({
    type: 'progress',
    pct: job.progressPct,
    elapsed: job.elapsedSec,
    eta: job.etaSec,
    etaSmoothed: job.etaSmoothedSec,
    throughputPixPerSec: job.throughputPixPerSec,
    throughputRaysPerSec: job.throughputRaysPerSec,
    pixelCount: job.pixelCount,
    rayCount: job.rayCount,
    donePixels: job.donePixels,
    doneRays: job.doneRays,
    jobId: job.id,
  });
  broadcastQueueSnapshot();
  persistQueueState();
}

function runAuxPreviewJob(job) {
  if (!job || job.kind !== 'render' || !job.binary || !Array.isArray(job.args)) return;
  if (job.args.includes('--anim')) return;

  let previewArgs = job.args.slice();
  ['--4k', '--2k', '--720p', '--hd', '--preview'].forEach(flag => {
    previewArgs = previewArgs.filter(a => a !== flag);
  });
  previewArgs = stripArgWithValue(previewArgs, '--custom-res', 2);
  previewArgs = stripArgWithValue(previewArgs, '--camera-spp');
  previewArgs = stripArgWithValue(previewArgs, '--geo-file');
  // Strip bundle-mode flags from the aux preview: it runs single-ray for instant feedback.
  previewArgs = previewArgs.filter(a => a !== '--bundles' && a !== '--anti-fireflies');
  previewArgs.push('--preview', '--camera-spp', '1');

  const p = spawn(job.binary, previewArgs, { cwd: ROOT });
  job.previewProc = p;
  p.on('close', () => {
    if (job.previewProc === p) job.previewProc = null;
  });
  p.stdout.on('data', chunk => {
    const line = chunk.toString();
    const saved = extractSavedFileFromStdoutChunk(line);
    if (saved) {
      const src = path.join(OUT_DIR, saved);
      const ext = path.extname(saved).toLowerCase() || '.png';
      const stableName = `job_${job.id}_preview${ext}`;
      const dst = path.join(THUMB_DIR, stableName);
      try {
        if (fs.existsSync(src)) {
          fs.copyFileSync(src, dst);
          fs.unlinkSync(src);
          job.previewFile = stableName;
        } else {
          job.previewFile = saved;
        }
      } catch {
        job.previewFile = saved;
      }
      broadcast({ type: 'job_preview', jobId: job.id, file: job.previewFile });
      broadcastQueueSnapshot();
      persistQueueState();
    }
  });
  p.on('error', () => {
    if (job.previewProc === p) job.previewProc = null;
    // Non-blocking helper process: ignore errors.
  });
}

function startNextQueuedJob() {
  if (activeJob || queuedJobs.length === 0) return;
  const job = queuedJobs.shift();
  if (!job) return;
  attachRenderMetrics(job);

  job.status = 'running';
  job.startedAt = new Date().toISOString();
  job.startedAtSec = nowSeconds();
  job.hasRealProgress = false;
  job.progressPct = 0;
  job.elapsedSec = 0;
  job.etaSec = 0;
  job.etaSmoothedSec = 0;
  job.throughputPixPerSec = 0;
  job.throughputRaysPerSec = 0;
  job.donePixels = 0;
  job.doneRays = 0;
  job.logsTail = [];
  job.warnings = [];
  job.fallbackUsed = false;
  job.stderrCarry = '';
  job.previewProc = null;

  const proc = spawn(job.binary, job.args, { cwd: ROOT });
  job.proc = proc;
  activeJob = job;
  startProgressHeartbeat(job);

  broadcast({ type: 'start', args: job.args, resolution: job.resolution, jobId: job.id });
  broadcastQueueSnapshot();
  persistQueueState();
  runAuxPreviewJob(job);

  proc.stdout.on('data', chunk => {
    const line = chunk.toString();
    parseProgressFromChunk(job, line);
    broadcast({ type: 'stdout', line, jobId: job.id });
    const saved = extractSavedFileFromStdoutChunk(line);
    if (saved) {
      job.outputFile = saved;
    }
    markFallbackFromLine(job, line);
    const tail = tailLines(line, 12);
    if (tail.length > 0) {
      job.logsTail = [...(job.logsTail || []), ...tail].slice(-80);
    }
  });

  proc.stderr.on('data', chunk => {
    const raw = chunk.toString();
    parseProgressFromChunk(job, raw);
    markFallbackFromLine(job, raw);
    const tail = tailLines(raw, 12);
    if (tail.length > 0) {
      job.logsTail = [...(job.logsTail || []), ...tail].slice(-80);
    }
  });

  proc.on('close', code => {
    stopProgressHeartbeat(job);
    const wasCancelled = !!job.cancelRequested;
    job.finishedAt = new Date().toISOString();
    job.code = code;
    job.status = wasCancelled ? 'cancelled' : (code === 0 ? 'done' : 'failed');
    if (job.previewProc && !job.previewProc.killed) {
      try { job.previewProc.kill('SIGTERM'); } catch {}
    }
    job.previewProc = null;
    if (!job.outputFile && code === 0) {
      job.outputFile = latestOutputFile();
    }
    rememberRecentJob(job);

    broadcast({ type: 'done', code, file: job.outputFile || null, jobId: job.id });
    activeJob = null;
    persistQueueState();
    broadcastQueueSnapshot();
    startNextQueuedJob();
  });
}

function enqueueJob(job) {
  const next = {
    ...job,
    id: nextJobId++,
    createdAt: new Date().toISOString(),
    status: 'queued',
    progressPct: 0,
    elapsedSec: 0,
    etaSec: 0,
    logsTail: [],
    warnings: [],
    fallbackUsed: false,
  };
  attachRenderMetrics(next);
  queuedJobs.push(next);
  persistQueueState();
  broadcastQueueSnapshot();
  startNextQueuedJob();
  return next;
}

function findRecentJobById(jobId) {
  return recentJobs.find(j => j.id === jobId) || null;
}

function enqueueFromRecentJobId(jobId) {
  const source = findRecentJobById(jobId);
  if (!source) {
    return { error: `Recent job ${jobId} not found`, code: 404 };
  }
  const backend = String(source.backend || 'cpu');
  const binary = (typeof source.binary === 'string' && fs.existsSync(source.binary))
    ? source.binary
    : resolveBinary(backend);
  if (!binary || !fs.existsSync(binary)) {
    return { error: `Binary unavailable for backend ${backend}`, code: 503 };
  }
  const args = rewriteCloneArgs(source.kind, source.args);
  if (!Array.isArray(args) || args.length === 0) {
    return { error: `Recent job ${jobId} has no reproducible arguments`, code: 400 };
  }
  const enqueued = enqueueJob({
    kind: source.kind === 'colorize' ? 'colorize' : 'render',
    binary,
    args,
    resolution: source.resolution || 'unknown',
    backend,
    chart: source.chart || 'ks',
    clonedFromId: source.id,
  });
  return { enqueued, source };
}

function latestOutputFile() {
  return fs.readdirSync(OUT_DIR)
    .filter(f => /\.(png|mp4)$/.test(f))
    .map(f => ({ f, t: fs.statSync(path.join(OUT_DIR, f)).mtime }))
    .sort((a, b) => b.t - a.t)[0]?.f ?? null;
}

function toBoundedInt(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(parsed)));
}

function parseDateYmd(value, endOfDay = false) {
  if (typeof value !== 'string') return null;
  const m = value.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return null;
  const year = Number(m[1]);
  const monthIdx = Number(m[2]) - 1;
  const day = Number(m[3]);
  const hour = endOfDay ? 23 : 0;
  const minute = endOfDay ? 59 : 0;
  const second = endOfDay ? 59 : 0;
  const ms = endOfDay ? 999 : 0;
  return new Date(Date.UTC(year, monthIdx, day, hour, minute, second, ms));
}

function parseRenderMeta(fileName) {
  const lower = String(fileName || '').toLowerCase();

  let resolution = 'unknown';
  if (lower.startsWith('4k_')) resolution = '4K';
  else if (lower.startsWith('2k_')) resolution = '2K';
  else if (lower.startsWith('1080p_')) resolution = '1080p';
  else if (lower.startsWith('720p_')) resolution = '720p';
  else if (lower.startsWith('512p_')) resolution = '512p';
  else if (lower.startsWith('480p_') || lower.startsWith('hd_')) resolution = '480p';
  else if (lower.startsWith('256p_')) resolution = '256p';
  else if (lower.startsWith('144p_')) resolution = '144p';
  else if (lower.startsWith('custom_')) resolution = 'custom';

  let backend = 'unknown';
  if (lower.includes('gpu-metal')) backend = 'metal';
  else if (lower.includes('gpu-cuda')) backend = 'cuda';
  else if (lower.includes('_cpu_') || lower.endsWith('_cpu.png') || lower.includes('_cpu-')) backend = 'cpu';

  let chart = 'unknown';
  if (lower.includes('_gks-') || lower.includes('_gks_')) chart = 'gks';
  else if (lower.includes('_ks-') || lower.includes('_ks_')) chart = 'ks';
  else if (lower.includes('_bl-') || lower.includes('_bl_')) chart = 'bl';

  let rayMode = 'single_ray';
  if (lower.includes('ray-bundle') || lower.includes('ray_bundle') || lower.includes('bundles')) {
    rayMode = 'ray_bundle';
  }

  let solver = 'standard';
  if (lower.includes('elliptic-closed') || lower.includes('elliptic_closed')) solver = 'elliptic_closed';
  else if (lower.includes('semi-analytic') || lower.includes('semi_analytic')) solver = 'semi_analytic';

  return { resolution, backend, chart, rayMode, solver };
}

function normalizeTypeToken(type) {
  return String(type || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_');
}

function normalizeResolutionToken(value) {
  const raw = String(value || '').trim();
  const lower = raw.toLowerCase();
  if (!raw || lower === 'all') return 'all';
  if (lower === '4k') return '4K';
  if (lower === '2k') return '2K';
  if (lower === '1080p') return '1080p';
  if (lower === '720p') return '720p';
  if (lower === '512p') return '512p';
  if (lower === '480p' || lower === 'hd') return '480p';
  if (lower === '256p') return '256p';
  if (lower === '144p') return '144p';
  if (lower === 'custom') return 'custom';
  return raw;
}

function matchesTypeFilter(meta, typeToken) {
  if (!typeToken || typeToken === 'all') return true;
  if (meta.backend === typeToken || meta.chart === typeToken || meta.rayMode === typeToken || meta.solver === typeToken) {
    return true;
  }
  if (typeToken === 'raybundle' || typeToken === 'bundle' || typeToken === 'raybundles') {
    return meta.rayMode === 'ray_bundle';
  }
  if (typeToken === 'single' || typeToken === 'single_ray' || typeToken === 'single_rays') {
    return meta.rayMode === 'single_ray';
  }
  if (typeToken === 'rk4' || typeToken === 'standard_rk' || typeToken === 'standard_rk4') {
    return meta.solver === 'standard';
  }
  return false;
}

function safeOutputFilePath(fileName) {
  const raw = String(fileName || '');
  const base = path.basename(raw);
  if (!base || base !== raw) return null;
  if (!/\.(png|jpg|jpeg|mp4)$/i.test(base)) return null;
  return path.join(OUT_DIR, base);
}

function ensureRenderThumbnail(sourceFilePath, sourceName, widthPx) {
  const parsed = path.parse(sourceName);
  const thumbName = `${parsed.name}_w${widthPx}.jpg`;
  const thumbPath = path.join(THUMB_DIR, thumbName);
  try {
    const srcStat = fs.statSync(sourceFilePath);
    if (fs.existsSync(thumbPath)) {
      const thumbStat = fs.statSync(thumbPath);
      if (thumbStat.mtimeMs >= srcStat.mtimeMs) {
        return thumbPath;
      }
    }
  } catch {
    return null;
  }

  if (process.platform === 'darwin') {
    const out = spawnSync(
      'sips',
      ['-s', 'format', 'jpeg', '-Z', String(widthPx), sourceFilePath, '--out', thumbPath],
      { stdio: 'ignore' }
    );
    if (out.status === 0 && fs.existsSync(thumbPath)) return thumbPath;
  }
  return null;
}

loadQueueState();

// ── Nav bake runner ───────────────────────────────────────────
async function runNavBake(job) {
  // CPU binary (OpenMP) is ~70× faster than Metal for small bake frames (160×90).
  const navBinary = BINARY_CPU || BINARY_METAL;
  if (!navBinary) {
    broadcast({ type: 'nav_bake_error', error: 'no binary available' });
    navBakeJob = null;
    return;
  }

  // Build static args (everything except --theta / --phi which go via stdin)
  const staticMsg = { ...job.renderParams, theta: 0, phi: 0 };
  const allArgs = buildNavArgs(staticMsg);
  // Remove --theta and --phi from args (sent per-frame via stdin instead)
  const thetaIdx = allArgs.indexOf('--theta');
  if (thetaIdx >= 0) allArgs.splice(thetaIdx, 2);
  const phiIdx = allArgs.indexOf('--phi');
  if (phiIdx >= 0) allArgs.splice(phiIdx, 2);
  allArgs.push('--bake-mode');

  const proc = spawn(navBinary, allArgs, { cwd: ROOT, stdio: ['pipe', 'pipe', 'ignore'] });
  job.currentProc = proc;

  let stdoutBuf = '';

  const processNextFrame = (frameIdx) => {
    return new Promise((resolve) => {
      if (frameIdx >= job.frames.length || job.cancelled) {
        proc.stdin.end();
        resolve();
        return;
      }
      const frame = job.frames[frameIdx];

      const onData = (chunk) => {
        stdoutBuf += chunk.toString();
        const lines = stdoutBuf.split('\n');
        stdoutBuf = lines.pop(); // keep incomplete line
        for (const line of lines) {
          const m = line.match(/Saved:\s+(\S+\.png)/i);
          if (m) {
            proc.stdout.removeListener('data', onData);
            const fullPath = path.isAbsolute(m[1]) ? m[1] : path.join(ROOT, m[1]);
            if (fs.existsSync(fullPath)) {
              try { fs.renameSync(fullPath, path.join(job.bakeDir, frame.file)); } catch {}
            }
            job.done++;
            broadcast({ type: 'nav_bake_progress', bakeId: job.id, done: job.done, total: job.total });
            resolve();
          }
        }
      };
      proc.stdout.on('data', onData);
      proc.stdin.write(`${frame.theta} ${frame.phi}\n`);
    });
  };

  // Process frames sequentially
  for (let i = 0; i < job.frames.length && !job.cancelled; i++) {
    await processNextFrame(i);
  }

  if (!proc.stdin.destroyed) proc.stdin.end();
  await new Promise(resolve => proc.on('close', resolve));

  navBakeJob = null;
  if (!job.cancelled) {
    broadcast({ type: 'nav_bake_done', bakeId: job.id, total: job.total });
  }
}

// ── Navigate helper ───────────────────────────────────────────
function buildNavArgs(msg) {
  // 320×180 for fast interactive preview
  const args = ['--custom-res', '320', '180', '--ks', '--solver-mode', 'standard',
                '--intersection-hermite'];
  const theta = Number(msg.theta);
  const phi   = Number(msg.phi);
  if (Number.isFinite(theta)) args.push('--theta', String(theta));
  if (Number.isFinite(phi))   args.push('--phi',   String(phi));
  const a = Number(msg.a);
  if (Number.isFinite(a))     args.push('--a', String(a));
  const diskOut = Number(msg.disk_out);
  if (Number.isFinite(diskOut) && diskOut > 0) args.push('--disk-out', String(diskOut));
  const rObs = Number(msg.r_obs);
  if (Number.isFinite(rObs) && rObs > 0) args.push('--r-obs', String(rObs));
  const fov = Number(msg.fov);
  if (Number.isFinite(fov) && fov > 0) args.push('--fov', String(fov));

  const palette = String(msg.disk_palette || 'blackbody');
  if (palette === 'interstellar') {
    args.push('--disk-interstellar');
    const p = Number(msg.interstellar_p);
    if (Number.isFinite(p)) args.push('--disk-interstellar-p', String(p));
  } else if (palette === 'stratified') {
    args.push('--disk-stratified');
  }

  const brightness = Number(msg.disk_brightness);
  if (Number.isFinite(brightness) && brightness > 0)
    args.push('--disk-brightness', String(brightness));

  if (msg.doppler_enabled === true)  args.push('--doppler');
  else if (msg.doppler_enabled === false) args.push('--no-doppler');

  if (msg.radial_term_zero_torque === true) args.push('--radial-term-zero-torque');

  const bg = typeof msg.background === 'string' ? msg.background.trim() : '';
  if (bg && bg !== 'none') {
    const bgPath = path.join(ASSETS_DIR, bg);
    if (fs.existsSync(bgPath)) args.push('--bg', bgPath);
  }
  return args;
}

// ── WebSocket ─────────────────────────────────────────────────
wss.on('connection', ws => {
  // Send current job status on connect
  if (activeJob) {
    ws.send(JSON.stringify({ type: 'status', running: true }));
  } else {
    ws.send(JSON.stringify({ type: 'status', running: false }));
  }
  ws.send(JSON.stringify({ type: 'queue_state', ...buildQueueSnapshot() }));

  // ── Per-connection navigate state ─────────────────────────
  let navProc    = null;
  let navDebounce = null;

  ws.on('message', raw => {
    let msg;
    try { msg = JSON.parse(String(raw)); } catch { return; }

    if (msg.type === 'navigate') {
      if (navDebounce) { clearTimeout(navDebounce); navDebounce = null; }
      if (navProc)     { try { navProc.kill('SIGTERM'); } catch {} navProc = null; }

      navDebounce = setTimeout(() => {
        navDebounce = null;
        const navBinary = BINARY_CPU || BINARY_METAL;
        if (!navBinary) {
          if (ws.readyState === WebSocket.OPEN)
            ws.send(JSON.stringify({ type: 'nav_error', error: 'no binary available' }));
          return;
        }

        const args = buildNavArgs(msg);
        const proc = spawn(navBinary, args, { cwd: ROOT });
        navProc = proc;

        let savedFile = null;
        proc.stdout.on('data', chunk => {
          const m = chunk.toString().match(/Saved:\s+(\S+\.png)/i);
          if (m) savedFile = m[1].trim();
        });

        proc.on('close', code => {
          if (navProc === proc) navProc = null;
          if (code !== 0 || !savedFile) return;
          const fullPath = path.isAbsolute(savedFile)
            ? savedFile : path.join(ROOT, savedFile);
          if (!fs.existsSync(fullPath)) return;
          try {
            const buf = fs.readFileSync(fullPath);
            const dataUrl = 'data:image/png;base64,' + buf.toString('base64');
            if (ws.readyState === WebSocket.OPEN)
              ws.send(JSON.stringify({ type: 'nav_frame', dataUrl }));
            fs.unlinkSync(fullPath);
          } catch (err) {
            console.warn('[nav] read/send error:', err.message);
          }
        });

        proc.on('error', err => {
          if (navProc === proc) navProc = null;
          console.warn('[nav] spawn error:', err.message);
        });
      }, 50);
    }
  });

  ws.on('close', () => {
    if (navDebounce) { clearTimeout(navDebounce); navDebounce = null; }
    if (navProc) { try { navProc.kill('SIGTERM'); } catch {} navProc = null; }
  });
});

// ── GET /api/info ─────────────────────────────────────────────
app.get('/api/info', (req, res) => {
  const backgrounds = fs.readdirSync(ASSETS_DIR)
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    .sort((a, b) => {
      if (a === DEFAULT_BACKGROUND) return -1;
      if (b === DEFAULT_BACKGROUND) return 1;
      return a.localeCompare(b);
    });

  res.json({
    resolutions: Object.keys(RESOLUTIONS),
    resolutionSizes: RESOLUTIONS,
    backgrounds,
    backends: availableBackends(),
  });
});

// ── GET /api/status ───────────────────────────────────────────
app.get('/api/status', (req, res) => {
  res.json({
    running: !!activeJob,
    startedAtSec: activeJob?.startedAtSec ?? null,
    lastFile: latestOutputFile(),
    activeJobId: activeJob?.id ?? null,
    queuedCount: queuedJobs.length,
    recentCount: recentJobs.length,
  });
});

// ── GET /api/queue ────────────────────────────────────────────
app.get('/api/queue', (req, res) => {
  res.json(buildQueueSnapshot());
});

// ── GET /api/renders ─────────────────────────────────────────
app.get('/api/renders', (req, res) => {
  const q = typeof req.query.q === 'string' ? req.query.q.trim().toLowerCase() : '';
  const resolutionFilter = normalizeResolutionToken(req.query.resolution);
  const backendFilter = typeof req.query.backend === 'string'
    ? req.query.backend.trim().toLowerCase()
    : 'all';
  const chartFilter = typeof req.query.chart === 'string'
    ? req.query.chart.trim().toLowerCase()
    : 'all';
  const typeToken = normalizeTypeToken(req.query.type);
  const fromDate = parseDateYmd(req.query.from, false);
  const toDate = parseDateYmd(req.query.to, true);
  const limit = toBoundedInt(req.query.limit, 10, 1, 2000);
  const page = toBoundedInt(req.query.page, 1, 1, 1000000);
  const pageSize = toBoundedInt(req.query.page_size ?? req.query.pageSize, limit, 1, 200);
  const includeTotal = String(req.query.include_total ?? req.query.includeTotal ?? '') === '1'
    || req.query.page !== undefined
    || req.query.page_size !== undefined
    || req.query.pageSize !== undefined;

  let files = fs.readdirSync(OUT_DIR)
    .filter(f => /\.png$/.test(f))
    .map(f => {
      const stat = fs.statSync(path.join(OUT_DIR, f));
      return {
        name: f,
        size: stat.size,
        mtime: stat.mtime.toISOString(),
        mtimeMs: stat.mtime.getTime(),
        meta: parseRenderMeta(f),
      };
    })
    .sort((a, b) => b.mtimeMs - a.mtimeMs);

  files = files.filter(f => {
    if (q && !f.name.toLowerCase().includes(q)) return false;
    if (resolutionFilter !== 'all') {
      if (resolutionFilter === 'custom') {
        if (f.meta.resolution !== 'custom') return false;
      } else if (f.meta.resolution !== resolutionFilter) {
        return false;
      }
    }
    if (backendFilter !== 'all' && f.meta.backend !== backendFilter) return false;
    if (chartFilter !== 'all' && f.meta.chart !== chartFilter) return false;
    if (!matchesTypeFilter(f.meta, typeToken)) return false;
    if (fromDate && f.mtimeMs < fromDate.getTime()) return false;
    if (toDate && f.mtimeMs > toDate.getTime()) return false;
    return true;
  });

  if (!includeTotal) {
    files = files.slice(0, limit);
    const payload = files.map(({ mtimeMs, ...rest }) => rest);
    return res.json(payload);
  }

  const total = files.length;
  const start = (page - 1) * pageSize;
  const items = files.slice(start, start + pageSize).map(({ mtimeMs, ...rest }) => rest);
  return res.json({
    items,
    total,
    page,
    pageSize,
    totalPages: Math.max(1, Math.ceil(total / pageSize)),
    hasNext: start + pageSize < total,
    hasPrev: page > 1,
  });
});

// ── GET /api/renders/:file ────────────────────────────────────
app.get('/api/renders/:file', (req, res) => {
  const file = safeOutputFilePath(req.params.file);
  if (!file || !fs.existsSync(file)) return res.sendStatus(404);
  res.sendFile(file);
});

// ── GET /api/renders-thumb/:file ─────────────────────────────
app.get('/api/renders-thumb/:file', (req, res) => {
  const source = safeOutputFilePath(req.params.file);
  if (!source || !fs.existsSync(source)) return res.sendStatus(404);
  const sourceName = path.basename(source);
  const wRaw = Number(req.query.w);
  const width = Number.isFinite(wRaw) ? Math.max(64, Math.min(1024, Math.floor(wRaw))) : 360;
  const thumb = ensureRenderThumbnail(source, sourceName, width);
  // Short cache: thumbnails are regenerated when source mtime changes.
  res.setHeader('Cache-Control', 'public, max-age=60');
  if (thumb && fs.existsSync(thumb)) return res.sendFile(thumb);
  // Fallback path (e.g. non-macOS): serve original render.
  return res.sendFile(source);
});

// ── GET /api/live-preview/:file ───────────────────────────────
app.get('/api/live-preview/:file', (req, res) => {
  const base = path.basename(String(req.params.file || ''));
  if (!base || base !== req.params.file) return res.sendStatus(404);
  if (!/\.(png|jpg|jpeg)$/i.test(base)) return res.sendStatus(404);
  const file = path.join(THUMB_DIR, base);
  if (!fs.existsSync(file)) return res.sendStatus(404);
  res.setHeader('Cache-Control', 'no-cache');
  return res.sendFile(file);
});

// ── GET /api/geo-files ────────────────────────────────────────
app.get('/api/geo-files', (req, res) => {
  const files = fs.readdirSync(OUT_DIR)
    .filter(f => f.endsWith('.kgeo'))
    .map(f => {
      const stat = fs.statSync(path.join(OUT_DIR, f));
      return { name: f, size: stat.size, mtime: stat.mtime };
    })
    .sort((a, b) => b.mtime - a.mtime);
  res.json(files);
});

// ── POST /api/colorize ────────────────────────────────────────
app.post('/api/colorize', (req, res) => {
  const { geoFile, exposure, gamma, tempScale } = req.body;
  if (!geoFile) return res.status(400).json({ error: 'geoFile required' });

  const geoPath = path.join(OUT_DIR, geoFile);
  if (!fs.existsSync(geoPath)) return res.status(404).json({ error: 'geo file not found' });

  const args = ['--color-only', geoPath];
  if (exposure  !== undefined) args.push('--exposure',   String(exposure));
  if (gamma     !== undefined) args.push('--gamma',      String(gamma));
  if (tempScale !== undefined) args.push('--temp-scale', String(tempScale));

  const binary = resolveBinary('cpu');
  if (!binary || !fs.existsSync(binary)) {
    return res.status(503).json({ error: 'No renderer binary found for colorize' });
  }
  const enqueued = enqueueJob({
    kind: 'colorize',
    binary,
    args,
    resolution: 'recolor',
    backend: 'cpu',
    chart: 'ks',
  });
  const queuePosition = activeJob && activeJob.id === enqueued.id
    ? 0
    : queuedJobs.findIndex(j => j.id === enqueued.id) + 1;
  res.json({
    status: queuePosition === 0 ? 'started' : 'queued',
    jobId: enqueued.id,
    queuePosition,
    args,
  });
});

// ── POST /api/render ──────────────────────────────────────────
app.post('/api/render', (req, res) => {
  const p = req.body;
  const binary = resolveBinary(p.backend || 'cpu');
  if (!fs.existsSync(binary)) return res.status(503).json({ error: `Binary not found: ${binary}` });

  // Build argv
  const args = [];

  // Resolution
  const res_key = p.resolution || '720p';
  const dim = RESOLUTIONS[res_key] || RESOLUTIONS['720p'];

  // Map resolution to flags
  if      (res_key === '480p')  args.push('--hd');
  else if (res_key === '720p')  args.push('--720p');
  else if (res_key === '144p' || res_key === '256p' || res_key === '512p') {
    args.push('--custom-res', String(dim.w), String(dim.h));
  } else if (res_key === '1080p') { /* default */ }
  else if (res_key === '2K')    args.push('--2k');
  else if (res_key === '4K')    args.push('--4k');

  if (p.bundles)  args.push('--bundles');
  if (p.anti_fireflies) args.push('--anti-fireflies');
  if (p.dopri5)   args.push('--dopri5');
  if (p.gpu_fp64) args.push('--gpu-fp64');
  if (p.max_steps !== undefined) {
    const v = Number(p.max_steps);
    if (Number.isFinite(v)) args.push('--max-steps', String(Math.max(1, Math.floor(v))));
  }
  if (p.step_init !== undefined) {
    const v = Number(p.step_init);
    if (Number.isFinite(v)) args.push('--step-init', String(Math.max(1e-10, v)));
  }
  if (p.integrator_tol !== undefined) {
    const v = Number(p.integrator_tol);
    if (Number.isFinite(v)) args.push('--integrator-tol', String(Math.max(1e-10, v)));
  }
  if (p.camera_spp !== undefined) {
    const v = Number(p.camera_spp);
    if (Number.isFinite(v)) args.push('--camera-spp', String(Math.max(1, Math.floor(v))));
  }
  let solverMode = 'standard';
  if (typeof p.solver_mode === 'string') {
    const m = p.solver_mode.toLowerCase();
    if (m === 'semi' || m === 'semi-analytic' || m === 'semi_analytic') solverMode = 'semi-analytic';
    else if (m === 'elliptic' || m === 'elliptic-closed' || m === 'elliptic_closed') solverMode = 'elliptic-closed';
  } else if (p.semi_analytic) {
    // Backward compatibility with older frontend payloads.
    solverMode = 'semi-analytic';
  }
  args.push('--solver-mode', solverMode);
  if (p.integration_chart === 'bl') args.push('--bl');
  else if (p.integration_chart === 'gks') args.push('--gks');
  else args.push('--ks');
  // Keep disk-ray intersection on Hermite by default for smoother event localization.
  args.push('--intersection-hermite');

  if (p.a      !== undefined) args.push('--a',       String(p.a));
  args.push('--disk-out', String(p.disk_out !== undefined ? p.disk_out : DEFAULT_DISK_OUT));
  if (p.theta  !== undefined) args.push('--theta',   String(p.theta));
  args.push('--r-obs', String(p.r_obs !== undefined ? p.r_obs : DEFAULT_R_OBS));
  if (p.q      !== undefined && p.q  !== 0) args.push('--charge', String(p.q));
  if (p.lambda !== undefined && p.lambda !== 0) args.push('--lambda', String(p.lambda));
  if (p.fov    !== undefined && p.fov !== 30) args.push('--fov', String(p.fov));
  if (p.phi    !== undefined && p.phi !== 0)  args.push('--phi',    String(p.phi));

  // Animation flags
  if (p.anim) {
    args.push('--anim');
    if (p.anim_frames)   args.push('--frames',   String(p.anim_frames));
    if (p.anim_fps)      args.push('--fps',       String(p.anim_fps));
    if (p.anim_crf)      args.push('--crf',       String(p.anim_crf));
    if (p.anim_orbits)   args.push('--orbits',    String(p.anim_orbits));
    if (p.anim_ease)     args.push('--ease');
    if (p.anim_theta_start !== undefined) args.push('--theta-start',    String(p.anim_theta_start));
    if (p.anim_theta_end   !== undefined) args.push('--theta-end',      String(p.anim_theta_end));
    if (p.anim_phi_start   !== undefined) args.push('--phi-start',      String(p.anim_phi_start));
    if (p.anim_phi_end     !== undefined) args.push('--phi-end',        String(p.anim_phi_end));
    if (p.anim_r_start     !== undefined) args.push('--r-start',        String(p.anim_r_start));
    if (p.anim_r_end       !== undefined) args.push('--r-end',          String(p.anim_r_end));
    if (p.anim_a_start     !== undefined) args.push('--a-start',        String(p.anim_a_start));
    if (p.anim_a_end       !== undefined) args.push('--a-end',          String(p.anim_a_end));
    if (p.anim_disk_start  !== undefined) args.push('--disk-out-start', String(p.anim_disk_start));
    if (p.anim_disk_end    !== undefined) args.push('--disk-out-end',   String(p.anim_disk_end));
  }

  const requestedBg = (typeof p.background === 'string' && p.background.trim().length > 0 && p.background.trim() !== 'none')
    ? p.background.trim()
    : null;
  if (requestedBg) {
    const bgPath = path.join(ASSETS_DIR, requestedBg);
    if (fs.existsSync(bgPath)) {
      args.push('--bg', bgPath);
    } else {
      console.warn(`[render] background not found: ${requestedBg}`);
    }
  }

  if (p.disk_brightness !== undefined) {
    const v = Number(p.disk_brightness);
    if (Number.isFinite(v)) args.push('--disk-brightness', String(Math.max(0, v)));
  }
  if (p.disk_opacity !== undefined) {
    const v = Number(p.disk_opacity);
    if (Number.isFinite(v)) args.push('--disk-opacity', String(Math.max(0, Math.min(1, v))));
  }
  if (p.doppler_enabled === false) {
    args.push('--no-doppler');
  } else if (p.doppler_enabled === true) {
    args.push('--doppler');
  }
  if (p.disk_inner_emission_floor !== undefined) {
    const v = Number(p.disk_inner_emission_floor);
    if (Number.isFinite(v)) args.push('--disk-inner-emission-floor', String(Math.max(0, v)));
  }
  if (p.disk_inner_emission_floor_width !== undefined) {
    const v = Number(p.disk_inner_emission_floor_width);
    if (Number.isFinite(v)) args.push('--disk-inner-emission-floor-width', String(Math.max(1e-4, v)));
  }
  if (p.radial_term_zero_torque === false) {
    args.push('--no-radial-term-zero-torque');
  } else if (p.radial_term_zero_torque === true) {
    args.push('--radial-term-zero-torque');
  }
  if (p.radial_term_r3_decay === false) {
    args.push('--no-radial-term-r3');
  } else if (p.radial_term_r3_decay === true) {
    args.push('--radial-term-r3');
  }
  if (p.radial_term_relativistic === false) {
    args.push('--no-radial-term-relativistic');
  } else if (p.radial_term_relativistic === true) {
    args.push('--radial-term-relativistic');
  }
  if (p.radial_term_b_denom === false) {
    args.push('--no-radial-term-b-denom');
  } else if (p.radial_term_b_denom === true) {
    args.push('--radial-term-b-denom');
  }

  // Disk palette
  const radialProfileMode =
    typeof p.disk_radial_profile === 'string' ? p.disk_radial_profile : 'page_thorne';
  if (radialProfileMode === 'physical_nt' || radialProfileMode === 'physical-nt' || radialProfileMode === 'nt') {
    args.push('--disk-radial-profile', 'physical_nt');
  } else {
    args.push('--disk-radial-profile', 'page_thorne');
  }

  const paletteMode = typeof p.disk_palette === 'string' ? p.disk_palette : 'blackbody';
  if (paletteMode === 'stratified') {
    args.push('--disk-stratified');
    if (p.disk_rings !== undefined) {
      const v = Math.min(32, Math.max(1, Math.floor(Number(p.disk_rings))));
      args.push('--disk-rings', String(v));
    }
    if (p.disk_sectors !== undefined) {
      const v = Math.min(256, Math.max(1, Math.floor(Number(p.disk_sectors))));
      args.push('--disk-sectors', String(v));
    }
    if (p.disk_sigma !== undefined) {
      const v = Math.max(0.01, Number(p.disk_sigma));
      args.push('--disk-sigma', String(v));
    }
    if (p.disk_hue_offset !== undefined) {
      const v = Number(p.disk_hue_offset);
      if (Number.isFinite(v)) args.push('--disk-hue-offset', String(v));
    }
  } else if (paletteMode === 'interstellar') {
    args.push('--disk-interstellar');
    if (p.interstellar_omega0 !== undefined) {
      const v = Number(p.interstellar_omega0);
      if (Number.isFinite(v)) args.push('--disk-interstellar-omega0', String(v));
    }
    if (p.interstellar_p !== undefined) {
      const v = Number(p.interstellar_p);
      if (Number.isFinite(v)) args.push('--disk-interstellar-p', String(v));
    }
    // Artistic exponential decay: opt-in, physically unfounded. The falloff value
    // is only meaningful when it is enabled, so only forward it then.
    if (p.interstellar_inner_glow === true) {
      args.push('--disk-interstellar-inner-glow');
      const v = Number(p.interstellar_inner_falloff_scale);
      if (Number.isFinite(v)) args.push('--disk-interstellar-inner-falloff', String(v));
    }
    if (p.interstellar_band_strength !== undefined) {
      const v = Number(p.interstellar_band_strength);
      if (Number.isFinite(v)) args.push('--disk-interstellar-band-strength', String(v));
    }
    if (p.interstellar_band_frequency !== undefined) {
      const v = Number(p.interstellar_band_frequency);
      if (Number.isFinite(v)) args.push('--disk-interstellar-band-frequency', String(v));
    }
    if (p.interstellar_band_warp !== undefined) {
      const v = Number(p.interstellar_band_warp);
      if (Number.isFinite(v)) args.push('--disk-interstellar-band-warp', String(v));
    }
    if (p.interstellar_turbulence_strength !== undefined) {
      const v = Number(p.interstellar_turbulence_strength);
      if (Number.isFinite(v)) args.push('--disk-interstellar-turbulence', String(v));
    }
    if (p.interstellar_hdr_intensity !== undefined) {
      const v = Number(p.interstellar_hdr_intensity);
      if (Number.isFinite(v)) args.push('--disk-interstellar-hdr', String(v));
    }
    if (p.interstellar_softness_in_scale !== undefined) {
      const v = Number(p.interstellar_softness_in_scale);
      if (Number.isFinite(v)) args.push('--disk-interstellar-softness-in', String(v));
    }
    if (p.interstellar_softness_out_scale !== undefined) {
      const v = Number(p.interstellar_softness_out_scale);
      if (Number.isFinite(v)) args.push('--disk-interstellar-softness-out', String(v));
    }
    if (p.interstellar_edge_transparency !== undefined) {
      const v = Number(p.interstellar_edge_transparency);
      if (Number.isFinite(v)) args.push('--disk-interstellar-edge-transparency', String(v));
    }
    if (p.interstellar_outer_r !== undefined) {
      const v = Number(p.interstellar_outer_r);
      if (Number.isFinite(v)) args.push('--disk-interstellar-outer-r', String(v));
    }
    if (p.interstellar_outer_g !== undefined) {
      const v = Number(p.interstellar_outer_g);
      if (Number.isFinite(v)) args.push('--disk-interstellar-outer-g', String(v));
    }
    if (p.interstellar_outer_b !== undefined) {
      const v = Number(p.interstellar_outer_b);
      if (Number.isFinite(v)) args.push('--disk-interstellar-outer-b', String(v));
    }
    if (p.interstellar_time !== undefined) {
      const v = Number(p.interstellar_time);
      if (Number.isFinite(v)) args.push('--disk-interstellar-time', String(v));
    }
  } else {
    args.push('--disk-blackbody');
  }

  // Wormhole (DNEG metric)
  if (p.wormhole) {
    args.push('--wormhole');
    if (p.wh_rho     !== undefined) args.push('--wh-throat',  String(p.wh_rho));
    if (p.wh_M_lens  !== undefined) args.push('--wh-lensing', String(p.wh_M_lens));
    if (p.wh_a_tunnel !== undefined) args.push('--wh-tunnel',  String(p.wh_a_tunnel));
    if (typeof p.bg_b === 'string' && p.bg_b.trim().length > 0) {
      const bgBPath = path.join(ASSETS_DIR, p.bg_b.trim());
      if (fs.existsSync(bgBPath)) {
        args.push('--bg-b', bgBPath);
      } else {
        console.warn(`[render] --bg-b not found: ${p.bg_b}`);
      }
    }
  }

  // Always save geodesic cache alongside every render
  if (!p.anim) {
    const ts = new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 15);
    const geoPath = path.join(OUT_DIR, `geo_${ts}.kgeo`);
    args.push('--geo-file', geoPath);
  }

  const enqueued = enqueueJob({
    kind: 'render',
    binary,
    args,
    resolution: res_key,
    backend: String(p.backend || 'cpu'),
    chart: p.integration_chart === 'bl' ? 'bl' : (p.integration_chart === 'gks' ? 'gks' : 'ks'),
  });
  const queuePosition = activeJob && activeJob.id === enqueued.id
    ? 0
    : queuedJobs.findIndex(j => j.id === enqueued.id) + 1;
  res.json({
    status: queuePosition === 0 ? 'started' : 'queued',
    jobId: enqueued.id,
    queuePosition,
    args,
  });
});

// ── POST /api/cancel ─────────────────────────────────────────
app.post('/api/cancel', (req, res) => {
  const wantedId = Number(req.body?.jobId);
  const hasId = Number.isFinite(wantedId);

  if (hasId) {
    if (activeJob && activeJob.id === wantedId) {
      activeJob.cancelRequested = true;
      activeJob.proc?.kill('SIGTERM');
      if (activeJob.previewProc && !activeJob.previewProc.killed) {
        try { activeJob.previewProc.kill('SIGTERM'); } catch {}
      }
      activeJob.previewProc = null;
      persistQueueState();
      return res.json({ status: 'cancelling', jobId: wantedId, active: true });
    }
    const idx = queuedJobs.findIndex(j => j.id === wantedId);
    if (idx >= 0) {
      const [removed] = queuedJobs.splice(idx, 1);
      removed.status = 'cancelled';
      removed.finishedAt = new Date().toISOString();
      removed.code = null;
      rememberRecentJob(removed);
      persistQueueState();
      broadcastQueueSnapshot();
      return res.json({ status: 'cancelled', jobId: wantedId, active: false });
    }
    return res.status(404).json({ error: `Job ${wantedId} not found` });
  }

  if (activeJob) {
    activeJob.cancelRequested = true;
    activeJob.proc?.kill('SIGTERM');
    if (activeJob.previewProc && !activeJob.previewProc.killed) {
      try { activeJob.previewProc.kill('SIGTERM'); } catch {}
    }
    activeJob.previewProc = null;
    persistQueueState();
    return res.json({ status: 'cancelling', jobId: activeJob.id, active: true });
  }
  if (queuedJobs.length > 0) {
    const removed = queuedJobs.shift();
    removed.status = 'cancelled';
    removed.finishedAt = new Date().toISOString();
    removed.code = null;
    rememberRecentJob(removed);
    persistQueueState();
    broadcastQueueSnapshot();
    return res.json({ status: 'cancelled', jobId: removed.id, active: false });
  }
  return res.status(404).json({ error: 'No active or queued jobs' });
});

// ── POST /api/queue/reorder ───────────────────────────────────
app.post('/api/queue/reorder', (req, res) => {
  const jobId = Number(req.body?.jobId);
  const direction = String(req.body?.direction || '').toLowerCase();
  if (!Number.isFinite(jobId)) {
    return res.status(400).json({ error: 'jobId required' });
  }
  if (direction !== 'up' && direction !== 'down') {
    return res.status(400).json({ error: 'direction must be up or down' });
  }
  const idx = queuedJobs.findIndex(j => j.id === jobId);
  if (idx < 0) return res.status(404).json({ error: `Job ${jobId} not found in queue` });
  const newIdx = direction === 'up' ? idx - 1 : idx + 1;
  if (newIdx < 0 || newIdx >= queuedJobs.length) {
    return res.json({ status: 'noop', jobId, direction });
  }
  const tmp = queuedJobs[idx];
  queuedJobs[idx] = queuedJobs[newIdx];
  queuedJobs[newIdx] = tmp;
  persistQueueState();
  broadcastQueueSnapshot();
  return res.json({ status: 'ok', jobId, direction });
});

// ── POST /api/queue/retry ─────────────────────────────────────
app.post('/api/queue/retry', (req, res) => {
  const jobId = Number(req.body?.jobId);
  if (!Number.isFinite(jobId)) {
    return res.status(400).json({ error: 'jobId required' });
  }
  const result = enqueueFromRecentJobId(jobId);
  if (result.error) {
    return res.status(result.code || 400).json({ error: result.error });
  }
  const enqueued = result.enqueued;
  const queuePosition = activeJob && activeJob.id === enqueued.id
    ? 0
    : queuedJobs.findIndex(j => j.id === enqueued.id) + 1;
  return res.json({
    status: queuePosition === 0 ? 'started' : 'queued',
    action: 'retry',
    sourceJobId: jobId,
    jobId: enqueued.id,
    queuePosition,
  });
});

// ── POST /api/queue/duplicate ─────────────────────────────────
app.post('/api/queue/duplicate', (req, res) => {
  const jobId = Number(req.body?.jobId);
  if (!Number.isFinite(jobId)) {
    return res.status(400).json({ error: 'jobId required' });
  }
  const result = enqueueFromRecentJobId(jobId);
  if (result.error) {
    return res.status(result.code || 400).json({ error: result.error });
  }
  const enqueued = result.enqueued;
  const queuePosition = activeJob && activeJob.id === enqueued.id
    ? 0
    : queuedJobs.findIndex(j => j.id === enqueued.id) + 1;
  return res.json({
    status: queuePosition === 0 ? 'started' : 'queued',
    action: 'duplicate',
    sourceJobId: jobId,
    jobId: enqueued.id,
    queuePosition,
  });
});

// ── GET /api/nav-bakes ────────────────────────────────────────
app.get('/api/nav-bakes', (req, res) => {
  const active = navBakeJob
    ? { id: navBakeJob.id, done: navBakeJob.done, total: navBakeJob.total, running: true }
    : null;
  if (!fs.existsSync(NAV_BAKES_DIR)) return res.json({ active, bakes: [] });
  const bakes = fs.readdirSync(NAV_BAKES_DIR)
    .filter(d => fs.statSync(path.join(NAV_BAKES_DIR, d)).isDirectory())
    .map(d => {
      const mf = path.join(NAV_BAKES_DIR, d, 'manifest.json');
      if (!fs.existsSync(mf)) return null;
      try {
        const m = JSON.parse(fs.readFileSync(mf, 'utf8'));
        return { id: d, thetaStep: m.thetaStep, phiStep: m.phiStep, total: m.frames.length, createdAt: m.createdAt };
      } catch { return null; }
    })
    .filter(Boolean)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  res.json({ active, bakes });
});

// ── POST /api/nav-bake ────────────────────────────────────────
app.post('/api/nav-bake', (req, res) => {
  if (req.body.cancel) {
    if (navBakeJob) {
      navBakeJob.cancelled = true;
      if (navBakeJob.currentProc) {
        try { navBakeJob.currentProc.stdin.end(); } catch {}
        try { navBakeJob.currentProc.kill('SIGTERM'); } catch {}
      }
      navBakeJob = null;
      broadcast({ type: 'nav_bake_cancelled' });
    }
    return res.json({ status: 'cancelled' });
  }
  if (navBakeJob) return res.status(409).json({ error: 'bake already running' });

  const thetaStep = Math.max(1, Math.min(30, Number(req.body.theta_step) || 10));
  const phiStep   = Math.max(1, Math.min(30, Number(req.body.phi_step)   || 10));
  const bakeId    = Date.now().toString(36);
  const bakeDir   = path.join(NAV_BAKES_DIR, bakeId);
  fs.mkdirSync(bakeDir, { recursive: true });

  const thetaVals = [], phiVals = [];
  for (let t = thetaStep; t < 180; t += thetaStep) thetaVals.push(t);
  for (let p = 0; p < 360; p += phiStep) phiVals.push(p);
  const frames = [];
  for (const theta of thetaVals)
    for (const phi of phiVals)
      frames.push({ theta, phi, file: `t${theta}_p${phi}.png` });

  const renderParams = {
    a: req.body.a, disk_out: req.body.disk_out, r_obs: req.body.r_obs,
    fov: req.body.fov, disk_palette: req.body.disk_palette,
    disk_brightness: req.body.disk_brightness, doppler_enabled: req.body.doppler_enabled,
    radial_term_zero_torque: req.body.radial_term_zero_torque,
    interstellar_p: req.body.interstellar_p, background: req.body.background,
  };

  fs.writeFileSync(path.join(bakeDir, 'manifest.json'), JSON.stringify({
    thetaStep, phiStep, frames, renderParams, createdAt: new Date().toISOString()
  }));

  navBakeJob = { id: bakeId, bakeDir, frames, renderParams, done: 0, total: frames.length, cancelled: false, currentProc: null };
  res.json({ status: 'started', bakeId, total: frames.length });
  runNavBake(navBakeJob);
});

// ── DELETE /api/nav-bake/:id ──────────────────────────────────
app.delete('/api/nav-bake/:id', (req, res) => {
  const id = req.params.id.replace(/[^a-z0-9]/gi, '');
  const bakeDir = path.join(NAV_BAKES_DIR, id);
  if (!fs.existsSync(bakeDir)) return res.status(404).json({ error: 'not found' });
  fs.rmSync(bakeDir, { recursive: true, force: true });
  res.json({ status: 'deleted' });
});

// ── GET /api/nav-bakes/:id/:file ──────────────────────────────
app.get('/api/nav-bakes/:id/:file', (req, res) => {
  const id   = req.params.id.replace(/[^a-z0-9]/gi, '');
  const file = req.params.file.replace(/[^a-z0-9._]/gi, '');
  const full = path.join(NAV_BAKES_DIR, id, file);
  if (!fs.existsSync(full)) return res.status(404).end();
  res.sendFile(full);
});

// ── Start ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`KNdS Render Server → http://localhost:${PORT}`);
  console.log(`Backends: ${availableBackends().join(', ')}`);
  console.log(`CPU:   ${BINARY_CPU ?? 'not found'}`);
  console.log(`Metal: ${BINARY_METAL ?? 'not found'}`);
  if (BINARY_METAL_LEGACY && fs.existsSync(BINARY_METAL_LEGACY) && BINARY_METAL !== BINARY_METAL_LEGACY) {
    console.log(`Metal legacy (ignored): ${BINARY_METAL_LEGACY}`);
  }
  console.log(`Output: ${OUT_DIR}`);
});
