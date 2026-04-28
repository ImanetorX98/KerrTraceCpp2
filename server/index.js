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
const { spawn } = require('child_process');

const app    = express();
const server = http.createServer(app);
const wss    = new WebSocket.Server({ server, path: '/ws' });

const ROOT         = path.resolve(__dirname, '..');
const BINARY_MAIN  = path.join(ROOT, 'build', 'kerr_tracer');
const BINARY_CPU_ONLY = path.join(ROOT, 'build_cpu', 'kerr_tracer');
const BINARY_METAL_LEGACY = path.join(ROOT, 'build', 'kerr_tracer_metal');
const BINARY_CUDA  = path.join(ROOT, 'build', 'kerr_tracer_cuda');
const OUT_DIR      = path.join(ROOT, 'out');
const ASSETS_DIR   = path.join(ROOT, 'assets', 'backgrounds');
const DEFAULT_BACKGROUND = 'sfondo5.jpg';

function firstExisting(paths) {
  for (const p of paths) {
    if (p && fs.existsSync(p)) return p;
  }
  return null;
}

// Prefer dedicated CPU build when available.
const BINARY_CPU = firstExisting([BINARY_CPU_ONLY, BINARY_MAIN]);
// Prefer unified up-to-date build for Metal; legacy target is fallback only.
const BINARY_METAL = firstExisting([BINARY_MAIN, BINARY_METAL_LEGACY]);

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

const DEFAULT_R_OBS = 40;
const DEFAULT_DISK_OUT = 12;

app.use(cors());
app.use(express.json());
app.use('/renders', express.static(OUT_DIR));

// ── Active render state ───────────────────────────────────────
let activeJob = null;   // { proc, clients: Set<WebSocket>, outfile }

function nowSeconds() {
  return Date.now() / 1000;
}

function startProgressHeartbeat(job) {
  if (!job) return;
  job.startedAtSec = nowSeconds();
  job.hasRealProgress = false;
  job.heartbeat = setInterval(() => {
    if (!activeJob || activeJob !== job) return;
    if (job.hasRealProgress) return;
    const elapsed = Math.max(0, nowSeconds() - job.startedAtSec);
    // Heartbeat progress: elapsed-only update (UI can show indeterminate bar).
    broadcast({ type: 'progress', elapsed });
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

// ── WebSocket ─────────────────────────────────────────────────
wss.on('connection', ws => {
  // Send current job status on connect
  if (activeJob) {
    ws.send(JSON.stringify({ type: 'status', running: true }));
  } else {
    ws.send(JSON.stringify({ type: 'status', running: false }));
  }
});

// ── GET /api/info ─────────────────────────────────────────────
app.get('/api/info', (req, res) => {
  const backgrounds = fs.readdirSync(ASSETS_DIR)
    .filter(f => /\.(jpg|jpeg|png)$/i.test(f))
    .map(f => f)
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

// ── GET /api/renders ─────────────────────────────────────────
app.get('/api/renders', (req, res) => {
  const files = fs.readdirSync(OUT_DIR)
    .filter(f => /\.png$/.test(f))
    .map(f => {
      const stat = fs.statSync(path.join(OUT_DIR, f));
      return { name: f, size: stat.size, mtime: stat.mtime };
    })
    .sort((a, b) => b.mtime - a.mtime);
  res.json(files);
});

// ── GET /api/renders/:file ────────────────────────────────────
app.get('/api/renders/:file', (req, res) => {
  const file = path.join(OUT_DIR, req.params.file);
  if (!fs.existsSync(file)) return res.sendStatus(404);
  res.sendFile(file);
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
  if (activeJob) return res.status(409).json({ error: 'Render already running' });

  const { geoFile, exposure, gamma, tempScale } = req.body;
  if (!geoFile) return res.status(400).json({ error: 'geoFile required' });

  const geoPath = path.join(OUT_DIR, geoFile);
  if (!fs.existsSync(geoPath)) return res.status(404).json({ error: 'geo file not found' });

  const args = ['--color-only', geoPath];
  if (exposure  !== undefined) args.push('--exposure',   String(exposure));
  if (gamma     !== undefined) args.push('--gamma',      String(gamma));
  if (tempScale !== undefined) args.push('--temp-scale', String(tempScale));

  broadcast({ type: 'start', args, resolution: 'recolor' });

  const binary = resolveBinary('cpu');
  if (!binary || !fs.existsSync(binary)) {
    return res.status(503).json({ error: 'No renderer binary found for colorize' });
  }
  const proc = spawn(binary, args, { cwd: ROOT });
  const job = { proc, heartbeat: null, startedAtSec: nowSeconds(), hasRealProgress: false };
  activeJob = job;
  startProgressHeartbeat(job);

  proc.stdout.on('data', chunk => {
    broadcast({ type: 'stdout', line: chunk.toString() });
  });

  proc.stderr.on('data', chunk => {
    const raw = chunk.toString();
    const match = raw.match(/\]\s+(\d+)%\s+([\d.]+)s elapsed.*?([\d.]+)s ETA/);
    if (match) {
      job.hasRealProgress = true;
      broadcast({ type: 'progress', pct: parseInt(match[1]), elapsed: parseFloat(match[2]), eta: parseFloat(match[3]) });
    }
  });

  proc.on('close', code => {
    stopProgressHeartbeat(job);
    const outFile = fs.readdirSync(OUT_DIR)
      .filter(f => /\.(png|mp4)$/.test(f))
      .map(f => ({ f, t: fs.statSync(path.join(OUT_DIR, f)).mtime }))
      .sort((a, b) => b.t - a.t)[0]?.f ?? null;

    broadcast({ type: 'done', code, file: outFile });
    activeJob = null;
  });

  res.json({ status: 'started', args });
});

// ── POST /api/render ──────────────────────────────────────────
app.post('/api/render', (req, res) => {
  if (activeJob) return res.status(409).json({ error: 'Render already running' });

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

  const requestedBg = (typeof p.background === 'string' && p.background.trim().length > 0)
    ? p.background.trim()
    : DEFAULT_BACKGROUND;
  if (requestedBg) {
    const bgPath = path.join(ASSETS_DIR, requestedBg);
    if (fs.existsSync(bgPath)) {
      args.push('--bg', bgPath);
    } else {
      console.warn(`[render] background not found: ${requestedBg}`);
    }
  }

  // Disk palette
  if (p.disk_palette === 'interstellar') {
    args.push('--disk-interstellar');
    if (p.disk_rings   !== undefined) args.push('--disk-rings',   String(Math.max(1, Math.floor(Number(p.disk_rings)))));
    if (p.disk_sectors !== undefined) args.push('--disk-sectors', String(Math.max(1, Math.floor(Number(p.disk_sectors)))));
    if (p.disk_sigma   !== undefined) args.push('--disk-sigma',   String(Math.max(0.01, Number(p.disk_sigma))));
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

  broadcast({ type: 'start', args, resolution: res_key });

  const proc = spawn(binary, args, { cwd: ROOT });
  const job = { proc, heartbeat: null, startedAtSec: nowSeconds(), hasRealProgress: false };
  activeJob = job;
  startProgressHeartbeat(job);

  proc.stdout.on('data', chunk => {
    const line = chunk.toString();
    // Parse progress line from stderr (progress bar writes to stderr)
    broadcast({ type: 'stdout', line });
  });

  proc.stderr.on('data', chunk => {
    const raw = chunk.toString();
    // Extract percentage from progress bar
    const match = raw.match(/\]\s+(\d+)%\s+([\d.]+)s elapsed.*?([\d.]+)s ETA/);
    if (match) {
      job.hasRealProgress = true;
      broadcast({
        type: 'progress',
        pct: parseInt(match[1]),
        elapsed: parseFloat(match[2]),
        eta: parseFloat(match[3]),
      });
    }
  });

  proc.on('close', code => {
    stopProgressHeartbeat(job);
    const outFile = fs.readdirSync(OUT_DIR)
      .filter(f => /\.(png|mp4)$/.test(f))
      .map(f => ({ f, t: fs.statSync(path.join(OUT_DIR, f)).mtime }))
      .sort((a, b) => b.t - a.t)[0]?.f ?? null;

    broadcast({ type: 'done', code, file: outFile });
    activeJob = null;
  });

  res.json({ status: 'started', args });
});

// ── POST /api/cancel ─────────────────────────────────────────
app.post('/api/cancel', (req, res) => {
  if (!activeJob) return res.status(404).json({ error: 'No active render' });
  activeJob.proc.kill('SIGTERM');
  res.json({ status: 'cancelled' });
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
