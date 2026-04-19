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
const BINARY_CPU   = path.join(ROOT, 'build', 'kerr_tracer');
const BINARY_METAL = path.join(ROOT, 'build', 'kerr_tracer_metal');
const BINARY_CUDA  = path.join(ROOT, 'build', 'kerr_tracer_cuda');
const OUT_DIR      = path.join(ROOT, 'out');
const ASSETS_DIR   = path.join(ROOT, 'assets', 'backgrounds');

function resolveBinary(backend) {
  if (backend === 'metal' && fs.existsSync(BINARY_METAL)) return BINARY_METAL;
  if (backend === 'cuda'  && fs.existsSync(BINARY_CUDA))  return BINARY_CUDA;
  return BINARY_CPU;
}

function availableBackends() {
  const b = [];
  if (fs.existsSync(BINARY_CPU))   b.push('cpu');
  if (fs.existsSync(BINARY_METAL)) b.push('metal');
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

app.use(cors());
app.use(express.json());
app.use('/renders', express.static(OUT_DIR));

// ── Active render state ───────────────────────────────────────
let activeJob = null;   // { proc, clients: Set<WebSocket>, outfile }

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
    .map(f => f);

  res.json({
    resolutions: Object.keys(RESOLUTIONS),
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

  const proc = spawn(BINARY, args, { cwd: ROOT });
  activeJob = { proc };

  proc.stdout.on('data', chunk => {
    broadcast({ type: 'stdout', line: chunk.toString() });
  });

  proc.stderr.on('data', chunk => {
    const raw = chunk.toString();
    const match = raw.match(/\]\s+(\d+)%\s+([\d.]+)s elapsed.*?([\d.]+)s ETA/);
    if (match) {
      broadcast({ type: 'progress', pct: parseInt(match[1]), elapsed: parseFloat(match[2]), eta: parseFloat(match[3]) });
    }
  });

  proc.on('close', code => {
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
  if (p.dopri5)   args.push('--dopri5');
  if (p.semi_analytic) args.push('--semi-analytic');
  if (p.integration_chart === 'bl') args.push('--bl');
  else args.push('--ks');

  if (p.a      !== undefined) args.push('--a',       String(p.a));
  if (p.disk_out !== undefined) args.push('--disk-out', String(p.disk_out));
  if (p.theta  !== undefined) args.push('--theta',   String(p.theta));
  if (p.r_obs  !== undefined) args.push('--r-obs',   String(p.r_obs));
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

  if (p.background) {
    const bgPath = path.join(ASSETS_DIR, p.background);
    if (fs.existsSync(bgPath)) args.push('--bg', bgPath);
  }

  // Always save geodesic cache alongside every render
  if (!p.anim) {
    const ts = new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 15);
    const geoPath = path.join(OUT_DIR, `geo_${ts}.kgeo`);
    args.push('--geo-file', geoPath);
  }

  broadcast({ type: 'start', args, resolution: res_key });

  const proc = spawn(binary, args, { cwd: ROOT });
  activeJob = { proc };

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
      broadcast({
        type: 'progress',
        pct: parseInt(match[1]),
        elapsed: parseFloat(match[2]),
        eta: parseFloat(match[3]),
      });
    }
  });

  proc.on('close', code => {
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
  console.log(`CPU:   ${BINARY_CPU}`);
  console.log(`Metal: ${BINARY_METAL}`);
  console.log(`Output: ${OUT_DIR}`);
});
