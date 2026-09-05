import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Subject } from 'rxjs';

export interface RenderParams {
  resolution: string;
  // Black hole
  a: number;
  q: number;
  lambda: number;
  disk_out: number;
  // Camera
  theta: number;
  phi: number;
  r_obs: number;
  fov: number;
  // Options
  backend: string;
  integration_chart: 'ks' | 'bl' | 'gks';
  solver_mode: 'standard' | 'semi_analytic' | 'elliptic_closed';
  semi_analytic: boolean;
  bundles: boolean;
  anti_fireflies: boolean;
  gpu_fp64: boolean;
  dopri5: boolean;
  max_steps: number;
  step_init: number;
  integrator_tol: number;
  camera_spp: number;
  background: string;
  // Disk palette
  disk_palette: 'blackbody' | 'stratified' | 'interstellar';
  disk_radial_profile: 'page_thorne' | 'physical_nt';
  disk_brightness: number;
  disk_opacity: number;
  doppler_enabled: boolean;
  disk_inner_emission_floor: number;
  disk_inner_emission_floor_width: number;
  radial_term_zero_torque: boolean;
  radial_term_r3_decay: boolean;
  radial_term_relativistic: boolean;
  radial_term_b_denom: boolean;
  disk_rings: number;
  disk_sectors: number;
  disk_sigma: number;
  disk_hue_offset: number;
  interstellar_omega0: number;
  interstellar_p: number;
  interstellar_physical_profile: boolean;
  interstellar_inner_glow: boolean;
  interstellar_inner_falloff_scale: number;
  interstellar_band_strength: number;
  interstellar_band_frequency: number;
  interstellar_band_warp: number;
  interstellar_turbulence_strength: number;
  interstellar_hdr_intensity: number;
  interstellar_softness_in_scale: number;
  interstellar_softness_out_scale: number;
  interstellar_edge_transparency: number;
  interstellar_time: number;
  interstellar_outer_r: number;
  interstellar_outer_g: number;
  interstellar_outer_b: number;
  // Scene mode
  scene_mode: 'black_hole' | 'wormhole';
  // Wormhole (DNEG metric)
  wormhole: boolean;
  wh_rho: number;
  wh_M_lens: number;
  wh_a_tunnel: number;
  bg_b: string;
  // Animation
  anim: boolean;
  anim_frames: number;
  anim_fps: number;
  anim_crf: number;
  anim_orbits: number;
  anim_ease: boolean;
  anim_theta_start: number;
  anim_theta_end: number;
  anim_phi_start: number;
  anim_phi_end: number;
  anim_r_start: number;
  anim_r_end: number;
  anim_a_start: number;
  anim_a_end: number;
  anim_disk_start: number;
  anim_disk_end: number;
}

export interface QueueJobState {
  id: number;
  kind: 'render' | 'colorize';
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  resolution: string;
  backend: string;
  chart: 'ks' | 'bl' | 'gks' | 'unknown';
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  progressPct?: number;
  elapsedSec?: number;
  etaSec?: number;
  etaSmoothedSec?: number;
  throughputPixPerSec?: number;
  throughputRaysPerSec?: number;
  pixelCount?: number;
  rayCount?: number;
  cameraSpp?: number;
  donePixels?: number;
  doneRays?: number;
  code?: number | null;
  outputFile?: string | null;
  previewFile?: string | null;
  queueIndex?: number | null;
  fallbackUsed?: boolean;
  warnings?: string[];
  logsTail?: string[];
}

export interface QueueStateResponse {
  active: QueueJobState | null;
  queued: QueueJobState[];
  recent: QueueJobState[];
}

export interface WsMessage {
  type: 'status' | 'start' | 'progress' | 'done' | 'stdout' | 'queue_state' | 'job_preview' | 'nav_frame' | 'nav_error'
      | 'nav_bake_progress' | 'nav_bake_done' | 'nav_bake_cancelled' | 'nav_bake_error';
  running?: boolean;
  pct?: number;
  elapsed?: number;
  eta?: number;
  etaSmoothed?: number;
  throughputPixPerSec?: number;
  throughputRaysPerSec?: number;
  pixelCount?: number;
  rayCount?: number;
  donePixels?: number;
  doneRays?: number;
  code?: number;
  file?: string;
  line?: string;
  args?: string[];
  resolution?: string;
  jobId?: number;
  active?: QueueJobState | null;
  queued?: QueueJobState[];
  recent?: QueueJobState[];
  // Navigate
  dataUrl?: string;
  error?: string;
  // Nav bake
  bakeId?: string;
  done?: number;
  total?: number;
}

export interface RenderFile {
  name: string;
  size: number;
  mtime: string;
  meta?: RenderFileMeta;
}

export interface RenderFileMeta {
  resolution: string;
  backend: 'cpu' | 'metal' | 'cuda' | 'unknown';
  chart: 'ks' | 'bl' | 'gks' | 'unknown';
  rayMode?: 'single_ray' | 'ray_bundle';
  solver?: 'standard' | 'semi_analytic' | 'elliptic_closed';
}

export interface RenderMeta extends RenderFile {
  resolution: string;
  backend: 'cpu' | 'metal' | 'cuda' | 'unknown';
  chart: 'ks' | 'bl' | 'gks' | 'unknown';
}

export interface GeoFile {
  name: string;
  size: number;
  mtime: string;
}

export interface ColorizeParams {
  geoFile: string;
  exposure: number;
  gamma: number;
  tempScale: number;
}

export interface NavBakeMeta {
  id: string;
  thetaStep: number;
  phiStep: number;
  total: number;
  createdAt: string;
}

export interface ApiInfo {
  resolutions: string[];
  resolutionSizes?: Record<string, { w: number; h: number }>;
  backgrounds: string[];
  backends: string[];
}

export interface ApiStatus {
  running: boolean;
  startedAtSec?: number | null;
  lastFile?: string | null;
  activeJobId?: number | null;
  queuedCount?: number;
  recentCount?: number;
}

export interface RenderHistoryQuery {
  limit?: number;
  page?: number;
  page_size?: number;
  include_total?: 0 | 1;
  q?: string;
  resolution?: string;
  backend?: string;
  chart?: string;
  type?: string;
  from?: string;
  to?: string;
}

export interface RenderHistoryPage {
  items: RenderFile[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

@Injectable({ providedIn: 'root' })
export class RenderService {
  readonly messages$ = new Subject<WsMessage>();

  private ws: WebSocket | null = null;

  constructor(private http: HttpClient) {
    this.connect();
  }

  private connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    // Use same-origin WS path so it works behind reverse proxies/tunnels
    // (e.g. Cloudflare Tunnel from phone) and in local dev via proxy.
    this.ws = new WebSocket(`${proto}://${location.host}/ws`);

    this.ws.onmessage = ev => {
      try {
        const msg: WsMessage = JSON.parse(ev.data);
        this.messages$.next(msg);
      } catch { /* ignore malformed */ }
    };

    this.ws.onclose = () => {
      setTimeout(() => this.connect(), 2000);
    };
  }

  getInfo() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<ApiInfo>('/api/info', { params });
  }

  getStatus() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<ApiStatus>('/api/status', { params });
  }

  getRenders(query: RenderHistoryQuery = {}) {
    let params = new HttpParams().set('_ts', Date.now().toString());
    Object.entries(query).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      const s = String(v).trim();
      if (!s) return;
      params = params.set(k, s);
    });
    return this.http.get<RenderFile[]>('/api/renders', { params });
  }

  getRendersPage(query: RenderHistoryQuery = {}) {
    let params = new HttpParams().set('_ts', Date.now().toString());
    Object.entries(query).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      const s = String(v).trim();
      if (!s) return;
      params = params.set(k, s);
    });
    return this.http.get<RenderHistoryPage>('/api/renders', { params });
  }

  startRender(params: RenderParams) {
    return this.http.post<{ status: string; args: string[]; jobId?: number; queuePosition?: number }>('/api/render', params);
  }

  cancelRender(jobId?: number) {
    const body = jobId ? { jobId } : {};
    return this.http.post<{ status: string; jobId?: number; active?: boolean }>('/api/cancel', body);
  }

  getQueueState() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<QueueStateResponse>('/api/queue', { params });
  }

  reorderQueue(jobId: number, direction: 'up' | 'down') {
    return this.http.post<{ status: string }>('/api/queue/reorder', { jobId, direction });
  }

  retryRecentJob(jobId: number) {
    return this.http.post<{ status: string; action: 'retry'; jobId?: number; queuePosition?: number }>('/api/queue/retry', { jobId });
  }

  duplicateRecentJob(jobId: number) {
    return this.http.post<{ status: string; action: 'duplicate'; jobId?: number; queuePosition?: number }>('/api/queue/duplicate', { jobId });
  }

  getGeoFiles() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<GeoFile[]>('/api/geo-files', { params });
  }

  colorize(params: ColorizeParams) {
    return this.http.post<{ status: string; args: string[]; jobId?: number; queuePosition?: number }>('/api/colorize', params);
  }

  renderUrl(filename: string): string {
    return `/api/renders/${encodeURIComponent(filename)}`;
  }

  renderThumbUrl(filename: string, width = 360): string {
    const w = Math.max(64, Math.min(1024, Math.floor(width)));
    return `/api/renders-thumb/${encodeURIComponent(filename)}?w=${w}`;
  }

  livePreviewUrl(filename: string): string {
    return `/api/live-preview/${encodeURIComponent(filename)}?_ts=${Date.now()}`;
  }

  getNavBakes() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<{active: any, bakes: NavBakeMeta[]}>('/api/nav-bakes', { params });
  }

  startNavBake(body: Record<string, unknown>) {
    return this.http.post<{ status: string; bakeId?: string; total?: number }>('/api/nav-bake', body);
  }

  cancelNavBake() {
    return this.http.post<{ status: string }>('/api/nav-bake', { cancel: true });
  }

  deleteNavBake(id: string) {
    return this.http.delete<{ status: string }>(`/api/nav-bake/${id}`);
  }

  navBakeFrameUrl(bakeId: string, theta: number, phi: number): string {
    return `/api/nav-bakes/${bakeId}/t${theta}_p${phi}.png`;
  }

  sendWs(data: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}
