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
  integration_chart: 'ks' | 'bl';
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
  disk_palette: 'blackbody' | 'interstellar';
  disk_rings: number;
  disk_sectors: number;
  disk_sigma: number;
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

export interface WsMessage {
  type: 'status' | 'start' | 'progress' | 'done' | 'stdout';
  running?: boolean;
  pct?: number;
  elapsed?: number;
  eta?: number;
  code?: number;
  file?: string;
  line?: string;
  args?: string[];
  resolution?: string;
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
  chart: 'ks' | 'bl' | 'unknown';
  rayMode?: 'single_ray' | 'ray_bundle';
  solver?: 'standard' | 'semi_analytic' | 'elliptic_closed';
}

export interface RenderMeta extends RenderFile {
  resolution: string;
  backend: 'cpu' | 'metal' | 'cuda' | 'unknown';
  chart: 'ks' | 'bl' | 'unknown';
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
}

export interface RenderHistoryQuery {
  limit?: number;
  q?: string;
  resolution?: string;
  backend?: string;
  chart?: string;
  type?: string;
  from?: string;
  to?: string;
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

  startRender(params: RenderParams) {
    return this.http.post<{ status: string; args: string[] }>('/api/render', params);
  }

  cancelRender() {
    return this.http.post<{ status: string }>('/api/cancel', {});
  }

  getGeoFiles() {
    const params = new HttpParams().set('_ts', Date.now().toString());
    return this.http.get<GeoFile[]>('/api/geo-files', { params });
  }

  colorize(params: ColorizeParams) {
    return this.http.post<{ status: string; args: string[] }>('/api/colorize', params);
  }

  renderUrl(filename: string): string {
    return `/api/renders/${encodeURIComponent(filename)}`;
  }

  renderThumbUrl(filename: string, width = 360): string {
    const w = Math.max(64, Math.min(1024, Math.floor(width)));
    return `/api/renders-thumb/${encodeURIComponent(filename)}?w=${w}`;
  }
}
