import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
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
  bundles: boolean;
  dopri5: boolean;
  background: string;
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
  backgrounds: string[];
  backends: string[];
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
    // In dev, proxy doesn't handle WS — connect directly to backend port
    const host = location.hostname;
    this.ws = new WebSocket(`${proto}://${host}:3001/ws`);

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
    return this.http.get<ApiInfo>('/api/info');
  }

  getRenders() {
    return this.http.get<RenderFile[]>('/api/renders');
  }

  startRender(params: RenderParams) {
    return this.http.post<{ status: string; args: string[] }>('/api/render', params);
  }

  cancelRender() {
    return this.http.post<{ status: string }>('/api/cancel', {});
  }

  getGeoFiles() {
    return this.http.get<GeoFile[]>('/api/geo-files');
  }

  colorize(params: ColorizeParams) {
    return this.http.post<{ status: string; args: string[] }>('/api/colorize', params);
  }

  renderUrl(filename: string): string {
    return `/api/renders/${filename}`;
  }
}
