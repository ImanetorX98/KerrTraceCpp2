import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Subject } from 'rxjs';

export interface RenderParams {
  resolution: string;
  a: number;
  disk_out: number;
  theta: number;
  r_obs: number;
  bundles: boolean;
  dopri5: boolean;
  background: string;
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

export interface ApiInfo {
  resolutions: string[];
  backgrounds: string[];
  binary: boolean;
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

  renderUrl(filename: string): string {
    // Serve PNG version via backend (auto-converts PPM)
    const png = filename.replace(/\.ppm$/, '.png');
    return `/api/renders/${png}`;
  }
}
