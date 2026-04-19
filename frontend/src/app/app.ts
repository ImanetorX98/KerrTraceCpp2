import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { Subscription } from 'rxjs';

import { MatProgressBarModule } from '@angular/material/progress-bar';
import { NumericInputComponent } from './numeric-input.component';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';

import { RenderService, RenderParams, RenderFile, GeoFile, ColorizeParams, ApiInfo } from './render.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, FormsModule, HttpClientModule,
    MatProgressBarModule, MatSelectModule,
    MatIconModule, MatTooltipModule,
    NumericInputComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App implements OnInit, OnDestroy {
  // ── Info from server ──────────────────────────────────────────
  info: ApiInfo | null = null;

  // ── Render parameters ─────────────────────────────────────────
  params: RenderParams = {
    resolution: '720p',
    a: 0.998,
    q: 0.0,
    lambda: 0.0,
    disk_out: 25,
    theta: 80,
    phi: 0,
    r_obs: 30,
    fov: 30,
    backend: 'cpu',
    bundles: false,
    dopri5: false,
    background: '',
    anim: false,
    anim_frames: 60,
    anim_fps: 30,
    anim_crf: 18,
    anim_orbits: 1,
    anim_ease: true,
    anim_a_start: 0.998,
    anim_a_end: 0.998,
    anim_theta_start: 80,
    anim_theta_end: 80,
    anim_phi_start: 0,
    anim_phi_end: 360,
    anim_r_start: 30,
    anim_r_end: 30,
    anim_disk_start: 25,
    anim_disk_end: 25,
  };

  // ── State signals ─────────────────────────────────────────────
  readonly status   = signal<'idle' | 'running' | 'done' | 'error'>('idle');
  readonly progress = signal(0);
  readonly elapsed  = signal(0);
  readonly eta      = signal(0);
  readonly logLines = signal<string[]>([]);

  // ── Gallery ───────────────────────────────────────────────────
  renders: RenderFile[] = [];
  readonly activeRender = signal<string | null>(null);

  // ── Post-process panel ────────────────────────────────────────
  geoFiles: GeoFile[] = [];
  colorParams: ColorizeParams = { geoFile: '', exposure: 1.0, gamma: 2.2, tempScale: 1.0 };
  postProcessTab: 'render' | 'recolor' = 'render';

  readonly previewUrl = computed(() => {
    const r = this.activeRender();
    return r ? this.svc.renderUrl(r) : null;
  });

  private sub: Subscription | null = null;

  constructor(readonly svc: RenderService) {}

  ngOnInit() {
    this.svc.getInfo().subscribe(info => {
      this.info = info;
    });
    this.loadRenders();
    this.loadGeoFiles();

    this.sub = this.svc.messages$.subscribe(msg => {
      switch (msg.type) {
        case 'status':
          if (msg.running) this.status.set('running');
          break;
        case 'start':
          this.status.set('running');
          this.progress.set(0);
          this.logLines.set([`Starting: ${msg.args?.join(' ')}`]);
          break;
        case 'progress':
          this.progress.set(msg.pct ?? 0);
          this.elapsed.set(msg.elapsed ?? 0);
          this.eta.set(msg.eta ?? 0);
          break;
        case 'stdout':
          if (msg.line?.trim()) {
            this.logLines.update(l => [...l.slice(-49), msg.line!.trim()]);
          }
          break;
        case 'done':
          this.status.set(msg.code === 0 ? 'done' : 'error');
          this.progress.set(100);
          if (msg.file) {
            this.activeRender.set(msg.file);
            setTimeout(() => { this.loadRenders(); this.loadGeoFiles(); }, 500);
          }
          break;
      }
    });
  }

  ngOnDestroy() {
    this.sub?.unsubscribe();
  }

  startRender() {
    this.svc.startRender(this.params).subscribe({
      next: () => { this.status.set('running'); this.progress.set(0); },
      error: err => {
        if (err.status === 409) alert('Render already running');
        else console.error(err);
      },
    });
  }

  cancelRender() {
    this.svc.cancelRender().subscribe();
  }

  loadRenders() {
    this.svc.getRenders().subscribe(files => {
      this.renders = files;
      if (!this.activeRender() && files.length > 0) {
        this.activeRender.set(files[0].name);
      }
    });
  }

  loadGeoFiles() {
    this.svc.getGeoFiles().subscribe(files => {
      this.geoFiles = files;
      if (!this.colorParams.geoFile && files.length > 0) {
        this.colorParams.geoFile = files[0].name;
      }
    });
  }

  startColorize() {
    if (!this.colorParams.geoFile) return;
    this.svc.colorize(this.colorParams).subscribe({
      next: () => { this.status.set('running'); this.progress.set(0); },
      error: err => {
        if (err.status === 409) alert('Render already running');
        else console.error(err);
      },
    });
  }

  selectRender(name: string) {
    this.activeRender.set(name);
  }

  fmt(v: number, d = 2): string {
    return v.toFixed(d);
  }

  fmtSize(bytes: number): string {
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }
}
