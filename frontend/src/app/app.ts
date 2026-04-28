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
  readonly supersamplingLevels = [1, 2, 3, 4, 6, 8, 12, 16];

  // ── Info from server ──────────────────────────────────────────
  info: ApiInfo | null = null;

  // ── Render parameters ─────────────────────────────────────────
  params: RenderParams = {
    resolution: '720p',
    a: 0.998,
    q: 0.0,
    lambda: 0.0,
    disk_out: 12,
    theta: 80,
    phi: 0,
    r_obs: 40,
    fov: 30,
    backend: 'cpu',
    integration_chart: 'ks',
    solver_mode: 'standard',
    semi_analytic: false,
    bundles: false,
    anti_fireflies: false,
    gpu_fp64: false,
    dopri5: false,
    max_steps: 60000,
    step_init: 1.0,
    integrator_tol: 2e-5,
    camera_spp: 2,
    background: 'sfondo5.jpg',
    disk_palette: 'blackbody',
    disk_rings: 7,
    disk_sectors: 14,
    disk_sigma: 0.5,
    wormhole: false,
    wh_rho: 1.0,
    wh_M_lens: 1.0,
    wh_a_tunnel: 0.01,
    bg_b: '',
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
    anim_r_start: 40,
    anim_r_end: 40,
    anim_disk_start: 12,
    anim_disk_end: 12,
  };

  // ── State signals ─────────────────────────────────────────────
  readonly status   = signal<'idle' | 'running' | 'done' | 'error'>('idle');
  readonly progress = signal(0);
  readonly hasDeterminateProgress = signal(false);
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
      if (info.backgrounds?.length) {
        if (info.backgrounds.includes('sfondo5.jpg')) {
          this.params.background = 'sfondo5.jpg';
        } else if (!info.backgrounds.includes(this.params.background)) {
          this.params.background = info.backgrounds[0];
        }
      }
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
          this.hasDeterminateProgress.set(false);
          this.elapsed.set(0);
          this.eta.set(0);
          this.logLines.set([`Starting: ${msg.args?.join(' ')}`]);
          break;
        case 'progress':
          if (typeof msg.pct === 'number') {
            this.progress.set(msg.pct);
            this.hasDeterminateProgress.set(true);
          }
          if (typeof msg.elapsed === 'number') {
            this.elapsed.set(msg.elapsed);
          }
          if (typeof msg.eta === 'number') {
            this.eta.set(msg.eta);
          }
          break;
        case 'stdout':
          if (msg.line?.trim()) {
            this.logLines.update(l => [...l.slice(-49), msg.line!.trim()]);
          }
          break;
        case 'done':
          this.status.set(msg.code === 0 ? 'done' : 'error');
          this.progress.set(100);
          this.refreshOutputs(msg.file ?? null);
          break;
      }
    });
  }

  ngOnDestroy() {
    this.sub?.unsubscribe();
  }

  toggleBundles() {
    this.params.bundles = !this.params.bundles;
    if (!this.params.bundles) {
      this.params.anti_fireflies = false;
    }
  }

  toggleAntiFireflies() {
    if (!this.params.bundles) {
      this.params.anti_fireflies = false;
      return;
    }
    this.params.anti_fireflies = !this.params.anti_fireflies;
  }

  toggleGpuFp64() {
    if (this.params.backend === 'cpu') {
      this.params.gpu_fp64 = false;
      return;
    }
    this.params.gpu_fp64 = !this.params.gpu_fp64;
  }

  isEllipticClosedAvailable(): boolean {
    const eps = 1e-12;
    return Math.abs(this.params.q) <= eps && Math.abs(this.params.lambda) <= eps;
  }

  setCharge(v: number) {
    this.params.q = v;
    this.enforceSolverConstraints();
  }

  setLambda(v: number) {
    this.params.lambda = v;
    this.enforceSolverConstraints();
  }

  private enforceSolverConstraints() {
    if (!this.isEllipticClosedAvailable() && this.params.solver_mode === 'elliptic_closed') {
      this.params.solver_mode = 'standard';
    }
  }

  startRender() {
    // Keep legacy payload field in sync for backward compatibility.
    this.enforceSolverConstraints();
    this.params.semi_analytic = this.params.solver_mode === 'semi_analytic';
    if (!this.params.bundles) {
      this.params.anti_fireflies = false;
    }
    this.svc.startRender(this.params).subscribe({
      next: () => {
        this.status.set('running');
        this.progress.set(0);
        this.hasDeterminateProgress.set(false);
        this.elapsed.set(0);
        this.eta.set(0);
      },
      error: err => {
        if (err.status === 409) alert('Render already running');
        else console.error(err);
      },
    });
  }

  cancelRender() {
    this.svc.cancelRender().subscribe();
  }

  private refreshOutputs(preferredFile: string | null) {
    const refreshOnce = () => {
      this.loadRenders();
      this.loadGeoFiles();
      if (preferredFile) this.activeRender.set(preferredFile);
    };
    // Retry a few times to avoid races between process exit and fs mtime visibility.
    [0, 250, 800, 1800].forEach(delay => {
      setTimeout(refreshOnce, delay);
    });
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
      next: () => {
        this.status.set('running');
        this.progress.set(0);
        this.hasDeterminateProgress.set(false);
        this.elapsed.set(0);
        this.eta.set(0);
      },
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

  fmtClock(seconds: number): string {
    const s = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) {
      return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    }
    return `${m}:${String(sec).padStart(2, '0')}`;
  }

  fmtSize(bytes: number): string {
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }

  currentResolutionSize(): { w: number; h: number } | null {
    const sizes = this.info?.resolutionSizes;
    if (!sizes) return null;
    return sizes[this.params.resolution] ?? null;
  }

  raysPerFrame(): number | null {
    const sz = this.currentResolutionSize();
    if (!sz) return null;
    return Math.max(1, Math.floor(this.params.camera_spp)) * sz.w * sz.h;
  }

  fmtInt(v: number): string {
    return new Intl.NumberFormat('it-IT').format(Math.floor(v));
  }
}
