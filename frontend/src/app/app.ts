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

interface RenderMetaLite {
  resolution: string;
  backend: 'cpu' | 'metal' | 'cuda' | 'unknown';
  chart: 'ks' | 'bl' | 'unknown';
}

interface SavedPreset {
  name: string;
  createdAt: string;
  params: RenderParams;
}

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
  readonly historyResolutionOptions = ['all', '144p', '256p', '480p', '512p', '720p', '1080p', '2K', '4K', 'custom'];
  readonly historyBackendOptions = ['all', 'cpu', 'metal', 'cuda'];
  readonly historyChartOptions = ['all', 'ks', 'bl'];

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
    background: 'black.png',
    scene_mode: 'black_hole',
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
  readonly compareRender = signal<string | null>(null);
  compareMode = false;
  compareLayout: 'split' | 'side_by_side' = 'split';
  compareSplitPercent = 50;
  historyQuery = '';
  historyResolutionFilter = 'all';
  historyBackendFilter = 'all';
  historyChartFilter = 'all';
  private readonly presetStorageKey = 'knds_presets_v1';
  presets: SavedPreset[] = [];
  presetDraftName = '';
  selectedPresetName = '';

  // ── Post-process panel ────────────────────────────────────────
  geoFiles: GeoFile[] = [];
  colorParams: ColorizeParams = { geoFile: '', exposure: 1.0, gamma: 2.2, tempScale: 1.0 };
  postProcessTab: 'render' | 'recolor' = 'render';

  readonly previewUrl = computed(() => {
    const r = this.activeRender();
    return r ? this.svc.renderUrl(r) : null;
  });

  readonly comparePreviewUrl = computed(() => {
    const r = this.compareRender();
    return r ? this.svc.renderUrl(r) : null;
  });

  private sub: Subscription | null = null;
  private statusPollTimer: ReturnType<typeof setInterval> | null = null;

  constructor(readonly svc: RenderService) {}

  ngOnInit() {
    this.loadPresets();
    this.svc.getInfo().subscribe(info => {
      this.info = info;
      if (info.backgrounds?.length && !info.backgrounds.includes(this.params.background)) {
        this.params.background = info.backgrounds[0];
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
          this.startStatusPolling();
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
          this.stopStatusPolling();
          this.refreshOutputs(msg.file ?? null);
          break;
      }
    });

    // Fallback for remote/tunnel sessions: keep UI in sync even if WS drops.
    this.svc.getStatus().subscribe(s => {
      if (s.running) {
        this.status.set('running');
        this.startStatusPolling();
      }
    });
  }

  ngOnDestroy() {
    this.sub?.unsubscribe();
    this.stopStatusPolling();
  }

  setSceneMode(mode: 'black_hole' | 'wormhole') {
    this.params.scene_mode = mode;
    this.params.wormhole = (mode === 'wormhole');
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
        this.startStatusPolling();
      },
      error: err => {
        if (err.status === 409) alert('Render already running');
        else console.error(err);
      },
    });
  }

  cancelRender() {
    this.svc.cancelRender().subscribe();
    this.stopStatusPolling();
  }

  private refreshOutputs(preferredFile: string | null) {
    const refreshOnce = () => {
      this.svc.getRenders().subscribe(files => {
        this.renders = files;
        const select = preferredFile ?? (files.length > 0 ? files[0].name : null);
        if (select) this.activeRender.set(select);
        this.syncCompareSelection();
      });
      this.loadGeoFiles();
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
      this.syncCompareSelection();
    });
  }

  private startStatusPolling() {
    if (this.statusPollTimer !== null) return;
    this.statusPollTimer = setInterval(() => {
      if (this.status() !== 'running') return;
      this.svc.getStatus().subscribe({
        next: s => {
          if (!s.running) {
            this.status.set('done');
            this.progress.set(100);
            this.stopStatusPolling();
            this.refreshOutputs(s.lastFile ?? null);
          }
        },
        error: () => {
          // Keep running; WS may still deliver completion.
        },
      });
    }, 2000);
  }

  private stopStatusPolling() {
    if (this.statusPollTimer === null) return;
    clearInterval(this.statusPollTimer);
    this.statusPollTimer = null;
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
    if (this.compareMode && !this.compareRender()) {
      const alt = this.renders.find(r => r.name !== name);
      this.compareRender.set(alt?.name ?? null);
    }
  }

  toggleCompareMode() {
    this.compareMode = !this.compareMode;
    if (this.compareMode && !this.compareRender()) {
      const active = this.activeRender();
      const alt = this.renders.find(r => r.name !== active);
      this.compareRender.set(alt?.name ?? null);
    }
  }

  setCompareLayout(layout: 'split' | 'side_by_side') {
    this.compareLayout = layout;
  }

  clampCompareSplit() {
    const v = Number(this.compareSplitPercent);
    if (!Number.isFinite(v)) {
      this.compareSplitPercent = 50;
      return;
    }
    this.compareSplitPercent = Math.max(0, Math.min(100, Math.round(v)));
  }

  selectCompareRender(name: string) {
    this.compareRender.set(name);
  }

  swapCompare() {
    const a = this.activeRender();
    const b = this.compareRender();
    if (!a || !b) return;
    this.activeRender.set(b);
    this.compareRender.set(a);
  }

  filteredRenders(): RenderFile[] {
    const q = this.historyQuery.trim().toLowerCase();
    return this.renders.filter(r => {
      const meta = this.getRenderMeta(r.name);
      if (q && !r.name.toLowerCase().includes(q)) return false;
      if (this.historyResolutionFilter !== 'all') {
        if (this.historyResolutionFilter === 'custom') {
          if (meta.resolution !== 'custom') return false;
        } else if (meta.resolution !== this.historyResolutionFilter) {
          return false;
        }
      }
      if (this.historyBackendFilter !== 'all' && meta.backend !== this.historyBackendFilter) return false;
      if (this.historyChartFilter !== 'all' && meta.chart !== this.historyChartFilter) return false;
      return true;
    });
  }

  getRenderMeta(name: string): RenderMetaLite {
    const lower = name.toLowerCase();
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

    let backend: RenderMetaLite['backend'] = 'unknown';
    if (lower.includes('gpu-metal')) backend = 'metal';
    else if (lower.includes('gpu-cuda')) backend = 'cuda';
    else if (lower.includes('_cpu_') || lower.endsWith('_cpu.png') || lower.includes('_cpu-')) backend = 'cpu';

    let chart: RenderMetaLite['chart'] = 'unknown';
    if (lower.includes('_ks-') || lower.includes('_ks_')) chart = 'ks';
    else if (lower.includes('_bl-') || lower.includes('_bl_')) chart = 'bl';

    return { resolution, backend, chart };
  }

  humanBackend(meta: RenderMetaLite): string {
    if (meta.backend === 'metal') return 'GPU Metal';
    if (meta.backend === 'cuda') return 'GPU CUDA';
    if (meta.backend === 'cpu') return 'CPU';
    return 'N/A';
  }

  saveCurrentPreset() {
    const name = this.presetDraftName.trim();
    if (!name) return;
    const snapshot = this.cloneParams(this.params);
    const existingIdx = this.presets.findIndex(p => p.name.toLowerCase() === name.toLowerCase());
    const next: SavedPreset = {
      name,
      createdAt: new Date().toISOString(),
      params: snapshot,
    };
    if (existingIdx >= 0) this.presets.splice(existingIdx, 1);
    this.presets.unshift(next);
    this.presets = this.presets.slice(0, 40);
    this.selectedPresetName = name;
    this.persistPresets();
  }

  applySelectedPreset() {
    if (!this.selectedPresetName) return;
    const preset = this.presets.find(p => p.name === this.selectedPresetName);
    if (!preset) return;
    this.params = this.cloneParams(preset.params);
    this.enforceSolverConstraints();
    if (!this.params.bundles) this.params.anti_fireflies = false;
  }

  deleteSelectedPreset() {
    if (!this.selectedPresetName) return;
    this.presets = this.presets.filter(p => p.name !== this.selectedPresetName);
    this.persistPresets();
    this.selectedPresetName = '';
  }

  private cloneParams(src: RenderParams): RenderParams {
    return JSON.parse(JSON.stringify(src)) as RenderParams;
  }

  private loadPresets() {
    try {
      const raw = localStorage.getItem(this.presetStorageKey);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (!Array.isArray(data)) return;
      const presets = data
        .filter((p: any) => p && typeof p.name === 'string' && p.params)
        .map((p: any) => ({
          name: p.name,
          createdAt: typeof p.createdAt === 'string' ? p.createdAt : new Date().toISOString(),
          params: this.cloneParams(p.params as RenderParams),
        })) as SavedPreset[];
      this.presets = presets.slice(0, 40);
      if (this.presets.length > 0) this.selectedPresetName = this.presets[0].name;
    } catch {
      this.presets = [];
    }
  }

  private persistPresets() {
    localStorage.setItem(this.presetStorageKey, JSON.stringify(this.presets));
  }

  private syncCompareSelection() {
    const names = new Set(this.renders.map(r => r.name));
    const active = this.activeRender();
    const compare = this.compareRender();
    if (active && !names.has(active)) this.activeRender.set(this.renders[0]?.name ?? null);
    if (compare && !names.has(compare)) this.compareRender.set(null);
    if (this.compareMode && !this.compareRender()) {
      const a = this.activeRender();
      const alt = this.renders.find(r => r.name !== a);
      this.compareRender.set(alt?.name ?? null);
    }
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
