import { Component, OnInit, OnDestroy, HostListener, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { Subscription } from 'rxjs';

import { MatProgressBarModule } from '@angular/material/progress-bar';
import { NumericInputComponent } from './numeric-input.component';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';

import {
  RenderService,
  RenderParams,
  RenderFile,
  GeoFile,
  ColorizeParams,
  ApiInfo,
  RenderHistoryQuery,
  QueueJobState,
  QueueStateResponse,
  NavBakeMeta,
} from './render.service';

interface RenderMetaLite {
  resolution: string;
  backend: 'cpu' | 'metal' | 'cuda' | 'unknown';
  chart: 'ks' | 'bl' | 'gks' | 'unknown';
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
  readonly historyChartOptions = ['all', 'ks', 'gks', 'bl'];
  readonly historyTypeOptions = [
    'all',
    'single_ray',
    'ray_bundle',
    'standard_rk4',
    'semi_analytic',
    'elliptic_closed',
  ];

  // ── Info from server ──────────────────────────────────────────
  info: ApiInfo | null = null;

  // ── Render parameters ─────────────────────────────────────────
  params: RenderParams = {
    resolution: '720p',
    a: 0.5,
    q: 0.0,
    lambda: 0.0,
    disk_out: 12,
    theta: 80,
    phi: 0,
    r_obs: 60,
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
    camera_spp: 1,
    background: 'black.png',
    scene_mode: 'black_hole',
    disk_palette: 'blackbody',
    disk_radial_profile: 'page_thorne',
    disk_brightness: 1.0,
    disk_opacity: 1.0,
    doppler_enabled: true,
    disk_inner_emission_floor: 0.0,
    disk_inner_emission_floor_width: 0.25,
    radial_term_zero_torque: true,
    radial_term_r3_decay: true,
    radial_term_relativistic: true,
    radial_term_b_denom: true,
    disk_rings: 7,
    disk_sectors: 14,
    disk_sigma: 0.5,
    disk_hue_offset: 0.0,
    interstellar_omega0: 1.0,
    interstellar_p: 2.2,
    interstellar_inner_glow: false,
    interstellar_inner_falloff_scale: 0.7,
    interstellar_band_strength: 0.18,
    interstellar_band_frequency: 20.0,
    interstellar_band_warp: 2.0,
    interstellar_turbulence_strength: 1.0,
    interstellar_hdr_intensity: 4.0,
    interstellar_softness_in_scale: 0.08,
    interstellar_softness_out_scale: 0.15,
    interstellar_edge_transparency: 0.02,
    interstellar_time: 0.0,
    interstellar_outer_r: 0.16,
    interstellar_outer_g: 0.045,
    interstellar_outer_b: 0.018,
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
    anim_a_start: 0.5,
    anim_a_end: 0.5,
    anim_theta_start: 80,
    anim_theta_end: 80,
    anim_phi_start: 0,
    anim_phi_end: 360,
    anim_r_start: 60,
    anim_r_end: 60,
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
  readonly queueActive = signal<QueueJobState | null>(null);
  readonly queuePending = signal<QueueJobState[]>([]);
  readonly queueRecent = signal<QueueJobState[]>([]);
  readonly livePreviewFile = signal<string | null>(null);

  // ── Gallery ───────────────────────────────────────────────────
  renders: RenderFile[] = [];
  readonly activeRender = signal<string | null>(null);
  readonly compareRender = signal<string | null>(null);
  compareMode = false;
  compareLayout: 'split' | 'side_by_side' = 'split';
  compareSplitPercent = 50;
  historyQuery = '';
  historyDateFrom = '';
  historyDateTo = '';
  historyTypeFilter = 'all';
  historyResolutionFilter = 'all';
  historyBackendFilter = 'all';
  historyChartFilter = 'all';
  historyPage = 1;
  historyPageSize = 24;
  historyTotal = 0;
  historyTotalPages = 1;
  historyHasNext = false;
  historyHasPrev = false;
  historyMode: 'latest' | 'search' = 'latest';
  readonly isMobileView = signal(false);
  private readonly presetStorageKey = 'knds_presets_v1';
  presets: SavedPreset[] = [];
  presetDraftName = '';
  selectedPresetName = '';

  // ── Post-process panel ────────────────────────────────────────
  geoFiles: GeoFile[] = [];
  colorParams: ColorizeParams = { geoFile: '', exposure: 1.0, gamma: 2.2, tempScale: 1.0 };
  postProcessTab: 'render' | 'recolor' | 'navigate' = 'render';

  // ── Navigate ──────────────────────────────────────────────────
  navTheta        = 80;
  navPhi          = 0;
  navSensitivity  = 0.35;
  navBackground: string | null = null;  // null = use params.background
  navPalette: 'blackbody' | 'stratified' | 'interstellar' | null = null;  // null = use params.disk_palette

  // Nav bake
  navBakeStepTheta   = signal(10); // degrees for θ
  navBakeStepPhi     = signal(10); // degrees for φ
  navBakeId          = signal<string | null>(null);   // active bake being used
  navBakeList        = signal<NavBakeMeta[]>([]);
  navBaking          = signal(false);
  navBakeDone        = signal(0);
  navBakeTotal       = signal(0);
  readonly navBakeEta = computed(() => {
    const done = this.navBakeDone(), total = this.navBakeTotal();
    if (!this.navBaking() || done === 0 || total === 0) return null;
    const avg = this.navAvgMs() > 0 ? this.navAvgMs() : 1000;
    return Math.round((total - done) * avg / 1000);
  });
  readonly navBakePct = computed(() => {
    const total = this.navBakeTotal();
    return total > 0 ? Math.round(this.navBakeDone() / total * 100) : 0;
  });
  readonly navBakeEstFrames = computed(() => {
    const tCount = Math.floor(179 / this.navBakeStepTheta());
    const pCount = Math.ceil(360 / this.navBakeStepPhi());
    return tCount * pCount;
  });

  readonly navFrameUrl    = signal<string | null>(null);
  readonly navLoading     = signal(false);
  readonly navElapsedMs   = signal(0);
  readonly navAvgMs       = signal(0);
  readonly navProgressPct = computed(() => {
    const avg = this.navAvgMs();
    if (avg <= 0) return -1;  // -1 = indeterminate
    return Math.min(99, Math.round(this.navElapsedMs() / avg * 100));
  });
  private navStartMs      = 0;
  private navAvgHistory: number[] = [];
  private navElapsedTimer: ReturnType<typeof setInterval> | null = null;
  private navDragging  = false;
  private navDragX0    = 0;
  private navDragY0    = 0;
  private navTheta0    = 80;
  private navPhi0      = 0;
  private navThrottle: ReturnType<typeof setTimeout> | null = null;

  readonly previewUrl = computed(() => {
    if (this.status() === 'running' && this.livePreviewFile()) {
      return this.svc.livePreviewUrl(this.livePreviewFile()!);
    }
    const r = this.activeRender();
    return r ? this.svc.renderUrl(r) : null;
  });

  readonly comparePreviewUrl = computed(() => {
    const r = this.compareRender();
    return r ? this.svc.renderUrl(r) : null;
  });

  private sub: Subscription | null = null;
  private statusPollTimer: ReturnType<typeof setInterval> | null = null;
  private readonly chartEpsilon = 1e-12;

  constructor(readonly svc: RenderService) {}

  @HostListener('window:resize')
  onWindowResize() {
    this.updateViewportFlag();
  }

  private updateViewportFlag() {
    this.isMobileView.set(window.innerWidth <= 860);
  }

  ngOnInit() {
    this.syncIntegrationChartForMetric(true);
    this.updateViewportFlag();
    this.loadPresets();
    this.svc.getInfo().subscribe(info => {
      this.info = info;
      if (info.backgrounds?.length && !info.backgrounds.includes(this.params.background)) {
        this.params.background = info.backgrounds[0];
      }
    });
    this.loadLatestRenders();
    this.loadGeoFiles();
    this.loadQueueState();
    this.loadNavBakes();

    this.sub = this.svc.messages$.subscribe(msg => {
      switch (msg.type) {
        case 'status':
          if (msg.running) this.status.set('running');
          break;
        case 'queue_state':
          this.applyQueueState({
            active: msg.active ?? null,
            queued: msg.queued ?? [],
            recent: msg.recent ?? [],
          });
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
        case 'job_preview':
          if (msg.file) {
            this.livePreviewFile.set(msg.file);
          }
          break;
        case 'progress':
          if (typeof msg.pct === 'number') {
            this.progress.set(msg.pct);
            this.hasDeterminateProgress.set(true);
          }
          if (typeof msg.elapsed === 'number') {
            this.elapsed.set(msg.elapsed);
          }
          if (typeof msg.etaSmoothed === 'number') {
            this.eta.set(msg.etaSmoothed);
          } else if (typeof msg.eta === 'number') {
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
          this.livePreviewFile.set(null);
          this.stopStatusPolling();
          this.refreshOutputs(msg.file ?? null);
          break;
        case 'nav_frame':
          if (msg.dataUrl) {
            this.navStopTimer();
            this.navFrameUrl.set(msg.dataUrl);
            this.navLoading.set(false);
          }
          break;
        case 'nav_error':
          this.navStopTimer();
          this.navLoading.set(false);
          break;
        case 'nav_bake_progress':
          this.navBakeDone.set(msg.done ?? 0);
          this.navBakeTotal.set(msg.total ?? 0);
          break;
        case 'nav_bake_done':
          this.navBaking.set(false);
          this.navBakeDone.set(msg.total ?? 0);
          this.loadNavBakes();
          break;
        case 'nav_bake_cancelled':
          this.navBaking.set(false);
          break;
        case 'nav_bake_error':
          this.navBaking.set(false);
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

  private applyQueueState(state: QueueStateResponse) {
    this.queueActive.set(state.active ?? null);
    this.queuePending.set(Array.isArray(state.queued) ? state.queued : []);
    this.queueRecent.set(Array.isArray(state.recent) ? state.recent.slice(0, 24) : []);
    if (state.active?.previewFile) {
      this.livePreviewFile.set(state.active.previewFile);
    }
    if (state.active?.status === 'running') {
      this.status.set('running');
    } else if (!state.active && this.status() === 'running') {
      this.status.set('done');
      this.livePreviewFile.set(null);
    }
  }

  loadQueueState() {
    this.svc.getQueueState().subscribe({
      next: s => this.applyQueueState(s),
      error: () => {
        // noop
      },
    });
  }

  setSceneMode(mode: 'black_hole' | 'wormhole') {
    this.params.scene_mode = mode;
    this.params.wormhole = (mode === 'wormhole');
    if (mode === 'black_hole') {
      this.syncIntegrationChartForMetric(true);
    }
  }

  applyAnimationPreset(mode: 'black_hole' | 'wormhole') {
    this.params.anim = true;
    this.setSceneMode(mode);

    if (mode === 'black_hole') {
      this.params.anim_frames = 180;
      this.params.anim_fps = 30;
      this.params.anim_crf = 18;
      this.params.anim_orbits = 1;
      this.params.anim_ease = true;
      this.params.anim_a_start = 0.7;
      this.params.anim_a_end = 0.7;
      this.params.anim_theta_start = 80;
      this.params.anim_theta_end = 80;
      this.params.anim_phi_start = 0;
      this.params.anim_phi_end = 360;
      this.params.anim_r_start = 60;
      this.params.anim_r_end = 60;
      this.params.anim_disk_start = 12;
      this.params.anim_disk_end = 12;
      return;
    }

    // Wormhole preset: no spin sweep, camera-forward pass with moderate pan.
    this.params.wh_rho = 1.0;
    this.params.wh_M_lens = 1.0;
    this.params.wh_a_tunnel = 0.01;
    this.params.anim_frames = 220;
    this.params.anim_fps = 30;
    this.params.anim_crf = 18;
    this.params.anim_orbits = 0.5;
    this.params.anim_ease = true;
    this.params.anim_a_start = this.params.a;
    this.params.anim_a_end = this.params.a;
    this.params.anim_theta_start = 78;
    this.params.anim_theta_end = 78;
    this.params.anim_phi_start = -40;
    this.params.anim_phi_end = 40;
    this.params.anim_r_start = 60;
    this.params.anim_r_end = 20;
    this.params.anim_disk_start = this.params.disk_out;
    this.params.anim_disk_end = this.params.disk_out;
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
    return Math.abs(this.params.q) <= this.chartEpsilon && Math.abs(this.params.lambda) <= this.chartEpsilon;
  }

  setCharge(v: number) {
    this.params.q = v;
    this.enforceSolverConstraints();
  }

  setLambda(v: number) {
    this.params.lambda = v;
    this.enforceSolverConstraints();
  }

  selectIntegrationChart(chart: 'ks' | 'gks' | 'bl') {
    this.params.integration_chart = chart;
    this.enforceSolverConstraints();
  }

  private enforceSolverConstraints() {
    this.syncIntegrationChartForMetric(true);
    if (!this.isEllipticClosedAvailable() && this.params.solver_mode === 'elliptic_closed') {
      this.params.solver_mode = 'standard';
    }
  }

  private expectedIntegrationChartForMetric(): 'ks' | 'gks' {
    return this.isEllipticClosedAvailable() ? 'ks' : 'gks';
  }

  private syncIntegrationChartForMetric(force = false) {
    if (this.params.scene_mode !== 'black_hole') return;
    const expected = this.expectedIntegrationChartForMetric();
    if (force || this.params.integration_chart === 'ks' || this.params.integration_chart === 'gks') {
      this.params.integration_chart = expected;
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
      next: rsp => {
        if (rsp.status === 'started') {
          this.status.set('running');
          this.progress.set(0);
          this.hasDeterminateProgress.set(false);
          this.elapsed.set(0);
          this.eta.set(0);
          this.startStatusPolling();
        } else {
          this.logLines.update(l => [...l.slice(-49), `Queued render job #${rsp.jobId ?? '?'} (pos ${rsp.queuePosition ?? '?'})`]);
        }
        this.loadQueueState();
      },
      error: err => {
        console.error(err);
      },
    });
  }

  cancelRender(jobId?: number) {
    this.svc.cancelRender(jobId).subscribe({
      next: () => this.loadQueueState(),
      error: err => console.error(err),
    });
    this.stopStatusPolling();
  }

  private refreshOutputs(preferredFile: string | null) {
    const refreshOnce = () => {
      this.loadRenders(preferredFile);
      this.loadQueueState();
      this.loadGeoFiles();
    };
    // Retry a few times to avoid races between process exit and fs mtime visibility.
    [0, 250, 800, 1800].forEach(delay => {
      setTimeout(refreshOnce, delay);
    });
  }

  loadRenders(preferredFile: string | null = null) {
    const query = this.buildHistoryQuery();
    this.svc.getRendersPage(query).subscribe(page => {
      this.renders = page.items;
      this.historyTotal = page.total;
      this.historyTotalPages = page.totalPages;
      this.historyHasNext = page.hasNext;
      this.historyHasPrev = page.hasPrev;
      const select = preferredFile
        ?? this.activeRender()
        ?? (page.items.length > 0 ? page.items[0].name : null);
      if (select) this.activeRender.set(select);
      this.syncCompareSelection();
    });
  }

  loadLatestRenders() {
    this.historyMode = 'latest';
    this.historyPage = 1;
    this.historyPageSize = 10;
    this.loadRenders();
  }

  runHistorySearch() {
    this.historyMode = 'search';
    this.historyPage = 1;
    this.historyPageSize = 24;
    this.loadRenders();
  }

  historyNextPage() {
    if (this.historyMode !== 'search') return;
    if (!this.historyHasNext) return;
    this.historyPage += 1;
    this.loadRenders();
  }

  historyPrevPage() {
    if (this.historyMode !== 'search') return;
    if (!this.historyHasPrev) return;
    this.historyPage = Math.max(1, this.historyPage - 1);
    this.loadRenders();
  }

  resetHistoryToLatest() {
    this.historyQuery = '';
    this.historyDateFrom = '';
    this.historyDateTo = '';
    this.historyTypeFilter = 'all';
    this.historyResolutionFilter = 'all';
    this.historyBackendFilter = 'all';
    this.historyChartFilter = 'all';
    this.historyMode = 'latest';
    this.historyPage = 1;
    this.historyPageSize = 10;
    this.loadRenders();
  }

  historySummaryLabel(): string {
    if (this.historyMode === 'latest') {
      return 'Mostrando le ultime 10 immagini';
    }
    return `Archivio: ${this.historyTotal} risultati`;
  }

  private buildHistoryQuery(): RenderHistoryQuery {
    if (this.historyMode !== 'search') {
      // Strict default mode: always show only latest 10.
      return {
        include_total: 1,
        page: this.historyPage,
        page_size: 10,
        limit: 10,
      };
    }

    const q = this.historyQuery.trim();
    const from = this.historyDateFrom;
    const to = this.historyDateTo;
    const normalizedFrom = from && to && from > to ? to : from;
    const normalizedTo = from && to && from > to ? from : to;
    return {
      include_total: 1,
      page: this.historyPage,
      page_size: this.historyPageSize,
      limit: this.historyPageSize,
      q: q || undefined,
      from: normalizedFrom || undefined,
      to: normalizedTo || undefined,
      type: this.historyTypeFilter !== 'all' ? this.historyTypeFilter : undefined,
      resolution: this.historyResolutionFilter !== 'all' ? this.historyResolutionFilter : undefined,
      backend: this.historyBackendFilter !== 'all' ? this.historyBackendFilter : undefined,
      chart: this.historyChartFilter !== 'all' ? this.historyChartFilter : undefined,
    };
  }

  private startStatusPolling() {
    if (this.statusPollTimer !== null) return;
    this.statusPollTimer = setInterval(() => {
      if (this.status() !== 'running') return;
      this.svc.getStatus().subscribe({
        next: s => {
          this.loadQueueState();
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
      next: rsp => {
        if (rsp.status === 'started') {
          this.status.set('running');
          this.progress.set(0);
          this.hasDeterminateProgress.set(false);
          this.elapsed.set(0);
          this.eta.set(0);
        } else {
          this.logLines.update(l => [...l.slice(-49), `Queued colorize job #${rsp.jobId ?? '?'} (pos ${rsp.queuePosition ?? '?'})`]);
        }
        this.loadQueueState();
      },
      error: err => {
        console.error(err);
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
    return this.renders;
  }

  getRenderMeta(renderOrName: RenderFile | string): RenderMetaLite {
    if (typeof renderOrName !== 'string') {
      const serverMeta = renderOrName.meta;
      if (serverMeta) {
        return {
          resolution: serverMeta.resolution,
          backend: serverMeta.backend,
          chart: serverMeta.chart,
        };
      }
    }

    const name = typeof renderOrName === 'string' ? renderOrName : renderOrName.name;
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
    if (lower.includes('_gks-') || lower.includes('_gks_')) chart = 'gks';
    else if (lower.includes('_ks-') || lower.includes('_ks_')) chart = 'ks';
    else if (lower.includes('_bl-') || lower.includes('_bl_')) chart = 'bl';

    return { resolution, backend, chart };
  }

  humanBackend(meta: RenderMetaLite): string {
    if (meta.backend === 'metal') return 'GPU Metal';
    if (meta.backend === 'cuda') return 'GPU CUDA';
    if (meta.backend === 'cpu') return 'CPU';
    return 'N/A';
  }

  historyTypeLabel(type: string): string {
    if (type === 'all') return 'Tipo: tutti';
    if (type === 'single_ray') return 'Tipo: single-ray';
    if (type === 'ray_bundle') return 'Tipo: ray-bundle';
    if (type === 'standard_rk4') return 'Tipo: standard RK4';
    if (type === 'semi_analytic') return 'Tipo: semi-analytic';
    if (type === 'elliptic_closed') return 'Tipo: elliptic-closed';
    return `Tipo: ${type}`;
  }

  queueStatusLabel(status: QueueJobState['status']): string {
    if (status === 'queued') return 'Queued';
    if (status === 'running') return 'Running';
    if (status === 'done') return 'Done';
    if (status === 'failed') return 'Failed';
    if (status === 'cancelled') return 'Cancelled';
    return status;
  }

  queueStatusClass(status: QueueJobState['status']): string {
    return `job-${status}`;
  }

  cancelQueuedJob(jobId: number) {
    this.cancelRender(jobId);
  }

  moveQueuedJob(jobId: number, direction: 'up' | 'down') {
    this.svc.reorderQueue(jobId, direction).subscribe({
      next: () => this.loadQueueState(),
      error: err => console.error(err),
    });
  }

  retryRecentJob(jobId: number) {
    this.svc.retryRecentJob(jobId).subscribe({
      next: rsp => {
        this.logLines.update(l => [...l.slice(-49), `Retry queued #${rsp.jobId ?? '?'} from recent #${jobId}`]);
        this.loadQueueState();
      },
      error: err => console.error(err),
    });
  }

  duplicateRecentJob(jobId: number) {
    this.svc.duplicateRecentJob(jobId).subscribe({
      next: rsp => {
        this.logLines.update(l => [...l.slice(-49), `Duplicate queued #${rsp.jobId ?? '?'} from recent #${jobId}`]);
        this.loadQueueState();
      },
      error: err => console.error(err),
    });
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
    const next = this.cloneParams(preset.params);
    if (!next.disk_radial_profile) next.disk_radial_profile = 'page_thorne';
    if (typeof (next as any).doppler_enabled !== 'boolean') next.doppler_enabled = true;
    if (typeof (next as any).disk_inner_emission_floor !== 'number') next.disk_inner_emission_floor = 0.0;
    if (typeof (next as any).disk_inner_emission_floor_width !== 'number') next.disk_inner_emission_floor_width = 0.25;
    if (typeof (next as any).radial_term_zero_torque !== 'boolean') next.radial_term_zero_torque = true;
    if (typeof (next as any).radial_term_r3_decay !== 'boolean') next.radial_term_r3_decay = true;
    if (typeof (next as any).radial_term_relativistic !== 'boolean') next.radial_term_relativistic = true;
    if (typeof (next as any).radial_term_b_denom !== 'boolean') next.radial_term_b_denom = true;
    this.params = next;
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

  fmtRate(v: number): string {
    if (!Number.isFinite(v) || v <= 0) return '0';
    return new Intl.NumberFormat('it-IT', { maximumFractionDigits: 0 }).format(v);
  }

  // ── Navigate ──────────────────────────────────────────────────

  navSyncFromParams() {
    this.navTheta = this.params.theta;
    this.navPhi   = ((this.params.phi % 360) + 360) % 360;
    this.sendNavigateNow();
  }

  navSyncToParams() {
    this.params.theta = this.navTheta;
    this.params.phi   = this.navPhi;
  }

  private navStartTimer() {
    if (this.navElapsedTimer) clearInterval(this.navElapsedTimer);
    this.navStartMs = Date.now();
    this.navElapsedMs.set(0);
    this.navElapsedTimer = setInterval(() => {
      this.navElapsedMs.set(Date.now() - this.navStartMs);
    }, 50);
  }

  private navStopTimer() {
    if (this.navElapsedTimer) { clearInterval(this.navElapsedTimer); this.navElapsedTimer = null; }
    const duration = Date.now() - this.navStartMs;
    if (duration > 50) {
      this.navAvgHistory.push(duration);
      if (this.navAvgHistory.length > 5) this.navAvgHistory.shift();
      const avg = Math.round(this.navAvgHistory.reduce((a, b) => a + b, 0) / this.navAvgHistory.length);
      this.navAvgMs.set(avg);
    }
  }

  loadNavBakes() {
    this.svc.getNavBakes().subscribe(r => {
      this.navBakeList.set(r.bakes ?? []);
    });
  }

  startNavBake() {
    const body: Record<string, unknown> = {
      theta_step: this.navBakeStepTheta(), phi_step: this.navBakeStepPhi(),
      a: this.params.a, disk_out: this.params.disk_out, r_obs: this.params.r_obs,
      fov: this.params.fov, disk_palette: this.navPalette ?? this.params.disk_palette,
      disk_brightness: this.params.disk_brightness,
      doppler_enabled: this.params.doppler_enabled,
      radial_term_zero_torque: this.params.radial_term_zero_torque,
      interstellar_p: this.params.interstellar_p,
      background: this.navBackground ?? this.params.background,
    };
    this.svc.startNavBake(body).subscribe(r => {
      if (r.bakeId) {
        this.navBaking.set(true);
        this.navBakeDone.set(0);
        this.navBakeTotal.set(r.total ?? 0);
      }
    });
  }

  cancelNavBake() {
    this.svc.cancelNavBake().subscribe();
    this.navBaking.set(false);
  }

  deleteNavBake(id: string) {
    this.svc.deleteNavBake(id).subscribe(() => this.loadNavBakes());
  }

  useNavBake(id: string) {
    this.navBakeId.set(id);
  }

  findBakedFrameUrl(theta: number, phi: number): string | null {
    const id = this.navBakeId();
    if (!id) return null;
    const bake = this.navBakeList().find(b => b.id === id);
    if (!bake) return null;
    const st = bake.thetaStep;
    const sp = bake.phiStep;
    const tSnap = Math.max(st, Math.min(180 - st, Math.round(theta / st) * st));
    const pSnap = ((Math.round(phi / sp) * sp) % 360 + 360) % 360;
    return this.svc.navBakeFrameUrl(id, tSnap, pSnap);
  }

  sendNavigateNow() {
    const bakedUrl = this.findBakedFrameUrl(this.navTheta, this.navPhi);
    if (bakedUrl) {
      this.navFrameUrl.set(bakedUrl);
      return;
    }
    this.navLoading.set(true);
    this.navStartTimer();
    this.svc.sendWs({
      type:                    'navigate',
      theta:                   this.navTheta,
      phi:                     this.navPhi,
      a:                       this.params.a,
      disk_out:                this.params.disk_out,
      r_obs:                   this.params.r_obs,
      fov:                     this.params.fov,
      disk_palette:            this.navPalette ?? this.params.disk_palette,
      disk_brightness:         this.params.disk_brightness,
      doppler_enabled:         this.params.doppler_enabled,
      radial_term_zero_torque: this.params.radial_term_zero_torque,
      interstellar_p:          this.params.interstellar_p,
      background:              this.navBackground ?? this.params.background,
    });
  }

  private scheduleNavigate() {
    if (this.navThrottle) return;
    this.navThrottle = setTimeout(() => {
      this.navThrottle = null;
      this.sendNavigateNow();
    }, 80);
  }

  onNavMouseDown(e: MouseEvent) {
    this.navDragging = true;
    this.navDragX0   = e.clientX;
    this.navDragY0   = e.clientY;
    this.navTheta0   = this.navTheta;
    this.navPhi0     = this.navPhi;
    (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
  }

  onNavMouseMove(e: MouseEvent) {
    if (!this.navDragging) return;
    const dx = e.clientX - this.navDragX0;
    const dy = e.clientY - this.navDragY0;
    this.navPhi   = ((this.navPhi0 - dx * this.navSensitivity) % 360 + 360) % 360;
    this.navTheta = Math.max(1, Math.min(179,
                    this.navTheta0 + dy * this.navSensitivity));
    this.scheduleNavigate();
  }

  onNavMouseUp(e: MouseEvent) {
    if (!this.navDragging) return;
    this.navDragging = false;
    (e.currentTarget as HTMLElement).style.cursor = 'grab';
    if (this.navThrottle) { clearTimeout(this.navThrottle); this.navThrottle = null; }
    this.sendNavigateNow();
  }

  onNavMouseLeave(e: MouseEvent) {
    if (this.navDragging) this.onNavMouseUp(e);
  }
}
