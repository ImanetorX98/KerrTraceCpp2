import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { Subscription } from 'rxjs';

import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSelectModule } from '@angular/material/select';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';

import { RenderService, RenderParams, RenderFile, ApiInfo } from './render.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, FormsModule, HttpClientModule,
    MatSliderModule, MatButtonModule, MatButtonToggleModule,
    MatProgressBarModule, MatSelectModule, MatSlideToggleModule,
    MatIconModule, MatTooltipModule,
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
    disk_out: 25,
    theta: 80,
    r_obs: 30,
    bundles: false,
    dopri5: false,
    background: '',
  };

  // ── State signals ─────────────────────────────────────────────
  readonly status   = signal<'idle' | 'running' | 'done' | 'error'>('idle');
  readonly progress = signal(0);
  readonly elapsed  = signal(0);
  readonly eta      = signal(0);
  readonly logLines = signal<string[]>([]);

  // ── Gallery ───────────────────────────────────────────────────
  renders: RenderFile[] = [];
  activeRender: string | null = null;

  readonly previewUrl = computed(() => {
    if (!this.activeRender) return null;
    return this.svc.renderUrl(this.activeRender);
  });

  private sub: Subscription | null = null;

  constructor(readonly svc: RenderService) {}

  ngOnInit() {
    this.svc.getInfo().subscribe(info => {
      this.info = info;
    });
    this.loadRenders();

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
            this.activeRender = msg.file;
            setTimeout(() => this.loadRenders(), 500);
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
      if (!this.activeRender && files.length > 0) {
        this.activeRender = files[0].name;
      }
    });
  }

  selectRender(name: string) {
    this.activeRender = name;
  }

  fmt(v: number, d = 2): string {
    return v.toFixed(d);
  }

  fmtSize(bytes: number): string {
    if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    return (bytes / 1e3).toFixed(0) + ' KB';
  }
}
