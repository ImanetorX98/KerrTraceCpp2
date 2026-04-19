import {
  Component, Input, Output, EventEmitter,
  ElementRef, ViewChild, AfterViewChecked,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'num-input',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="ni-wrap">
      <button class="ni-btn" (click)="decrement()" [disabled]="value <= min" tabindex="-1">−</button>

      <span class="ni-val" *ngIf="!editing" (click)="startEdit()">
        {{ display }}<span class="ni-unit" *ngIf="unit"> {{unit}}</span>
      </span>

      <input class="ni-edit" *ngIf="editing"
             #editInput
             type="number"
             [min]="min" [max]="max" [step]="step"
             [(ngModel)]="editValue"
             (blur)="commitEdit()"
             (keydown.enter)="commitEdit()"
             (keydown.escape)="cancelEdit()">

      <button class="ni-btn" (click)="increment()" [disabled]="value >= max" tabindex="-1">+</button>
    </div>
  `,
  styles: [`
    .ni-wrap {
      display: inline-flex;
      align-items: center;
      gap: 0;
      height: 30px;
      border: 1px solid #2a2a2a;
      border-radius: 5px;
      overflow: hidden;
      background: #0e0e0e;
    }

    .ni-btn {
      width: 28px;
      height: 100%;
      background: transparent;
      border: none;
      color: #888;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background .12s, color .12s;
      flex-shrink: 0;
      user-select: none;
      padding: 0;
      line-height: 1;

      &:hover:not(:disabled) {
        background: #1c1c1c;
        color: #f0f0f0;
      }

      &:active:not(:disabled) {
        background: #252525;
      }

      &:disabled {
        color: #333;
        cursor: not-allowed;
      }
    }

    .ni-val {
      flex: 1;
      min-width: 64px;
      text-align: center;
      font-size: 13px;
      font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
      font-variant-numeric: tabular-nums;
      color: #f0f0f0;
      cursor: text;
      user-select: none;
      padding: 0 4px;
      white-space: nowrap;

      &:hover {
        color: #fff;
        background: #161616;
      }

      .ni-unit {
        font-size: 11px;
        color: #555;
        margin-left: 2px;
      }
    }

    .ni-edit {
      flex: 1;
      min-width: 64px;
      height: 100%;
      background: #111;
      border: none;
      border-left: 1px solid #333;
      border-right: 1px solid #333;
      color: #fff;
      font-size: 13px;
      font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
      text-align: center;
      outline: none;
      padding: 0 4px;

      /* hide browser number spinners */
      &::-webkit-inner-spin-button,
      &::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
      -moz-appearance: textfield;
    }
  `],
})
export class NumericInputComponent implements AfterViewChecked {
  @Input() value    = 0;
  @Input() min      = 0;
  @Input() max      = 100;
  @Input() step     = 1;
  @Input() decimals = 2;
  @Input() unit     = '';

  @Output() valueChange = new EventEmitter<number>();

  @ViewChild('editInput') editInputRef?: ElementRef<HTMLInputElement>;

  editing   = false;
  editValue = '';
  private needsFocus = false;

  get display(): string {
    return this.value.toFixed(this.decimals);
  }

  startEdit() {
    this.editValue  = this.value.toFixed(this.decimals);
    this.editing    = true;
    this.needsFocus = true;
  }

  ngAfterViewChecked() {
    if (this.needsFocus && this.editInputRef) {
      this.editInputRef.nativeElement.focus();
      this.editInputRef.nativeElement.select();
      this.needsFocus = false;
    }
  }

  commitEdit() {
    const v = parseFloat(this.editValue);
    if (!isNaN(v)) this.emit(v);
    this.editing = false;
  }

  cancelEdit() {
    this.editing = false;
  }

  increment() { this.emit(this.value + this.step); }
  decrement() { this.emit(this.value - this.step); }

  private emit(v: number) {
    const factor = Math.pow(10, this.decimals + 2);
    v = Math.round(v * factor) / factor;
    v = Math.min(this.max, Math.max(this.min, v));
    // re-round to decimals after clamp
    const f2 = Math.pow(10, this.decimals);
    v = Math.round(v * f2) / f2;
    this.valueChange.emit(v);
  }
}
