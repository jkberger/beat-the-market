import { useState } from "react";
import type { RiskPreset, RiskSettings } from "@/api/types";
import { Segmented } from "@/components/ui/segmented";
import { Select } from "@/components/ui/input";

// Mirrors backend app/strategy/risk.py (spec section 9).
export const RISK_PRESETS: Record<Exclude<RiskPreset, "custom">, RiskSettings> = {
  conservative: { preset: "conservative", position_pct: 25, stop_atr_mult: 2, max_drawdown_pct: 10, standaside_bars: 20, objective: "sortino" },
  balanced: { preset: "balanced", position_pct: 50, stop_atr_mult: 3, max_drawdown_pct: 20, standaside_bars: 20, objective: "sharpe" },
  aggressive: { preset: "aggressive", position_pct: 100, stop_atr_mult: null, max_drawdown_pct: null, standaside_bars: 20, objective: "cagr" },
};

const OBJECTIVE_LABEL = {
  sortino: "Steady returns (Sortino, penalizes drawdowns)",
  sharpe: "Risk-adjusted return (Sharpe)",
  cagr: "Maximum total return (CAGR)",
};

const DESCRIPTION: Record<RiskPreset, string> = {
  conservative: "Small positions, tight stops, steps aside after a 10% drop.",
  balanced: "Half-size positions, moderate stops, steps aside after a 20% drop.",
  aggressive: "All-in on every signal, no stop, no drawdown limit.",
  custom: "Your own mix.",
};

function OptionalNumber({
  label,
  value,
  onChange,
  suffix,
}: {
  label: string;
  value: number | null;
  onChange: (v: number | null) => void;
  suffix: string;
}) {
  const [text, setText] = useState(value === null ? "" : String(value));
  return (
    <label className="flex items-center justify-between gap-3 text-sm">
      <span className="text-muted">{label}</span>
      <span className="flex items-center gap-1.5">
        <input
          aria-label={label}
          inputMode="decimal"
          value={text}
          placeholder="off"
          onChange={(e) => {
            setText(e.target.value);
            const raw = e.target.value.trim();
            if (raw === "") onChange(null);
            else if (Number.isFinite(Number(raw)) && Number(raw) > 0) onChange(Number(raw));
          }}
          className="h-8 w-20 rounded-md border border-border bg-surface-2 px-2 text-right text-sm num"
        />
        <span className="w-10 text-xs text-muted">{suffix}</span>
      </span>
    </label>
  );
}

export function RiskControl({ value, onChange }: { value: RiskSettings; onChange: (r: RiskSettings) => void }) {
  const edit = (patch: Partial<RiskSettings>) => onChange({ ...value, ...patch, preset: "custom" });
  return (
    <div className="flex flex-col gap-3">
      <Segmented
        className="flex w-full"
        label="Risk tolerance"
        value={value.preset}
        onChange={(p) => (p === "custom" ? onChange({ ...value, preset: "custom" }) : onChange(RISK_PRESETS[p]))}
        options={[
          { value: "conservative", label: "Conservative" },
          { value: "balanced", label: "Balanced" },
          { value: "aggressive", label: "Aggressive" },
          { value: "custom", label: "Custom" },
        ]}
      />
      <p className="text-xs text-muted">{DESCRIPTION[value.preset]}</p>
      {value.preset === "custom" ? (
        <div className="flex flex-col gap-2">
          <OptionalNumber label="Position size (% of equity)" suffix="%" value={value.position_pct}
            onChange={(v) => v !== null && edit({ position_pct: Math.min(100, v) })} />
          <OptionalNumber label="Stop loss (× ATR)" suffix="× ATR" value={value.stop_atr_mult}
            onChange={(v) => edit({ stop_atr_mult: v })} />
          <OptionalNumber label="Stand aside after drawdown of" suffix="%" value={value.max_drawdown_pct}
            onChange={(v) => edit({ max_drawdown_pct: v })} />
          <label className="flex items-center justify-between gap-3 text-sm">
            <span className="text-muted">Tune for</span>
            <Select aria-label="Tuning objective" value={value.objective}
              onChange={(e) => edit({ objective: e.target.value as RiskSettings["objective"] })} className="h-8 text-sm">
              {Object.entries(OBJECTIVE_LABEL).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </Select>
          </label>
        </div>
      ) : (
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <dt className="text-muted">Position size</dt>
          <dd className="text-right num">{value.position_pct}%</dd>
          <dt className="text-muted">Stop loss</dt>
          <dd className="text-right num">{value.stop_atr_mult ? `${value.stop_atr_mult} × ATR` : "none"}</dd>
          <dt className="text-muted">Drawdown limit</dt>
          <dd className="text-right num">{value.max_drawdown_pct ? `${value.max_drawdown_pct}%` : "none"}</dd>
          <dt className="text-muted">Tunes for</dt>
          <dd className="text-right">{value.objective.toUpperCase()}</dd>
        </dl>
      )}
    </div>
  );
}
