import { useEffect, useState } from "react";
import type { ParamValue, RangeSpec } from "@/api/types";
import { cn } from "@/lib/utils";

export function isRange(v: ParamValue | undefined): v is RangeSpec {
  return typeof v === "object" && v !== null;
}

export function defaultRange(v: number): RangeSpec {
  const step = Math.abs(v) >= 10 ? 5 : Math.abs(v) >= 2 ? 1 : 0.5;
  return { min: v, max: v + 2 * step, step };
}

function NumberBox({
  value,
  onChange,
  label,
  className,
}: {
  value: number;
  onChange: (v: number) => void;
  label: string;
  className?: string;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => {
    if (Number(text) !== value) setText(String(value));
  }, [value]);
  return (
    <input
      aria-label={label}
      inputMode="decimal"
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(n)) onChange(n);
      }}
      className={cn(
        "h-8 w-16 rounded-md border border-border bg-surface-2 px-2 text-sm num focus:outline-none focus:ring-2 focus:ring-accent/50",
        className,
      )}
    />
  );
}

interface Props {
  value: ParamValue;
  onChange: (v: ParamValue) => void;
  label: string;
  allowRange?: boolean;
}

/** A number that can be flipped into a min/max/step range for tuning. */
export function ParamInput({ value, onChange, label, allowRange = true }: Props) {
  const ranged = isRange(value);
  return (
    <span className="inline-flex items-center gap-1">
      {ranged ? (
        <span className="inline-flex items-center gap-1 rounded-md bg-accent/10 px-1 py-0.5">
          <NumberBox label={`${label} min`} value={value.min} onChange={(min) => onChange({ ...value, min })} />
          <span className="text-xs text-muted">to</span>
          <NumberBox label={`${label} max`} value={value.max} onChange={(max) => onChange({ ...value, max })} />
          <span className="text-xs text-muted">by</span>
          <NumberBox label={`${label} step`} value={value.step} onChange={(step) => onChange({ ...value, step })} className="w-12" />
        </span>
      ) : (
        <NumberBox label={label} value={value} onChange={onChange} />
      )}
      {allowRange && (
        <button
          type="button"
          role="switch"
          aria-checked={ranged}
          aria-label={`${label} range`}
          title={ranged ? "Use a single value" : "Tune over a range (walk-forward)"}
          onClick={() => onChange(ranged ? value.min : defaultRange(value))}
          className={cn(
            "rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
            ranged ? "bg-accent text-white" : "text-muted hover:bg-surface-2 hover:text-text",
          )}
        >
          range
        </button>
      )}
    </span>
  );
}
