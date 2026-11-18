import { Trash2 } from "lucide-react";
import type { Condition, IndicatorSpec, Op } from "@/api/types";
import { Select } from "@/components/ui/input";
import { OperandEditor } from "./OperandEditor";

export const OP_LABELS: Record<Op, string> = {
  ">": "is above",
  "<": "is below",
  crosses_above: "crosses above",
  crosses_below: "crosses below",
  rising: "is rising for",
  falling: "is falling for",
};

const NEEDS_RIGHT: Op[] = [">", "<", "crosses_above", "crosses_below"];

interface Props {
  value: Condition;
  onChange: (c: Condition) => void;
  onRemove: () => void;
  indicators: IndicatorSpec[];
}

export function ConditionRow({ value, onChange, onRemove, indicators }: Props) {
  const needsRight = NEEDS_RIGHT.includes(value.op);
  const setOp = (op: Op) => {
    const next: Condition = { ...value, op };
    if (NEEDS_RIGHT.includes(op)) next.right = value.right ?? { kind: "constant", value: 0 };
    else {
      next.right = null;
      next.bars = value.bars ?? 3;
    }
    onChange(next);
  };
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border bg-surface-2/50 p-2">
      <OperandEditor side="Left" value={value.left} indicators={indicators} allowConstant={false}
        onChange={(left) => onChange({ ...value, left })} />
      <Select aria-label="Comparison" value={value.op} onChange={(e) => setOp(e.target.value as Op)} className="h-8 text-sm font-medium">
        {(Object.keys(OP_LABELS) as Op[]).map((op) => (
          <option key={op} value={op}>
            {OP_LABELS[op]}
          </option>
        ))}
      </Select>
      {needsRight && value.right ? (
        <OperandEditor side="Right" value={value.right} indicators={indicators}
          onChange={(right) => onChange({ ...value, right })} />
      ) : (
        <span className="inline-flex items-center gap-1 text-sm">
          <input
            aria-label="Bars"
            type="number"
            min={1}
            max={100}
            value={value.bars ?? 1}
            onChange={(e) => onChange({ ...value, bars: Math.max(1, Number(e.target.value) || 1) })}
            className="h-8 w-14 rounded-md border border-border bg-surface-2 px-2 text-sm"
          />
          <span className="text-muted">bars</span>
        </span>
      )}
      <button type="button" onClick={onRemove} aria-label="Remove condition"
        className="ml-auto rounded-md p-1.5 text-muted hover:bg-surface hover:text-loss">
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}
