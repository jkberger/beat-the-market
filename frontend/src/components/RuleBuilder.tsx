import { Plus } from "lucide-react";
import type { Condition, Group, IndicatorSpec } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Segmented } from "@/components/ui/segmented";
import type { Side } from "@/lib/indicatorDefaults";
import { ConditionRow } from "./ConditionRow";

export function newCondition(): Condition {
  return { kind: "condition", left: { kind: "price", field: "close" }, op: ">", right: { kind: "constant", value: 0 }, bars: 1 };
}

interface Props {
  title: string;
  value: Group;
  onChange: (g: Group) => void;
  indicators: IndicatorSpec[];
  side: Side;
  depth?: number;
  onRemove?: () => void;
}

export function RuleBuilder({ title, value, onChange, indicators, side, depth = 0, onRemove }: Props) {
  const setItem = (i: number, item: Condition | Group) =>
    onChange({ ...value, items: value.items.map((x, j) => (j === i ? item : x)) });
  const removeItem = (i: number) => onChange({ ...value, items: value.items.filter((_, j) => j !== i) });

  return (
    <div className={depth ? "rounded-xl border border-dashed border-border p-3" : ""}>
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <span className="text-sm font-semibold">{title}</span>
        {value.items.length > 1 && (
          <Segmented
            label={`${title} join`}
            value={value.join}
            onChange={(join) => onChange({ ...value, join })}
            options={[
              { value: "and", label: "All (AND)" },
              { value: "or", label: "Any (OR)" },
            ]}
          />
        )}
        {onRemove && (
          <Button variant="ghost" size="sm" className="ml-auto" onClick={onRemove}>
            Remove group
          </Button>
        )}
      </div>
      <div className="flex flex-col gap-2">
        {value.items.map((item, i) =>
          item.kind === "group" ? (
            <RuleBuilder key={i} title="Group" value={item} indicators={indicators} side={side} depth={depth + 1}
              onChange={(g) => setItem(i, g)} onRemove={() => removeItem(i)} />
          ) : (
            <ConditionRow key={i} value={item} indicators={indicators} side={side}
              onChange={(c) => setItem(i, c)} onRemove={() => removeItem(i)} />
          ),
        )}
        {value.items.length === 0 && (
          <div className="rounded-xl border border-dashed border-border p-3 text-sm text-muted">
            No conditions yet.
          </div>
        )}
      </div>
      <div className="mt-2 flex gap-2">
        <Button variant="secondary" size="sm" onClick={() => onChange({ ...value, items: [...value.items, newCondition()] })}>
          <Plus className="h-3.5 w-3.5" /> Add condition
        </Button>
        {depth === 0 && (
          <Button variant="ghost" size="sm"
            onClick={() => onChange({ ...value, items: [...value.items, { kind: "group", join: "or", items: [newCondition()] }] })}>
            <Plus className="h-3.5 w-3.5" /> Add group
          </Button>
        )}
      </div>
    </div>
  );
}
