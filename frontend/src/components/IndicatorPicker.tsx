import { Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { IndicatorSpec } from "@/api/types";

const CATEGORY_LABEL: Record<IndicatorSpec["category"], string> = {
  trend: "Trend",
  momentum: "Momentum",
  volatility: "Volatility",
  volume: "Volume",
};

export function defaultParams(spec: IndicatorSpec): Record<string, number> {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
}

interface Props {
  indicators: IndicatorSpec[];
  onAdd: (id: string, params: Record<string, number>) => void;
}

export function IndicatorPicker({ indicators, onAdd }: Props) {
  const [query, setQuery] = useState("");
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = indicators.filter(
      (i) => !q || i.id.includes(q) || i.name.toLowerCase().includes(q),
    );
    return (Object.keys(CATEGORY_LABEL) as IndicatorSpec["category"][])
      .map((cat) => ({ cat, items: match.filter((i) => i.category === cat) }))
      .filter((g) => g.items.length);
  }, [indicators, query]);

  return (
    <div className="flex flex-col gap-3">
      <div className="relative">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search indicators"
          className="h-9 w-full rounded-lg border border-border bg-surface-2 pl-9 pr-3 text-sm placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50"
        />
      </div>
      <div className="flex max-h-[420px] flex-col gap-4 overflow-y-auto pr-1">
        {groups.map((g) => (
          <div key={g.cat}>
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">
              {CATEGORY_LABEL[g.cat]}
            </div>
            <div className="flex flex-col">
              {g.items.map((i) => (
                <button
                  key={i.id}
                  type="button"
                  onClick={() => onAdd(i.id, defaultParams(i))}
                  className="group flex items-center justify-between rounded-lg px-2 py-1.5 text-left text-sm hover:bg-surface-2"
                >
                  <span>
                    <span className="font-medium">{i.id.toUpperCase()}</span>
                    <span className="ml-2 text-xs text-muted">{i.name}</span>
                  </span>
                  <Plus className="h-4 w-4 text-muted opacity-0 group-hover:opacity-100" />
                </button>
              ))}
            </div>
          </div>
        ))}
        {groups.length === 0 && <div className="text-sm text-muted">No indicators match.</div>}
      </div>
    </div>
  );
}
