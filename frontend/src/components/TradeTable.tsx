import { useMemo, useState } from "react";
import type { Trade } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { formatMoney, formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

type Key = "entry_date" | "exit_date" | "return_pct" | "bars_held";

const REASON: Record<Trade["exit_reason"], string> = {
  signal: "Sell signal",
  stop: "Stop loss",
  target: "Take profit",
  max_hold: "Max hold",
  drawdown: "Drawdown limit",
  end: "End of test",
};

export function TradeTable({ trades }: { trades: Trade[] }) {
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: "entry_date", desc: true });
  const rows = useMemo(() => {
    const out = [...trades];
    out.sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      const c = x < y ? -1 : x > y ? 1 : 0;
      return sort.desc ? -c : c;
    });
    return out;
  }, [trades, sort]);
  const head = (key: Key, label: string, right = false) => (
    <th className={cn("px-3 py-2 font-medium", right && "text-right")}>
      <button
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }))}
        className="hover:text-text"
      >
        {label}
        {sort.key === key ? (sort.desc ? " ↓" : " ↑") : ""}
      </button>
    </th>
  );
  if (!trades.length) return <div className="p-6 text-center text-sm text-muted">No trades in this period.</div>;
  return (
    <div className="max-h-[420px] overflow-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-surface text-left text-xs text-muted">
          <tr className="border-b border-border">
            <th className="px-3 py-2 font-medium">Side</th>
            {head("entry_date", "Entry")}
            <th className="px-3 py-2 text-right font-medium">Entry price</th>
            {head("exit_date", "Exit")}
            <th className="px-3 py-2 text-right font-medium">Exit price</th>
            {head("return_pct", "Return", true)}
            {head("bars_held", "Days", true)}
            <th className="px-3 py-2 font-medium">Exit reason</th>
          </tr>
        </thead>
        <tbody className="num">
          {rows.map((t, i) => (
            <tr key={i} className="border-b border-border/60 hover:bg-surface-2/60">
              <td className="px-3 py-1.5 capitalize">{t.direction}</td>
              <td className="px-3 py-1.5">{t.entry_date}</td>
              <td className="px-3 py-1.5 text-right">{formatMoney(t.entry_price)}</td>
              <td className="px-3 py-1.5">{t.exit_date}</td>
              <td className="px-3 py-1.5 text-right">{formatMoney(t.exit_price)}</td>
              <td className={cn("px-3 py-1.5 text-right font-medium", signClass(t.return_pct))}>{formatPct(t.return_pct)}</td>
              <td className="px-3 py-1.5 text-right">{t.bars_held}</td>
              <td className="px-3 py-1.5"><Badge>{REASON[t.exit_reason]}</Badge></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
