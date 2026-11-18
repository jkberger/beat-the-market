import { useQueries, useQuery } from "@tanstack/react-query";
import { ArrowRight, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api/client";
import { ErrorBox, Spinner } from "@/components/ErrorBox";
import { IndicatorPicker } from "@/components/IndicatorPicker";
import { PriceChart, type LineData, type Pane } from "@/components/PriceChart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented } from "@/components/ui/segmented";
import { formatMoney, formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

type RangeKey = "6M" | "1Y" | "5Y" | "MAX";
const RANGE_DAYS: Record<RangeKey, number | null> = { "6M": 182, "1Y": 365, "5Y": 1826, MAX: null };

interface Chosen {
  key: string;
  id: string;
  params: Record<string, number>;
}

function label(c: Chosen) {
  const vals = Object.values(c.params);
  return `${c.id.toUpperCase()}${vals.length ? `(${vals.join(", ")})` : ""}`;
}

export function StockPage() {
  const symbol = (useParams().symbol ?? "").toUpperCase();
  const [range, setRange] = useState<RangeKey>("1Y");
  const [chosen, setChosen] = useState<Chosen[]>([
    { key: "sma-50", id: "sma", params: { length: 50 } },
    { key: "rsi-14", id: "rsi", params: { length: 14 } },
  ]);

  const barsQ = useQuery({ queryKey: ["bars", symbol], queryFn: () => api.bars(symbol) });
  const specsQ = useQuery({ queryKey: ["indicators"], queryFn: api.indicators });

  const start = useMemo(() => {
    const days = RANGE_DAYS[range];
    const all = barsQ.data?.bars;
    if (!days || !all?.length) return undefined;
    const last = new Date(all[all.length - 1].date);
    last.setDate(last.getDate() - days);
    return last.toISOString().slice(0, 10);
  }, [range, barsQ.data]);

  const visibleBars = useMemo(
    () => (barsQ.data?.bars ?? []).filter((b) => !start || b.date >= start),
    [barsQ.data, start],
  );

  const indicatorQs = useQueries({
    queries: chosen.map((c) => ({
      queryKey: ["ind", symbol, c.id, c.params, start],
      queryFn: () => api.computeIndicator(symbol, c.id, c.params, start),
      enabled: !!barsQ.data,
    })),
  });

  const { overlays, panes } = useMemo(() => {
    const overlays: LineData[] = [];
    const panes: Pane[] = [];
    chosen.forEach((c, i) => {
      const data = indicatorQs[i]?.data;
      const spec = specsQ.data?.find((s) => s.id === c.id);
      if (!data || !spec) return;
      const lines = spec.outputs.map((o) => ({
        name: spec.outputs.length > 1 ? `${label(c)} ${o}` : label(c),
        dates: data.dates,
        values: data.series[o],
      }));
      if (spec.overlay) overlays.push(...lines);
      else panes.push({ name: label(c), series: lines });
    });
    return { overlays, panes };
  }, [chosen, specsQ.data, indicatorQs.map((q) => q.dataUpdatedAt).join(",")]);

  if (barsQ.isLoading) return <Spinner label={`Loading ${symbol}…`} />;
  if (barsQ.error) return <ErrorBox error={barsQ.error} />;
  const all = barsQ.data!.bars;
  const last = all[all.length - 1];
  const prev = all[all.length - 2] ?? last;
  const change = last.close / prev.close - 1;
  const first = visibleBars[0] ?? last;
  const rangeChange = last.close / first.close - 1;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-sm text-muted">Daily · as of {last.date}</div>
          <h1 className="text-3xl font-semibold tracking-tight">{symbol}</h1>
          <div className="mt-1 flex items-baseline gap-3 num">
            <span className="text-2xl font-medium">{formatMoney(last.close)}</span>
            <span className={cn("text-sm font-medium", signClass(change))}>
              {change >= 0 ? "+" : ""}
              {formatPct(change, 2)} today
            </span>
            <span className={cn("text-sm", signClass(rangeChange))}>
              {rangeChange >= 0 ? "+" : ""}
              {formatPct(rangeChange)} over {range === "MAX" ? "all history" : range}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Segmented
            label="Chart range"
            value={range}
            onChange={setRange}
            options={(["6M", "1Y", "5Y", "MAX"] as RangeKey[]).map((r) => ({ value: r, label: r }))}
          />
          <Link to={`/build?symbol=${symbol}`}>
            <Button>
              Build a strategy <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <Card className="p-3">
          <PriceChart bars={visibleBars} overlays={overlays} panes={panes} />
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Indicators</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-wrap gap-2">
              {chosen.map((c) => (
                <span
                  key={c.key}
                  className="inline-flex items-center gap-1 rounded-md border border-border bg-surface-2 py-1 pl-2 pr-1 text-xs"
                >
                  {label(c)}
                  <button
                    aria-label={`Remove ${label(c)}`}
                    onClick={() => setChosen((cs) => cs.filter((x) => x.key !== c.key))}
                    className="rounded p-0.5 text-muted hover:bg-surface hover:text-text"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
              {chosen.length === 0 && <span className="text-xs text-muted">None yet. Add one below.</span>}
            </div>
            {specsQ.data && (
              <IndicatorPicker
                indicators={specsQ.data}
                onAdd={(id, params) =>
                  setChosen((cs) => [...cs, { key: `${id}-${Date.now()}`, id, params }])
                }
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
