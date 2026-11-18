import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Play, Save } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import type { BacktestRequest, RiskSettings, StopSpec, Strategy } from "@/api/types";
import { ErrorBox, Spinner } from "@/components/ErrorBox";
import { ParamInput } from "@/components/ParamInput";
import { RISK_PRESETS, RiskControl } from "@/components/RiskControl";
import { RuleBuilder } from "@/components/RuleBuilder";
import { SymbolPicker } from "@/components/SymbolPicker";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/input";
import { Segmented } from "@/components/ui/segmented";
import { Switch } from "@/components/ui/switch";
import { DEFAULT_STRATEGY, hasRanges } from "@/lib/defaults";

function useDebounced<T>(value: T, ms = 400): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function StopField({ label, value, onChange }: { label: string; value: StopSpec | null | undefined; onChange: (s: StopSpec | null) => void }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
      <Switch label={label} checked={!!value} onChange={(on) => onChange(on ? { kind: "percent", value: 5 } : null)} />
      {value && (
        <span className="inline-flex items-center gap-1.5">
          <ParamInput label={`${label} value`} value={value.value} onChange={(v) => onChange({ ...value, value: v })} />
          <Select aria-label={`${label} unit`} value={value.kind} className="h-8 text-xs"
            onChange={(e) => onChange({ ...value, kind: e.target.value as StopSpec["kind"] })}>
            <option value="percent">%</option>
            <option value="atr">× ATR</option>
          </Select>
        </span>
      )}
    </div>
  );
}

export function BuilderPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [strategy, setStrategy] = useState<Strategy>(DEFAULT_STRATEGY);
  const [symbols, setSymbols] = useState<string[]>([params.get("symbol")?.toUpperCase() || "AAPL"]);
  const [risk, setRisk] = useState<RiskSettings>(RISK_PRESETS.balanced);
  const [slippage, setSlippage] = useState(0);
  const [mode, setMode] = useState<BacktestRequest["mode"]>("walk_forward");
  const [scoreHoldout, setScoreHoldout] = useState(false);
  const [training, setTraining] = useState<"all" | "recent">("all");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");

  const specsQ = useQuery({ queryKey: ["indicators"], queryFn: api.indicators });
  const universeQ = useQuery({ queryKey: ["universe"], queryFn: api.universe });
  const settingsQ = useQuery({ queryKey: ["settings"], queryFn: api.getSettings });

  const savedId = params.get("strategy");
  useEffect(() => {
    if (savedId) api.getStrategy(Number(savedId)).then((r) => setStrategy(r.strategy)).catch(() => {});
  }, [savedId]);

  const debounced = useDebounced(strategy);
  const ready = debounced.buy.items.length > 0 && debounced.sell.items.length > 0;
  const describeQ = useQuery({
    queryKey: ["describe", debounced],
    queryFn: () => api.describe(debounced),
    enabled: ready,
    retry: false,
  });

  const ranged = hasRanges(strategy);
  const run = useMutation({
    mutationFn: () =>
      api.runBacktest({
        symbols,
        strategy,
        risk,
        config: { initial_equity: 10_000, slippage_bps: slippage },
        mode,
        start: start || null,
        end: end || null,
        score_holdout: mode === "walk_forward" && scoreHoldout,
        training,
      }),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["backtests"] });
      navigate(`/results/${r.id}`);
    },
  });
  const save = useMutation({
    mutationFn: () => api.saveStrategy(strategy),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["strategies"] }),
  });

  if (specsQ.isLoading) return <Spinner />;
  if (specsQ.error) return <ErrorBox error={specsQ.error} />;
  const indicators = specsQ.data!;
  const canRun = symbols.length > 0 && strategy.buy.items.length > 0 && strategy.sell.items.length > 0 && !(ranged && mode === "simple");

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Strategy builder</h1>
          <p className="text-sm text-muted">Combine indicators into buy and sell rules, then see how trading in and out would really have done.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => save.mutate()} disabled={save.isPending}>
            <Save className="h-4 w-4" /> {save.isSuccess ? "Saved" : "Save"}
          </Button>
          <Button size="lg" onClick={() => run.mutate()} disabled={!canRun || run.isPending}>
            <Play className="h-4 w-4" /> {run.isPending ? "Running…" : "Run backtest"}
          </Button>
        </div>
      </div>

      {run.error && <ErrorBox error={run.error} />}

      <Card className="border-accent/30 bg-accent/5">
        <CardContent className="flex flex-col gap-1 pt-5">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-accent">In plain English</span>
          <p className="text-base">
            {!ready ? "Add at least one buy condition and one sell condition." : describeQ.data?.text ?? (describeQ.error ? (describeQ.error as Error).message : "…")}
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <Input aria-label="Strategy name" value={strategy.name} onChange={(e) => setStrategy({ ...strategy, name: e.target.value })}
                className="h-10 max-w-sm border-transparent bg-transparent px-0 text-lg font-semibold focus:border-border focus:bg-surface-2 focus:px-3" />
              <CardDescription>Starts in cash. A buy signal goes in, a sell signal gets out, and it repeats every time the signals fire.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-6">
              <RuleBuilder title="Buy when" side="buy" value={strategy.buy} indicators={indicators} onChange={(buy) => setStrategy({ ...strategy, buy })} />
              <div className="h-px bg-border" />
              <RuleBuilder title="Sell when" side="sell" value={strategy.sell} indicators={indicators} onChange={(sell) => setStrategy({ ...strategy, sell })} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Exits and shorting</CardTitle>
              <CardDescription>Optional. A stop or target set here overrides the risk preset's stop.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <StopField label="Stop loss" value={strategy.stop_loss} onChange={(stop_loss) => setStrategy({ ...strategy, stop_loss })} />
              <StopField label="Take profit" value={strategy.take_profit} onChange={(take_profit) => setStrategy({ ...strategy, take_profit })} />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Switch label="Exit after a max holding period" checked={!!strategy.max_hold_bars}
                  onChange={(on) => setStrategy({ ...strategy, max_hold_bars: on ? 20 : null })} />
                {strategy.max_hold_bars && (
                  <span className="inline-flex items-center gap-1.5 text-sm">
                    <input aria-label="Max holding bars" type="number" min={1} value={strategy.max_hold_bars}
                      onChange={(e) => setStrategy({ ...strategy, max_hold_bars: Math.max(1, Number(e.target.value) || 1) })}
                      className="h-8 w-16 rounded-md border border-border bg-surface-2 px-2 text-sm" />
                    <span className="text-muted">days</span>
                  </span>
                )}
              </div>
              <Switch label="Short on sell signals (profit from drops too)" checked={strategy.allow_short}
                onChange={(allow_short) => setStrategy({ ...strategy, allow_short })} />
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader><CardTitle>Stocks</CardTitle><CardDescription>Each stock gets its own backtest, then a side-by-side comparison.</CardDescription></CardHeader>
            <CardContent>
              <SymbolPicker value={symbols} onChange={setSymbols} suggestions={universeQ.data?.quick_picks ?? []} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Risk tolerance</CardTitle><CardDescription>Applies to testing and to tuning.</CardDescription></CardHeader>
            <CardContent><RiskControl value={risk} onChange={setRisk} /></CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>How to test it</CardTitle></CardHeader>
            <CardContent className="flex flex-col gap-3">
              <Segmented label="Validation mode" value={mode} onChange={setMode}
                options={[{ value: "walk_forward", label: "Walk-forward" }, { value: "simple", label: "Simple" }]} />
              <p className="text-xs text-muted">
                {mode === "walk_forward"
                  ? "Tunes on past data, trades the next 6 months blind, slides forward 6 months and repeats, so you get many blind tests instead of one lucky one."
                  : "Runs your fixed rules over the dates you pick. Ranges need walk-forward."}
              </p>
              {ranged && mode === "simple" && <p className="text-xs text-loss">This strategy has ranges to tune. Switch to walk-forward.</p>}
              {mode === "walk_forward" && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs text-muted">Each test window trains on</span>
                  <Segmented label="Training history" value={training} onChange={setTraining} className="flex w-full"
                    options={[{ value: "all", label: "All history before it" }, { value: "recent", label: "Only the last 3 years" }]} />
                </div>
              )}
              {mode === "walk_forward" && (
                <div className="flex flex-col gap-1">
                  <Switch label="Also score the locked holdout" checked={scoreHoldout} onChange={setScoreHoldout} />
                  <p className="pl-11 text-xs text-muted">
                    Holdout starts {settingsQ.data?.holdout_start ?? "…"}. It's never used for tuning. Scoring it often turns it into training data, so the app counts each time.
                  </p>
                </div>
              )}
              <div className="grid grid-cols-2 gap-2">
                <label className="flex flex-col gap-1 text-xs text-muted">From
                  <Input type="date" value={start} onChange={(e) => setStart(e.target.value)} aria-label="Start date" />
                </label>
                <label className="flex flex-col gap-1 text-xs text-muted">To
                  <Input type="date" value={end} onChange={(e) => setEnd(e.target.value)} aria-label="End date" disabled={mode === "walk_forward"} />
                </label>
              </div>
              <label className="flex items-center justify-between gap-2 text-sm">
                <span className="text-muted">Slippage per trade</span>
                <span className="inline-flex items-center gap-1">
                  <input aria-label="Slippage basis points" type="number" min={0} max={500} value={slippage}
                    onChange={(e) => setSlippage(Math.max(0, Number(e.target.value) || 0))}
                    className="h-8 w-16 rounded-md border border-border bg-surface-2 px-2 text-right text-sm" />
                  <span className="text-xs text-muted">bps</span>
                </span>
              </label>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
