import { useQueries, useQuery } from "@tanstack/react-query";
import { ArrowRight, FlaskConical, LineChart, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { TickerSearch } from "@/components/TickerSearch";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatMoney, formatPct, signClass } from "@/lib/format";
import { cn } from "@/lib/utils";

function Sparkline({ values, up }: { values: number[]; up: boolean }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${30 - ((v - min) / (max - min || 1)) * 28}`).join(" ");
  return (
    <svg viewBox="0 0 100 32" className="h-10 w-full" preserveAspectRatio="none">
      <polyline points={pts} fill="none" strokeWidth="1.5" className={up ? "stroke-gain" : "stroke-loss"} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function HomePage() {
  const universe = useQuery({ queryKey: ["universe"], queryFn: api.universe });
  const picks = universe.data?.quick_picks ?? [];
  const quotes = useQueries({
    queries: picks.map((s) => ({ queryKey: ["bars", s], queryFn: () => api.bars(s) })),
  });
  const runs = useQuery({ queryKey: ["backtests"], queryFn: api.listBacktests });
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: api.listStrategies });

  return (
    <div className="flex flex-col gap-8">
      <section className="relative overflow-hidden rounded-3xl border border-border bg-gradient-to-br from-accent/20 via-surface to-surface p-8 sm:p-10">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Find out which signals really work.</h1>
          <p className="mt-3 text-muted">
            Build buy and sell rules from 25 indicators, then test them the honest way: on data the strategy
            never saw, side by side with simply holding the stock.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Link to="/build"><Button size="lg">Build a strategy <ArrowRight className="h-4 w-4" /></Button></Link>
            <TickerSearch className="w-64" />
          </div>
        </div>
        <div className="mt-8 grid gap-4 text-sm sm:grid-cols-3">
          {[
            { icon: LineChart, title: "25 indicators", text: "Trend, momentum, volatility and volume, combined with AND / OR." },
            { icon: FlaskConical, title: "Walk-forward testing", text: "Tune on the past, trade the next six months blind, repeat." },
            { icon: ShieldCheck, title: "Locked holdout", text: "A final slice of history no tuning ever touches." },
          ].map((f) => (
            <div key={f.title} className="flex gap-3">
              <f.icon className="mt-0.5 h-5 w-5 shrink-0 text-accent" />
              <div><div className="font-medium">{f.title}</div><div className="text-muted">{f.text}</div></div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold text-muted">Popular stocks</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {picks.map((s, i) => {
            const bars = quotes[i]?.data?.bars ?? [];
            const last = bars[bars.length - 1];
            const prev = bars[bars.length - 2];
            const change = last && prev ? last.close / prev.close - 1 : null;
            const recent = bars.slice(-90).map((b) => b.close);
            return (
              <Link key={s} to={`/stock/${s}`}>
                <Card className="p-4 transition-colors hover:border-accent/50">
                  <div className="flex items-baseline justify-between">
                    <span className="font-semibold">{s}</span>
                    <span className={cn("text-xs num", signClass(change))}>{change === null ? "" : `${change >= 0 ? "+" : ""}${formatPct(change, 2)}`}</span>
                  </div>
                  <div className="num text-lg">{last ? formatMoney(last.close) : quotes[i]?.error ? "unavailable" : "…"}</div>
                  <Sparkline values={recent} up={recent.length > 1 && recent[recent.length - 1] >= recent[0]} />
                </Card>
              </Link>
            );
          })}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Recent backtests</CardTitle><CardDescription>Click one to see the full results.</CardDescription></CardHeader>
          <CardContent>
            {runs.data?.length ? (
              <ul className="divide-y divide-border">
                {runs.data.map((r) => (
                  <li key={r.id}>
                    <Link to={`/results/${r.id}`} className="flex items-center justify-between gap-3 py-2 text-sm hover:text-accent">
                      <span><span className="font-medium">{r.strategy_name}</span> <span className="text-muted">on {r.symbol}</span></span>
                      <span className={cn("num", signClass(r.summary?.median.cagr))}>{formatPct(r.summary?.median.cagr ?? null)} CAGR</span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">No backtests yet. Build a strategy to run your first one.</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Saved strategies</CardTitle></CardHeader>
          <CardContent>
            {strategies.data?.length ? (
              <ul className="divide-y divide-border">
                {strategies.data.map((s) => (
                  <li key={s.id}>
                    <Link to={`/build?strategy=${s.id}`} className="flex items-center justify-between py-2 text-sm hover:text-accent">
                      <span className="font-medium">{s.name}</span>
                      <span className="text-xs text-muted">{s.created_at.slice(0, 10)}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">Strategies you save in the builder show up here.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
