import { ColorType, createChart, LineStyle, type Time } from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { Point } from "@/api/types";
import { useChartTheme } from "@/lib/theme";

export interface EquityLine {
  name: string;
  points: Point[];
  color?: string;
  dashed?: boolean;
}

/** Rebase a baseline so it starts at `start` value on the first date >= from. */
export function rebase(points: Point[], from: string, to: string, startValue: number): Point[] {
  const slice = points.filter((p) => p.date >= from && p.date <= to && p.value !== null);
  if (!slice.length) return [];
  const k = startValue / (slice[0].value as number);
  return slice.map((p) => ({ date: p.date, value: (p.value as number) * k }));
}

function drawdown(points: Point[]): Point[] {
  let peak = -Infinity;
  return points.map((p) => {
    const v = p.value ?? peak;
    peak = Math.max(peak, v);
    return { date: p.date, value: peak > 0 ? (v / peak - 1) * 100 : 0 };
  });
}

const data = (pts: Point[]) =>
  pts.filter((p) => p.value !== null).map((p) => ({ time: p.date as Time, value: p.value as number }));

export function EquityChart({ lines, height = 320 }: { lines: EquityLine[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  const theme = useChartTheme();

  useEffect(() => {
    if (!ref.current || !ddRef.current || !lines.length) return;
    const opts = (h: number) => ({
      height: h,
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: theme.muted,
        fontFamily: "'Inter Variable', Inter, system-ui, sans-serif" },
      grid: { vertLines: { visible: false }, horzLines: { color: theme.border } },
      rightPriceScale: { borderColor: theme.border },
      timeScale: { borderColor: theme.border },
    });
    const chart = createChart(ref.current, opts(height));
    lines.forEach((l, i) => {
      const s = chart.addLineSeries({
        color: l.color ?? (i === 0 ? theme.accent : theme.muted),
        lineWidth: i === 0 ? 3 : 2,
        lineStyle: l.dashed ? LineStyle.Dashed : LineStyle.Solid,
        priceLineVisible: false,
        title: l.name,
        priceFormat: { type: "price", precision: 0, minMove: 1 },
      });
      s.setData(data(l.points));
    });
    chart.timeScale().fitContent();

    const dd = createChart(ddRef.current, opts(110));
    const area = dd.addAreaSeries({
      lineColor: theme.loss,
      topColor: theme.loss.replace("rgb", "rgba").replace(")", ", 0.05)"),
      bottomColor: theme.loss.replace("rgb", "rgba").replace(")", ", 0.35)"),
      lineWidth: 1,
      priceLineVisible: false,
      title: "Drawdown %",
      invertFilledArea: true,
    });
    area.setData(data(drawdown(lines[0].points)));
    dd.timeScale().fitContent();
    chart.timeScale().subscribeVisibleLogicalRangeChange((r) => r && dd.timeScale().setVisibleLogicalRange(r));
    return () => {
      chart.remove();
      dd.remove();
    };
  }, [lines, theme, height]);

  return (
    <div>
      <div ref={ref} style={{ height }} />
      <div ref={ddRef} style={{ height: 110 }} className="mt-1 border-t border-border" />
    </div>
  );
}
