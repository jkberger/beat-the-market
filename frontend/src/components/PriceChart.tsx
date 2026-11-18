import {
  ColorType,
  createChart,
  CrosshairMode,
  type IChartApi,
  type LogicalRange,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { Bar, Marker } from "@/api/types";
import { SERIES_COLORS, useChartTheme, type ChartTheme } from "@/lib/theme";

export interface LineData {
  name: string;
  dates: string[];
  values: (number | null)[];
}

export interface Pane {
  name: string;
  series: LineData[];
}

interface Props {
  bars: Bar[];
  overlays?: LineData[];
  panes?: Pane[];
  markers?: Marker[];
  height?: number;
}

function baseOptions(theme: ChartTheme, height: number) {
  return {
    height,
    layout: {
      background: { type: ColorType.Solid, color: "transparent" },
      textColor: theme.muted,
      fontFamily: "'Inter Variable', Inter, system-ui, sans-serif",
    },
    grid: { vertLines: { color: theme.border + "00" }, horzLines: { color: theme.border } },
    rightPriceScale: { borderColor: theme.border },
    timeScale: { borderColor: theme.border },
    crosshair: { mode: CrosshairMode.Normal },
  };
}

function toLine(d: LineData) {
  const out: { time: Time; value: number }[] = [];
  d.dates.forEach((t, i) => {
    const v = d.values[i];
    if (v !== null && v !== undefined && Number.isFinite(v)) out.push({ time: t as Time, value: v });
  });
  return out;
}

export function PriceChart({ bars, overlays = [], panes = [], markers = [], height = 420 }: Props) {
  const mainRef = useRef<HTMLDivElement>(null);
  const paneRefs = useRef<(HTMLDivElement | null)[]>([]);
  const theme = useChartTheme();

  useEffect(() => {
    if (!mainRef.current || bars.length === 0) return;
    const charts: IChartApi[] = [];
    const main = createChart(mainRef.current, { ...baseOptions(theme, height), autoSize: true });
    charts.push(main);

    const candles = main.addCandlestickSeries({
      upColor: theme.gain,
      downColor: theme.loss,
      borderVisible: false,
      wickUpColor: theme.gain,
      wickDownColor: theme.loss,
    });
    candles.setData(
      bars.map((b) => ({ time: b.date as Time, open: b.open, high: b.high, low: b.low, close: b.close })),
    );
    const volume = main.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" } });
    main.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volume.setData(
      bars.map((b) => ({
        time: b.date as Time,
        value: b.volume,
        color: (b.close >= b.open ? theme.gain : theme.loss).replace("rgb", "rgba").replace(")", ", 0.25)"),
      })),
    );
    overlays.forEach((o, i) => {
      const s = main.addLineSeries({
        color: SERIES_COLORS[i % SERIES_COLORS.length],
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        title: o.name,
      });
      s.setData(toLine(o));
    });
    if (markers.length) {
      const first = bars[0].date;
      const ms: SeriesMarker<Time>[] = markers
        .filter((m) => m.date >= first)
        .map((m) => ({
          time: m.date as Time,
          position: m.side === "buy" ? "belowBar" : "aboveBar",
          color: m.side === "buy" ? theme.gain : theme.loss,
          shape: m.side === "buy" ? "arrowUp" : "arrowDown",
          text: m.side === "buy" ? "B" : "S",
        }));
      candles.setMarkers(ms);
    }

    panes.forEach((p, pi) => {
      const el = paneRefs.current[pi];
      if (!el) return;
      const chart = createChart(el, { ...baseOptions(theme, 140), autoSize: true });
      charts.push(chart);
      p.series.forEach((s, si) => {
        const line = chart.addLineSeries({
          color: SERIES_COLORS[(si + pi + overlays.length) % SERIES_COLORS.length],
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: true,
          title: s.name,
        });
        line.setData(toLine(s));
      });
    });

    // Keep every pane's time axis in sync with the main chart.
    let syncing = false;
    charts.forEach((c) =>
      c.timeScale().subscribeVisibleLogicalRangeChange((range: LogicalRange | null) => {
        if (syncing || !range) return;
        syncing = true;
        charts.forEach((o) => o !== c && o.timeScale().setVisibleLogicalRange(range));
        syncing = false;
      }),
    );
    main.timeScale().fitContent();
    return () => charts.forEach((c) => c.remove());
  }, [bars, overlays, panes, markers, theme, height]);

  return (
    <div className="flex flex-col gap-1">
      <div ref={mainRef} style={{ height }} className="w-full" />
      {panes.map((p, i) => (
        <div key={p.name} className="border-t border-border pt-1">
          <div className="px-1 text-[11px] font-medium text-muted">{p.name}</div>
          <div ref={(el) => (paneRefs.current[i] = el)} style={{ height: 140 }} className="w-full" />
        </div>
      ))}
    </div>
  );
}
