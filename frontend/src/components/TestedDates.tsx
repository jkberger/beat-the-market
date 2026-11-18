import type { WindowInfo } from "@/api/types";

interface Props {
  windows: WindowInfo[];
  holdout: { start: string; end: string } | null;
  holdoutStart?: string;
}

const days = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 86_400_000;

/** A plain-English timeline of what the strategy was tuned on, tested on, and held out from. */
export function TestedDates({ windows, holdout, holdoutStart }: Props) {
  if (!windows.length) return null;
  const first = windows[0].window;
  const last = windows[windows.length - 1].window;
  const end = holdout?.end ?? last.test_end;
  const total = Math.max(1, days(first.train_start, end));
  const pct = (a: string, b: string) => `${(days(a, b) / total) * 100}%`;

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-sm">
      <div className="font-semibold">How this was tested</div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-surface-2" aria-hidden>
        <div className="bg-muted/40" style={{ width: pct(first.train_start, first.test_start) }} title="First training window" />
        <div className="bg-accent/70" style={{ width: pct(first.test_start, last.test_end) }} title="Out-of-sample test windows" />
        {holdout && <div className="bg-gain/70" style={{ width: pct(holdout.start, holdout.end) }} title="Holdout" />}
      </div>
      <div className="grid gap-2 sm:grid-cols-3">
        <div className="flex gap-2">
          <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-muted/60" />
          <span>
            <span className="font-medium">Training.</span> First trained on {first.train_start} to {first.train_end}. Every later
            window is tuned only on the 3 years right before it.
          </span>
        </div>
        <div className="flex gap-2">
          <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-accent" />
          <span>
            <span className="font-medium">Out-of-sample.</span> Tested blind from {first.test_start} to {last.test_end}, in{" "}
            {windows.length} windows of 6 months. Those blind test windows, stitched together, are the out-of-sample result.
          </span>
        </div>
        <div className="flex gap-2">
          <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full bg-gain" />
          <span>
            <span className="font-medium">Holdout.</span>{" "}
            {holdout
              ? `${holdout.start} to ${holdout.end}. Never used for tuning, scored once with the final settings.`
              : `Starts ${holdoutStart ?? "after the last test window"}; not scored in this run.`}
          </span>
        </div>
      </div>
    </div>
  );
}
