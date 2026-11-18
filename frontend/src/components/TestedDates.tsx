import type { WindowInfo } from "@/api/types";

interface Props {
  windows: WindowInfo[];
  holdout: { start: string; end: string } | null;
  holdoutStart?: string;
}

/** A one-line answer to "what was this tested on?", with the details folded away. */
export function TestedDates({ windows, holdout, holdoutStart }: Props) {
  if (!windows.length) return null;
  const first = windows[0].window;
  const last = windows[windows.length - 1].window;
  return (
    <details className="group rounded-xl border border-border bg-surface-2/40 px-4 py-3 text-sm">
      <summary className="cursor-pointer select-none text-muted marker:text-muted">
        <span className="text-text">Tested on data it never trained on.</span> How does that work?
      </summary>
      <ul className="mt-3 flex list-disc flex-col gap-2 pl-5">
        <li>
          The strategy only ever learns from the <b>3 years</b> before each stretch it is tested on. It first trained on{" "}
          {first.train_start} to {first.train_end}.
        </li>
        <li>
          It was then tested blind from {first.test_start} to {last.test_end}, in {windows.length} windows of 6 months,
          re-tuning before each one. The results above are those blind windows joined end to end, as one account.
        </li>
        <li>
          {holdout
            ? `The final check (${holdout.start} to ${holdout.end}) was locked away the whole time and scored once at the end.`
            : `Data from ${holdoutStart ?? "the last two years"} on is locked away as a final check and was not scored in this run.`}
        </li>
      </ul>
    </details>
  );
}
