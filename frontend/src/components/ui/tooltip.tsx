import type { ReactNode } from "react";

export function Tooltip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-56 -translate-x-1/2 rounded-lg border border-border bg-surface px-3 py-2 text-xs font-normal leading-snug text-text opacity-0 shadow-lg transition-opacity group-hover:opacity-100"
      >
        {text}
      </span>
    </span>
  );
}
