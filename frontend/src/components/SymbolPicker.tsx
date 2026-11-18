import { X } from "lucide-react";
import { useState, type KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

interface Props {
  value: string[];
  onChange: (symbols: string[]) => void;
  suggestions: string[];
  max?: number;
}

export function SymbolPicker({ value, onChange, suggestions, max = 10 }: Props) {
  const [text, setText] = useState("");
  const add = (raw: string) => {
    const s = raw.trim().toUpperCase();
    if (s && !value.includes(s) && value.length < max) onChange([...value, s]);
    setText("");
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === "," || e.key === " ") {
      e.preventDefault();
      add(text);
    } else if (e.key === "Backspace" && !text && value.length) onChange(value.slice(0, -1));
  };
  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-h-9 flex-wrap items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-2 py-1.5">
        {value.map((s) => (
          <span key={s} className="inline-flex items-center gap-1 rounded-md bg-accent/15 py-0.5 pl-2 pr-1 text-xs font-semibold text-accent">
            {s}
            <button type="button" aria-label={`Remove ${s}`} onClick={() => onChange(value.filter((x) => x !== s))}
              className="rounded p-0.5 hover:bg-accent/20">
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        <input
          aria-label="Add symbol"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onBlur={() => text && add(text)}
          placeholder={value.length ? "" : "Type a ticker and press Enter"}
          className="min-w-[8rem] flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
        />
      </div>
      <div className="flex flex-wrap gap-1.5">
        {suggestions.map((s) => (
          <button key={s} type="button" onClick={() => (value.includes(s) ? onChange(value.filter((x) => x !== s)) : add(s))}
            className={cn("rounded-md border px-2 py-0.5 text-xs transition-colors",
              value.includes(s) ? "border-accent/50 text-accent" : "border-border text-muted hover:text-text")}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
