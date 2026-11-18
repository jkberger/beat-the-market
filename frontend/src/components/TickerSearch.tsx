import { Search } from "lucide-react";
import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";

export function TickerSearch({ className, autoFocus }: { className?: string; autoFocus?: boolean }) {
  const [value, setValue] = useState("");
  const navigate = useNavigate();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    const symbol = value.trim().toUpperCase();
    if (symbol) {
      navigate(`/stock/${encodeURIComponent(symbol)}`);
      setValue("");
    }
  };
  return (
    <form onSubmit={submit} className={cn("relative", className)}>
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Search a ticker, e.g. AAPL"
        aria-label="Search ticker"
        autoFocus={autoFocus}
        className="h-9 w-full rounded-lg border border-border bg-surface-2 pl-9 pr-3 text-sm placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50"
      />
    </form>
  );
}
