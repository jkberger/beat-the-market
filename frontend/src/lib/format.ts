const DASH = "—";

export function formatPct(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return `${(x * 100).toFixed(digits)}%`;
}

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function formatMoney(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return money.format(x);
}

export function formatNumber(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return DASH;
  return x.toFixed(digits);
}

export function signClass(x: number | null | undefined): string {
  if (x === null || x === undefined || x === 0) return "text-muted";
  return x > 0 ? "text-gain" : "text-loss";
}
