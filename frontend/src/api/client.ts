import type {
  BacktestRequest,
  BacktestResponse,
  Bar,
  IndicatorSpec,
  RiskSettings,
  RunListItem,
  SavedStrategy,
  Strategy,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  universe: () => request<{ symbols: string[]; quick_picks: string[] }>("/api/universe"),
  bars: (symbol: string, start?: string) =>
    request<{ symbol: string; bars: Bar[] }>(
      `/api/bars/${encodeURIComponent(symbol)}${start ? `?start=${start}` : ""}`,
    ),
  indicators: () => request<IndicatorSpec[]>("/api/indicators"),
  computeIndicator: (symbol: string, id: string, params: Record<string, number>, start?: string) =>
    post<{ dates: string[]; series: Record<string, (number | null)[]> }>("/api/indicators/compute", {
      symbol,
      id,
      params,
      start,
    }),
  riskPresets: () => request<Record<string, RiskSettings>>("/api/risk-presets"),
  describe: (strategy: Strategy) => post<{ text: string }>("/api/strategies/describe", strategy),
  listStrategies: () => request<SavedStrategy[]>("/api/strategies"),
  saveStrategy: (strategy: Strategy) => post<{ id: number }>("/api/strategies", strategy),
  getStrategy: (id: number) => request<{ id: number; strategy: Strategy }>(`/api/strategies/${id}`),
  deleteStrategy: (id: number) => request<void>(`/api/strategies/${id}`, { method: "DELETE" }),
  runBacktest: (req: BacktestRequest) => post<BacktestResponse>("/api/backtests", req),
  listBacktests: () => request<RunListItem[]>("/api/backtests"),
  getBacktest: (id: number) => request<BacktestResponse>(`/api/backtests/${id}`),
  getSettings: () => request<{ holdout_start: string }>("/api/settings"),
  putSettings: (holdout_start: string) =>
    request<{ holdout_start: string }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ holdout_start }),
    }),
};
