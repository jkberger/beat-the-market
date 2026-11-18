import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { IndicatorPicker } from "@/components/IndicatorPicker";
import type { IndicatorSpec } from "@/api/types";

const specs: IndicatorSpec[] = [
  { id: "rsi", name: "Relative Strength Index", category: "momentum", outputs: ["rsi"], overlay: false,
    params: [{ name: "length", kind: "int", default: 14, min: 2, max: 200 }] },
  { id: "sma", name: "Simple Moving Average", category: "trend", outputs: ["sma"], overlay: true,
    params: [{ name: "length", kind: "int", default: 20, min: 2, max: 400 }] },
  { id: "macd", name: "MACD", category: "trend", outputs: ["macd", "signal", "hist"], overlay: false,
    params: [] },
];

test("groups by category and filters by search", async () => {
  render(<IndicatorPicker indicators={specs} onAdd={() => {}} />);
  expect(screen.getByText("Trend")).toBeInTheDocument();
  expect(screen.getByText("Momentum")).toBeInTheDocument();
  await userEvent.type(screen.getByPlaceholderText(/search indicators/i), "rsi");
  expect(screen.getByRole("button", { name: /RSI/ })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /SMA/ })).not.toBeInTheDocument();
});

test("clicking calls onAdd with default params", async () => {
  const onAdd = vi.fn();
  render(<IndicatorPicker indicators={specs} onAdd={onAdd} />);
  await userEvent.click(screen.getByRole("button", { name: /RSI/ }));
  expect(onAdd).toHaveBeenCalledWith("rsi", { length: 14 });
});
