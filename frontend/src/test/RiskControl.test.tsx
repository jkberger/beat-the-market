import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import type { RiskSettings } from "@/api/types";
import { RISK_PRESETS, RiskControl } from "@/components/RiskControl";

function Harness({ onChange }: { onChange: (r: RiskSettings) => void }) {
  const [r, setR] = useState<RiskSettings>(RISK_PRESETS.balanced);
  return (
    <RiskControl
      value={r}
      onChange={(next) => {
        setR(next);
        onChange(next);
      }}
    />
  );
}

test("clicking Conservative applies its values", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("radio", { name: "Conservative" }));
  expect(onChange).toHaveBeenLastCalledWith(
    expect.objectContaining({ preset: "conservative", position_pct: 25, objective: "sortino" }),
  );
});

test("editing a field switches to custom", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("radio", { name: "Custom" }));
  const size = screen.getByLabelText("Position size (% of equity)");
  await userEvent.clear(size);
  await userEvent.type(size, "40");
  expect(onChange).toHaveBeenLastCalledWith(
    expect.objectContaining({ preset: "custom", position_pct: 40 }),
  );
});

test("presets mirror the spec table", () => {
  expect(RISK_PRESETS.aggressive).toMatchObject({ position_pct: 100, stop_atr_mult: null,
    max_drawdown_pct: null, objective: "cagr" });
  expect(RISK_PRESETS.balanced).toMatchObject({ position_pct: 50, stop_atr_mult: 3,
    max_drawdown_pct: 20, objective: "sharpe" });
});
