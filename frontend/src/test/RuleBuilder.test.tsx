import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import type { Condition, Group } from "@/api/types";
import { RuleBuilder } from "@/components/RuleBuilder";
import { specs } from "./fixtures";

function Harness({ onChange }: { onChange: (g: Group) => void }) {
  const [g, setG] = useState<Group>({ kind: "group", join: "and", items: [] });
  return (
    <RuleBuilder
      title="Buy when"
      value={g}
      indicators={specs}
      onChange={(next) => {
        setG(next);
        onChange(next);
      }}
    />
  );
}

function last(fn: ReturnType<typeof vi.fn>): Group {
  return fn.mock.calls[fn.mock.calls.length - 1][0];
}

test("adding a row and choosing crosses above 30", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  await userEvent.selectOptions(screen.getByLabelText("Comparison"), "crosses_above");
  const value = screen.getByLabelText("Right value");
  await userEvent.clear(value);
  await userEvent.type(value, "30");
  const cond = last(onChange).items[0] as Condition;
  expect(cond).toMatchObject({
    kind: "condition",
    op: "crosses_above",
    right: { kind: "constant", value: 30 },
  });
});

test("choosing an indicator on the left uses its default params", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  await userEvent.selectOptions(screen.getByLabelText("Left operand"), "ind:rsi");
  const cond = last(onChange).items[0] as Condition;
  expect(cond.left).toEqual({ kind: "indicator", id: "rsi", params: { length: 14 }, output: "rsi" });
});

test("turning on the range switch emits a range", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  const value = screen.getByLabelText("Right value");
  await userEvent.clear(value);
  await userEvent.type(value, "30");
  await userEvent.click(screen.getByRole("switch", { name: "Right value range" }));
  const cond = last(onChange).items[0] as Condition;
  expect(cond.right).toEqual({ kind: "constant", value: { min: 30, max: 40, step: 5 } });
});

test("rising hides the right operand and shows bars", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  await userEvent.selectOptions(screen.getByLabelText("Comparison"), "rising");
  expect(screen.queryByLabelText("Right operand")).not.toBeInTheDocument();
  expect(screen.getByLabelText("Bars")).toBeInTheDocument();
});

test("join toggle switches to or", async () => {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  await userEvent.click(screen.getByRole("button", { name: /add condition/i }));
  await userEvent.click(screen.getByRole("radio", { name: "Any (OR)" }));
  expect(last(onChange).join).toBe("or");
  expect(last(onChange).items).toHaveLength(2);
});
