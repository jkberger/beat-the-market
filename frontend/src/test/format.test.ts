import { formatMoney, formatNumber, formatPct } from "@/lib/format";

test("formatPct", () => {
  expect(formatPct(0.1234)).toBe("12.3%");
  expect(formatPct(null)).toBe("—");
  expect(formatPct(-0.05)).toBe("-5.0%");
  expect(formatPct(undefined)).toBe("—");
});

test("formatMoney", () => {
  expect(formatMoney(12345.678)).toBe("$12,345.68");
  expect(formatMoney(null)).toBe("—");
});

test("formatNumber", () => {
  expect(formatNumber(1.23456)).toBe("1.23");
  expect(formatNumber(null)).toBe("—");
});
