import type { IndicatorSpec, Operand, PriceField } from "@/api/types";
import { Select } from "@/components/ui/input";
import { defaultParams } from "./IndicatorPicker";
import { ParamInput } from "./ParamInput";

const PRICE: PriceField[] = ["close", "open", "high", "low"];

function encode(o: Operand): string {
  if (o.kind === "price") return `price:${o.field}`;
  if (o.kind === "indicator") return `ind:${o.id}`;
  return "const";
}

interface Props {
  value: Operand;
  onChange: (o: Operand) => void;
  indicators: IndicatorSpec[];
  side: "Left" | "Right";
  allowConstant?: boolean;
}

export function OperandEditor({ value, onChange, indicators, side, allowConstant = true }: Props) {
  const pick = (code: string) => {
    if (code.startsWith("price:")) onChange({ kind: "price", field: code.slice(6) as PriceField });
    else if (code.startsWith("ind:")) {
      const spec = indicators.find((i) => i.id === code.slice(4))!;
      onChange({ kind: "indicator", id: spec.id, params: defaultParams(spec), output: spec.outputs[0] });
    } else onChange({ kind: "constant", value: 0 });
  };
  const spec = value.kind === "indicator" ? indicators.find((i) => i.id === value.id) : undefined;

  return (
    <span className="inline-flex min-w-0 max-w-full flex-wrap items-center gap-1.5">
      <Select aria-label={`${side} operand`} value={encode(value)} onChange={(e) => pick(e.target.value)} className="h-8 w-full min-w-0 max-w-full text-sm sm:w-auto">
        <optgroup label="Price">
          {PRICE.map((f) => (
            <option key={f} value={`price:${f}`}>
              {f[0].toUpperCase() + f.slice(1)}
            </option>
          ))}
        </optgroup>
        <optgroup label="Indicators">
          {indicators.map((i) => (
            <option key={i.id} value={`ind:${i.id}`}>
              {i.id.toUpperCase()} · {i.name}
            </option>
          ))}
        </optgroup>
        {allowConstant && <option value="const">Number</option>}
      </Select>
      {value.kind === "constant" && (
        <ParamInput label={`${side} value`} value={value.value} onChange={(v) => onChange({ ...value, value: v })} />
      )}
      {value.kind === "indicator" && spec && (
        <>
          {spec.params.map((p) => (
            <span key={p.name} className="inline-flex items-center gap-1 text-xs text-muted">
              {p.name}
              <ParamInput
                label={`${side} ${spec.id} ${p.name}`}
                value={value.params[p.name] ?? p.default}
                onChange={(v) => onChange({ ...value, params: { ...value.params, [p.name]: v } })}
              />
            </span>
          ))}
          {spec.outputs.length > 1 && (
            <Select
              aria-label={`${side} output`}
              value={value.output}
              onChange={(e) => onChange({ ...value, output: e.target.value })}
              className="h-8 text-sm"
            >
              {spec.outputs.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </Select>
          )}
        </>
      )}
    </span>
  );
}
