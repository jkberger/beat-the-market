import { forwardRef, type InputHTMLAttributes, type SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const field =
  "h-9 rounded-lg border border-border bg-surface-2 px-3 text-sm text-text placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/50";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => <input ref={ref} className={cn(field, "w-full", className)} {...props} />,
);
Input.displayName = "Input";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => <select ref={ref} className={cn(field, "pr-8", className)} {...props} />,
);
Select.displayName = "Select";
