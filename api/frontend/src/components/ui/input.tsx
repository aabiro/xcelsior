import { cn } from "@/lib/utils";
import type { InputHTMLAttributes } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import { useState, useRef } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {}

/**
 * Text fields default to `autoComplete="off"`: most fields here are searches,
 * resource names and identifiers, where the browser's guess offered saved names
 * and addresses. A field that should be filled in passes its token (name, email,
 * organization, address-line1, one-time-code, ...).
 */
export function Input({ className, autoComplete = "off", ...props }: InputProps) {
  return (
    <input
      autoComplete={autoComplete}
      className={cn(
        "flex h-10 w-full rounded-lg border border-border bg-navy px-3 py-2 text-sm text-text-primary placeholder:text-text-muted",
        "focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

export function Label({
  className,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      className={cn("text-sm font-medium text-text-secondary", className)}
      {...props}
    />
  );
}

export function Select({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "flex h-10 w-full rounded-lg border border-border bg-navy px-3 py-2 text-sm text-text-primary",
        "focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent",
        className,
      )}
      {...props}
    />
  );
}

export function TextArea({
  className,
  autoComplete = "off",
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      autoComplete={autoComplete}
      className={cn(
        "flex w-full rounded-lg border border-border bg-navy px-3 py-2 text-sm text-text-primary placeholder:text-text-muted",
        "focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

interface NumberInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  value?: number | string;
  onChange?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function NumberInput({
  className,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  autoComplete = "off",
  ...props
}: NumberInputProps) {
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleIncrement = () => {
    if (disabled) return;
    const currentValue = typeof value === 'number' ? value : parseFloat(value as string) || 0;
    const newValue = currentValue + step;
    if (max === undefined || newValue <= max) {
      onChange?.(newValue);
    }
  };

  const handleDecrement = () => {
    if (disabled) return;
    const currentValue = typeof value === 'number' ? value : parseFloat(value as string) || 0;
    const newValue = currentValue - step;
    if (min === undefined || newValue >= min) {
      onChange?.(newValue);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (!isNaN(val)) {
      onChange?.(val);
    } else if (e.target.value === '') {
      onChange?.(min ?? 0);
    }
  };

  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="number"
        autoComplete={autoComplete}
        value={value}
        onChange={handleChange}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        disabled={disabled}
        min={min}
        max={max}
        step={step}
        className={cn(
          "flex h-10 w-full rounded-lg border border-border bg-navy px-3 py-2 pr-8 text-sm text-text-primary placeholder:text-text-muted",
          "focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "[appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none",
          className,
        )}
        {...props}
      />
      {isFocused && !disabled && (
        <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col bg-navy/95 rounded border border-border/50 shadow-lg backdrop-blur-sm">
          <button
            type="button"
            aria-label="Increase value"
            onMouseDown={(e) => {
              e.preventDefault();
            }}
            onClick={handleIncrement}
            className="px-0.5 py-px hover:bg-ice-blue/10 rounded-t transition-colors text-text-secondary hover:text-ice-blue"
            tabIndex={-1}
          >
            <ChevronUp className="h-2.5 w-2.5" />
          </button>
          <button
            type="button"
            aria-label="Decrease value"
            onMouseDown={(e) => {
              e.preventDefault();
            }}
            onClick={handleDecrement}
            className="px-0.5 py-px hover:bg-ice-blue/10 rounded-b transition-colors text-text-secondary hover:text-ice-blue"
            tabIndex={-1}
          >
            <ChevronDown className="h-2.5 w-2.5" />
          </button>
        </div>
      )}
    </div>
  );
}
