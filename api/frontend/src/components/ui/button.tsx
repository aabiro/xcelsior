import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ice-blue disabled:pointer-events-none disabled:opacity-50 cursor-pointer",
  {
    variants: {
      variant: {
        default: "btn-gradient-primary text-white font-semibold",
        destructive: "bg-accent-red text-white hover:bg-accent-red-hover",
        secondary: "bg-navy-lighter text-text-primary hover:bg-border-light",
        outline: "border border-border text-text-primary hover:bg-navy-light",
        ghost: "text-text-secondary hover:text-text-primary hover:bg-navy-light",
        gold: "bg-accent-gold text-navy hover:bg-accent-gold-hover font-semibold",
        success: "bg-emerald text-white font-semibold hover:bg-emerald/80 [text-shadow:0_1px_2px_rgba(0,0,0,0.28)] ring-1 ring-black/10",
      },
      size: {
        sm: "h-8 px-3 text-sm",
        md: "h-10 px-4 text-sm",
        lg: "h-12 px-6 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: { variant: "default", size: "md" },
  },
);

interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return (
    <button className={cn(buttonVariants({ variant, size }), className)} {...props} />
  );
}

export { buttonVariants };
