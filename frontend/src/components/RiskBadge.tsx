import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle, Info } from "lucide-react";

interface RiskBadgeProps {
  tier: string;
  className?: string;
}

export function RiskBadge({ tier, className }: RiskBadgeProps) {
  const isHigh = tier.toUpperCase() === "HIGH";
  const isMedium = tier.toUpperCase() === "MEDIUM";
  const isLow = tier.toUpperCase() === "LOW";

  return (
    <div
      className={cn(
        "inline-flex items-center px-3 py-1.5 rounded-full text-sm font-semibold border",
        isHigh && "bg-[var(--color-risk-high-bg)] text-[var(--color-risk-high)] border-[var(--color-risk-high)]/20",
        isMedium && "bg-[var(--color-risk-medium-bg)] text-[var(--color-risk-medium)] border-[var(--color-risk-medium)]/20",
        isLow && "bg-[var(--color-risk-low-bg)] text-[var(--color-risk-low)] border-[var(--color-risk-low)]/20",
        className
      )}
    >
      {isHigh && <AlertTriangle className="w-4 h-4 mr-1.5" />}
      {isMedium && <Info className="w-4 h-4 mr-1.5" />}
      {isLow && <CheckCircle className="w-4 h-4 mr-1.5" />}
      {tier.toUpperCase()} RISK
    </div>
  );
}
