import { motion } from "framer-motion";

interface BiologicalRatesProps {
  theta: {
    activation: number;
    prod: number;
    decay: number;
  };
}

export function BiologicalRates({ theta }: BiologicalRatesProps) {
  // Normalize visually for the UI (heuristics for display purposes)
  const actPercent = Math.min((theta.activation / 3) * 100, 100);
  const prodPercent = Math.min((theta.prod / 5) * 100, 100);
  const decayPercent = Math.min((theta.decay / 0.1) * 100, 100);

  const bars = [
    { label: "Immune Activation", value: theta.activation, percent: actPercent, color: "bg-blue-500", desc: "System responsiveness" },
    { label: "Antibody Production", value: theta.prod, percent: prodPercent, color: "bg-[var(--primary)]", desc: "Factory synthesis rate" },
    { label: "Antibody Decay", value: theta.decay, percent: decayPercent, color: "bg-purple-500", desc: "Clearance over time (Lower is better)" },
  ];

  return (
    <div className="space-y-4">
      {bars.map((bar) => (
        <div key={bar.label} className="w-full">
          <div className="flex justify-between items-end mb-1">
            <span className="text-sm font-medium text-[var(--foreground)]">{bar.label}</span>
            <span className="text-xs font-bold text-[var(--muted-foreground)]">{bar.value.toFixed(3)}</span>
          </div>
          <p className="text-[10px] text-[var(--muted-foreground)] mb-2">{bar.desc}</p>
          <div className="h-2 w-full bg-[var(--muted)] rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${bar.percent}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className={`h-full rounded-full ${bar.color}`}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
