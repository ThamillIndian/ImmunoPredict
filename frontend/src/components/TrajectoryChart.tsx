import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

interface TrajectoryChartProps {
  data: {
    day: number;
    median: number;
    p5: number;
    p95: number;
  }[];
}

export function TrajectoryChart({ data }: TrajectoryChartProps) {
  // Safe default
  if (!data || data.length === 0) return <div>No trajectory data available.</div>;

  return (
    <div className="w-full h-[400px] mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
          
          <XAxis 
            dataKey="day" 
            label={{ value: "Days Post-Vaccination", position: "insideBottom", offset: -10 }} 
            tick={{ fill: 'var(--muted-foreground)' }}
          />
          <YAxis 
            label={{ value: "Antibody Titer", angle: -90, position: "insideLeft" }}
            tick={{ fill: 'var(--muted-foreground)' }}
            domain={[0, (dataMax: number) => Math.ceil(dataMax + 20)]}
          />
          
          <Tooltip 
            contentStyle={{ backgroundColor: 'var(--card)', borderRadius: '8px', border: '1px solid var(--border)' }}
            labelFormatter={(label) => `Day ${label}`}
            formatter={(value: any) => [typeof value === 'number' ? value.toFixed(1) : value, ""]}
          />
          
          <Legend verticalAlign="top" height={36} />

          {/* Protective Threshold Line */}
          <ReferenceLine 
            y={45} 
            stroke="var(--color-risk-high)" 
            strokeDasharray="4 4" 
            label={{ position: 'insideTopLeft', value: 'Protective Limit (45)', fill: 'var(--color-risk-high)', fontSize: 12 }} 
          />

          {/* 90% Confidence Interval Area (p5 to p95) */}
          <Area 
            type="monotone" 
            dataKey="p95" 
            stroke="none" 
            fill="var(--color-medical-200)" 
            fillOpacity={0.3} 
            name="95th Percentile"
            isAnimationActive={true}
          />
          <Area 
            type="monotone" 
            dataKey="p5" 
            stroke="none" 
            fill="var(--background)" // "Erase" under p5 to create the band
            fillOpacity={1} 
            name="5th Percentile (Hide)"
            legendType="none" // hide from legend
            tooltipType="none" // hide from tooltip if you prefer, but we'll show it
          />

          {/* We use a neat trick for Area band: map p5 and p95 together.
              Better Way: Recharts supports array dataKey for ranges!
              But for compatibility, drawing two areas works, or simply custom shapes.
              Actually, Recharts Area `dataKey={['p5', 'p95']}` works beautifully in later versions.
              Let's use the explicit array dataKey: */}
              
          <Area 
            type="monotone" 
            dataKey={(point) => [point.p5, point.p95]} 
            stroke="none" 
            fill="var(--color-medical-300)" 
            fillOpacity={0.4} 
            name="90% Confidence Interval"
          />

          {/* Median Trajectory Line */}
          <Line 
            type="monotone" 
            dataKey="median" 
            stroke="var(--primary)" 
            strokeWidth={3}
            dot={{ r: 4, fill: "var(--primary)", strokeWidth: 2, stroke: "var(--background)" }}
            activeDot={{ r: 6 }}
            name="Forecasted Median Titer"
            isAnimationActive={true}
          />
          
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
