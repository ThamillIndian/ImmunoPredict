"use client";

import { useState } from "react";
import { PatientForm } from "@/components/PatientForm";
import { TrajectoryChart } from "@/components/TrajectoryChart";
import { RiskBadge } from "@/components/RiskBadge";
import { BiologicalRates } from "@/components/BiologicalRates";
import { submitPrediction, PatientPayload, PredictionResult } from "@/lib/api";
import { Activity, Loader2, ArrowRight } from "lucide-react";

export default function Dashboard() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (data: PatientPayload) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await submitPrediction(data);
      setResult(res);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to connect to the Hybrid AI Backend.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      {/* LEFT COLUMN: Input Form */}
      <div className="lg:col-span-4 space-y-6">
        <div className="glass-card rounded-xl p-6">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-foreground tracking-tight">Clinical Assessment</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Enter early biomarker data (Day 0-7) to forecast 90-day trajectory.
            </p>
          </div>
          <PatientForm onSubmit={handlePredict} isLoading={isLoading} />
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg text-sm font-medium border border-red-200">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* RIGHT COLUMN: Results Dashboard */}
      <div className="lg:col-span-8">
        {!result && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground glass-card rounded-xl border-dashed border-2 bg-muted/50 p-12">
            <Activity className="w-16 h-16 mb-4 text-muted-foreground/50" />
            <h3 className="text-xl font-semibold mb-2 text-foreground">Awaiting Patient Data</h3>
            <p className="text-center max-w-sm text-sm">
              The Hybrid ODE+AI Engine is standing by. Submit a clinical profile to generate a predictive risk report.
            </p>
          </div>
        )}

        {isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground glass-card rounded-xl p-12">
            <Loader2 className="w-12 h-12 mb-4 animate-spin text-[var(--primary)]" />
            <span className="font-semibold text-lg text-foreground">Simulating 90-day Trajectory...</span>
            <span className="text-sm mt-2">Solving Ordinary Differential Equations</span>
          </div>
        )}

        {result && !isLoading && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Top Analysis Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-card rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-5 bg-gradient-to-bl from-teal-500 rounded-bl-full w-32 h-32" />
                <h3 className="text-lg font-bold mb-4 opacity-80">Final Risk Decision</h3>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-3xl font-black text-foreground mb-1">
                      {result.risk_assessment.predicted_titer_28.toFixed(1)} <span className="text-sm font-medium text-muted-foreground">IU/mL</span>
                    </div>
                    <div className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">Day 28 Peak Titer</div>
                  </div>
                  <RiskBadge tier={result.risk_assessment.tier} className="scale-110 origin-top-right" />
                </div>
                <div className="mt-6 p-3 bg-muted rounded-lg text-sm text-foreground border border-border flex items-start gap-3">
                  <ArrowRight className="w-4 h-4 mt-0.5 text-primary shrink-0" />
                  <p><strong>Recommendation:</strong> {result.risk_assessment.recommendation}</p>
                </div>
              </div>

              <div className="glass-card rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4 opacity-80">Biological ODE Parameters (<span className="font-serif italic">θ</span>)</h3>
                <BiologicalRates theta={result.predicted_theta} />
              </div>
            </div>

            {/* Bottom Trajectory Chart */}
            <div className="glass-card rounded-xl p-6">
               <div className="flex justify-between items-center mb-2">
                 <h3 className="text-xl font-bold">Vaccine Efficacy Forecast</h3>
                 <span className="text-xs bg-indigo-50 text-indigo-700 px-2 py-1 rounded font-semibold border border-indigo-100">
                   Monte Carlo (100x)
                 </span>
               </div>
               <p className="text-sm text-muted-foreground mb-4">
                 Projected antibody levels indicating the 90% confidence envelope based on early biomarker vectors.
               </p>
               <TrajectoryChart data={result.full_trajectory} />
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
