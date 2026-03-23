"use client";

import { useState } from "react";
import { PatientPayload } from "@/lib/api";

interface PatientFormProps {
  onSubmit: (data: PatientPayload) => void;
  isLoading: boolean;
}

export function PatientForm({ onSubmit, isLoading }: PatientFormProps) {
  const [formData, setFormData] = useState<PatientPayload>({
    patient_id: `PT-${Math.floor(Math.random() * 10000)}`,
    vaccine_type: "A",
    age: 45,
    sex: 0,
    bmi: 24.5,
    comorbidity_score: 0,
    measurements: [
      { day: 0, wbc: 7.2, cytokine_il6: 2.1 },
    ],
  });

  const fillExample = (tier: 'LOW' | 'MEDIUM' | 'HIGH') => {
    if (tier === 'LOW') { // Changed to LOW (was HIGH)
      setFormData({
        patient_id: "PT-LOW-RISK",
        vaccine_type: "A",
        age: 46,
        sex: 0,
        bmi: 27.54,
        comorbidity_score: 0,
        measurements: [
          { day: 0, wbc: 6.73, cytokine_il6: 1.93, cytokine_tnfa: 1.51, cytokine_ifng: 0.83, lymphocytes: 2.08, neutrophils: 3.88 },
          { day: 1, wbc: 37.90, cytokine_il6: 201.37, cytokine_tnfa: 113.23, cytokine_ifng: 67.49, lymphocytes: 13.78, neutrophils: 4.0 },
          { day: 3, wbc: 47.11, cytokine_il6: 120.29, cytokine_tnfa: 67.11, cytokine_ifng: 75.11, lymphocytes: 40.07, neutrophils: 23.17 },
          { day: 7, wbc: 21.31, cytokine_il6: 15.13, cytokine_tnfa: 6.28, cytokine_ifng: 45.16, lymphocytes: 22.94, neutrophils: 6.39 },
        ]
      });
    } else if (tier === 'MEDIUM') {
      setFormData({
        patient_id: "PT-MED-RISK",
        vaccine_type: "A",
        age: 54,
        sex: 0,
        bmi: 24.79,
        comorbidity_score: 2,
        measurements: [
          { day: 0, wbc: 7.42, cytokine_il6: 1.83, cytokine_tnfa: 1.61, cytokine_ifng: 0.70, lymphocytes: 2.06, neutrophils: 4.44 },
          { day: 1, wbc: 35.20, cytokine_il6: 235.21, cytokine_tnfa: 96.75, cytokine_ifng: 61.80, lymphocytes: 8.41, neutrophils: 46.63 },
          { day: 3, wbc: 50.43, cytokine_il6: 213.86, cytokine_tnfa: 125.20, cytokine_ifng: 74.30, lymphocytes: 32.26, neutrophils: 48.62 },
          { day: 7, wbc: 33.08, cytokine_il6: 76.58, cytokine_tnfa: 36.49, cytokine_ifng: 63.69, lymphocytes: 27.49, neutrophils: 16.74 },
        ]
      });
    } else { // Changed to HIGH (was LOW)
      setFormData({
        patient_id: "PT-HIGH-RISK",
        vaccine_type: "A",
        age: 27,
        sex: 0,
        bmi: 27.63,
        comorbidity_score: 3,
        measurements: [
          { day: 0, wbc: 6.71, cytokine_il6: 1.94, cytokine_tnfa: 1.27, cytokine_ifng: 0.76, lymphocytes: 1.88, neutrophils: 4.06 },
          { day: 1, wbc: 42.40, cytokine_il6: 158.24, cytokine_tnfa: 108.37, cytokine_ifng: 62.45, lymphocytes: 12.45, neutrophils: 41.31 },
          { day: 3, wbc: 44.61, cytokine_il6: 190.82, cytokine_tnfa: 83.34, cytokine_ifng: 100.00, lymphocytes: 37.88, neutrophils: 37.09 },
          { day: 7, wbc: 27.97, cytokine_il6: 28.53, cytokine_tnfa: 18.07, cytokine_ifng: 40.16, lymphocytes: 2.0, neutrophils: 9.45 },
        ]
      });
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Demographics Group */}
      <div className="space-y-4">
        <div className="flex flex-col gap-2 border-b pb-2">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Patient Demographics</h3>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={() => fillExample('LOW')} className="text-[10px] bg-green-500/10 text-green-600 hover:bg-green-500/20 px-2 py-0.5 rounded border border-green-500/20 font-medium transition-colors">Low Risk (High Response)</button>
            <button type="button" onClick={() => fillExample('MEDIUM')} className="text-[10px] bg-amber-500/10 text-amber-600 hover:bg-amber-500/20 px-2 py-0.5 rounded border border-amber-500/20 font-medium transition-colors">Mid Risk (Borderline)</button>
            <button type="button" onClick={() => fillExample('HIGH')} className="text-[10px] bg-red-500/10 text-red-600 hover:bg-red-500/20 px-2 py-0.5 rounded border border-red-500/20 font-medium transition-colors">High Risk (Low Response)</button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">Patient ID</label>
              <input 
              type="text" 
              className="w-full rounded-md border border-border p-2 bg-background text-foreground"
              value={formData.patient_id}
              onChange={e => setFormData({...formData, patient_id: e.target.value})}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Vaccine Protocol</label>
            <select 
              className="w-full rounded-md border border-border p-2 bg-background text-foreground"
              value={formData.vaccine_type}
              onChange={e => setFormData({...formData, vaccine_type: e.target.value as "A"|"B"})}
            >
              <option value="A">Protocol A (mRNA)</option>
              <option value="B">Protocol B (Subunit)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Age</label>
            <input type="number" min="18" max="100" className="w-full rounded-md border border-border p-2 bg-background text-foreground" value={formData.age} onChange={e => setFormData({...formData, age: Number(e.target.value)})} />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">BMI</label>
            <input type="number" step="0.1" className="w-full rounded-md border border-border p-2 bg-background text-foreground" value={formData.bmi} onChange={e => setFormData({...formData, bmi: Number(e.target.value)})} />
          </div>
        </div>
      </div>

      {/* Biomarker Group */}
      <div className="space-y-4">
        <div className="flex justify-between items-center border-b pb-2">
          <h3 className="text-lg font-semibold">Early Biomarkers</h3>
          <div className="flex items-center gap-2">
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">Day 0-7 Required</span>
            {[0, 1, 3, 7].some(d => !formData.measurements.find(m => m.day === d)) && (
              <button
                type="button"
                className="text-xs bg-primary/10 text-primary hover:bg-primary/20 px-2 py-1 rounded-full transition-colors flex items-center font-medium"
                onClick={() => {
                  const allowedDays = [0, 1, 3, 7];
                  const currentDays = formData.measurements.map(m => m.day);
                  // Find the first day in allowedDays that isn't in currentDays
                  const nextDay = allowedDays.find(d => !currentDays.includes(d));
                  
                  if (nextDay !== undefined) {
                    setFormData({
                      ...formData,
                      measurements: [...formData.measurements, { day: nextDay }].sort((a, b) => a.day - b.day)
                    });
                  }
                }}
              >
                + Add Day { [0, 1, 3, 7].find(d => !formData.measurements.map(m => m.day).includes(d)) }
              </button>
            )}
          </div>
        </div>
        
        {formData.measurements.map((m, idx) => (
          <div key={idx} className="p-4 border border-border rounded-lg bg-card flex flex-col space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <label className="block text-xs font-semibold text-muted-foreground uppercase mb-1">Timing</label>
                <div className="font-medium text-lg text-foreground">Day {m.day}</div>
              </div>
              {idx > 1 && (
                <button
                  type="button"
                  className="text-xs text-red-500 hover:text-red-700 font-medium"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      measurements: formData.measurements.filter((_, i) => i !== idx)
                    });
                  }}
                >
                  Remove
                </button>
              )}
            </div>
            
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
              {[
                { label: "WBC Count", key: "wbc" },
                { label: "IL-6 Level", key: "cytokine_il6" },
                { label: "TNF-α Level", key: "cytokine_tnfa" },
                { label: "IFN-γ Level", key: "cytokine_ifng" },
                { label: "Lymphocytes", key: "lymphocytes" },
                { label: "Neutrophils", key: "neutrophils" },
              ].map(({ label, key }) => (
                <div key={key}>
                  <label className="block text-xs font-medium mb-1 text-foreground">{label}</label>
                  <input 
                    type="number" 
                    step="0.1" 
                    className="w-full rounded border border-border bg-background text-foreground p-1.5 focus:ring-1 focus:ring-primary outline-none" 
                    value={(m as any)[key] ?? ''} 
                    placeholder="Optional"
                    onChange={e => {
                      const val = e.target.value === '' ? undefined : Number(e.target.value);
                      const newM = [...formData.measurements];
                      (newM[idx] as any)[key] = val;
                      setFormData({...formData, measurements: newM});
                    }} 
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <button 
        type="submit" 
        disabled={isLoading}
        className="w-full py-3 px-4 bg-primary text-primary-foreground rounded-lg font-semibold shadow-md hover:bg-primary/90 transition-colors disabled:opacity-50"
      >
        {isLoading ? "Running Hybrid AI Simulation..." : "Assess Clinical Risk"}
      </button>
    </form>
  );
}
