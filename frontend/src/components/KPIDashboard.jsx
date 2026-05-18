import React from 'react';
import { Droplets, Activity, Database } from 'lucide-react';

export default function KPIDashboard({ predictionData, finalTankStatus }) {
  if (!predictionData || !finalTankStatus) return null;

  const totalRainfall = predictionData.predicted_rainfall_mm.reduce((acc, curr) => acc + curr, 0);
  const tankPercent = finalTankStatus.tank_status_percentage || 0;

  // Bar color: amber when critically low (<15%), otherwise liquid-blue gradient
  const isLow = tankPercent < 15;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">

      {/* KPI 1 — Rainfall */}
      <div
        id="kpi-rainfall"
        className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6 flex items-center gap-5"
      >
        <div className="p-3.5 bg-blue-50 rounded-xl flex-shrink-0">
          <Droplets className="w-8 h-8 text-blue-600" />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-0.5">
            Total Forecasted Rainfall
          </p>
          <p className="text-3xl font-bold text-blue-700 leading-tight">
            {totalRainfall.toFixed(1)}
            <span className="text-base font-medium text-slate-400 ml-1">mm</span>
          </p>
        </div>
      </div>

      {/* KPI 2 — Tank Volume */}
      <div
        id="kpi-tank-volume"
        className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6 flex items-center gap-5"
      >
        <div className="p-3.5 bg-emerald-50 rounded-xl flex-shrink-0">
          <Activity className="w-8 h-8 text-emerald-600" />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-0.5">
            Remaining Tank Volume
          </p>
          <p className="text-3xl font-bold text-slate-800 leading-tight">
            {finalTankStatus.tank_status_liters.toFixed(0)}
            <span className="text-base font-medium text-slate-400 ml-1">L</span>
          </p>
          <p className="text-xs text-slate-400 mt-0.5">End-of-period reserve</p>
        </div>
      </div>

      {/* KPI 3 — Water Tank Operational Status Meter */}
      <div
        id="kpi-tank-meter"
        className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6"
      >
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-slate-50 rounded-lg">
              <Database className="w-4 h-4 text-slate-500" />
            </div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Tank Operational Status
            </p>
          </div>
          <span
            className={`text-2xl font-bold ${
              isLow ? 'text-amber-500' : 'text-slate-800'
            }`}
          >
            {tankPercent.toFixed(1)}%
          </span>
        </div>

        {/* Progress Track */}
        <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-1000 ease-out ${
              isLow
                ? 'bg-amber-500'
                : 'bg-gradient-to-r from-blue-400 to-blue-600'
            }`}
            style={{ width: `${Math.min(tankPercent, 100)}%` }}
          />
        </div>

        <div className="flex justify-between items-center mt-2">
          <span className="text-[11px] text-slate-400">Empty</span>
          {isLow && (
            <span className="text-[11px] font-semibold text-amber-500">⚠ Low Capacity</span>
          )}
          <span className="text-[11px] text-slate-400">Full</span>
        </div>
      </div>

    </div>
  );
}
