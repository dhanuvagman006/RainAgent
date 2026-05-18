import React from 'react';
import { Trophy } from 'lucide-react';

export default function LeaderboardTable({ metricsData }) {
  if (!metricsData) return null;

  const models = Object.keys(metricsData).map(modelName => {
    const fm = metricsData[modelName].final_metrics;
    return { name: modelName, nse: fm.nse, rmse: fm.rmse, mae: fm.mae, loss: fm.loss };
  });

  const sortedModels = [...models].sort((a, b) => b.nse - a.nse);
  const bestModelName = sortedModels.length > 0 ? sortedModels[0].name : null;

  return (
    <div
      id="leaderboard-table"
      className="mt-8 mb-10"
    >
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="p-1.5 bg-amber-50 border border-amber-100 rounded-lg">
          <Trophy className="w-4 h-4 text-amber-500" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-slate-800">Global Cross-Model Leaderboard</h3>
          <p className="text-xs text-slate-500 mt-0.5">Ranked by Nash–Sutcliffe Efficiency (NSE ↑ higher is better)</p>
        </div>
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-100 shadow-sm">
        <table className="w-full text-left border-collapse text-sm">

          {/* Header */}
          <thead>
            <tr className="bg-slate-100">
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200">
                Rank
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200">
                Model Architecture
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200 text-center">
                NSE 🔼
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200 text-center">
                RMSE 🔽
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200 text-center">
                MAE 🔽
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200 text-center">
                Loss (Huber) 🔽
              </th>
              <th className="px-5 py-3.5 text-xs font-semibold text-slate-700 uppercase tracking-wider border-b border-slate-200 text-center">
                Status
              </th>
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {sortedModels.map((model, idx) => {
              const isBest = model.name === bestModelName;
              return (
                <tr
                  key={model.name}
                  className={[
                    'border-b border-slate-100 transition-colors',
                    isBest
                      ? 'bg-amber-50/60 border-l-4 border-l-amber-500'
                      : idx % 2 === 0
                        ? 'bg-white hover:bg-slate-50/70'
                        : 'bg-slate-50/50 hover:bg-slate-50/80',
                  ].join(' ')}
                >
                  {/* Rank */}
                  <td className="px-5 py-4 text-slate-500 font-mono text-xs">
                    {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx + 1}`}
                  </td>

                  {/* Model Name */}
                  <td className="px-5 py-4 font-semibold text-slate-800">
                    {model.name}
                  </td>

                  {/* NSE */}
                  <td className={`px-5 py-4 text-center font-mono ${isBest ? 'text-emerald-700 font-bold' : 'text-slate-700'}`}>
                    {model.nse?.toFixed(4)}
                  </td>

                  {/* RMSE */}
                  <td className="px-5 py-4 text-center font-mono text-slate-600">
                    {model.rmse?.toFixed(4)} mm
                  </td>

                  {/* MAE */}
                  <td className="px-5 py-4 text-center font-mono text-slate-600">
                    {model.mae?.toFixed(4)} mm
                  </td>

                  {/* Loss */}
                  <td className="px-5 py-4 text-center font-mono text-slate-600">
                    {model.loss?.toFixed(4)}
                  </td>

                  {/* Status Badge */}
                  <td className="px-5 py-4 text-center">
                    {isBest ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold
                                       bg-amber-100 text-amber-700 border border-amber-200">
                        <Trophy className="w-3 h-3" /> Best Model
                      </span>
                    ) : (
                      <span className="text-slate-300 text-lg leading-none">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
