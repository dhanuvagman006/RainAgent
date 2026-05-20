import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import { Info, TrendingDown } from 'lucide-react';

// Shared chart styling constants (light theme)
const CHART_GRID_COLOR = '#f1f5f9';    // slate-100
const CHART_AXIS_COLOR = '#475569';    // slate-600
const TOOLTIP_STYLE = {
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0',
  borderRadius: '8px',
  color: '#1e293b',
  boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
  fontSize: '12px'
};
const CURSOR_FILL = '#f1f5f9';

export default function AnalyticsView({ metricsData, selectedModel, weatherSummary }) {
  if (!metricsData) return (
    <div className="h-64 flex items-center justify-center bg-white border border-slate-100 rounded-xl shadow-sm mb-6">
      <p className="text-slate-400 text-sm">Loading metrics data…</p>
    </div>
  );

  const activeModel = metricsData[selectedModel] ? selectedModel : "LSTM";
  const history = metricsData[activeModel]?.history || {};

  const epochData = [];
  if (history.loss && history.val_loss) {
    for (let i = 0; i < history.loss.length; i++) {
      epochData.push({ epoch: i + 1, loss: history.loss[i], val_loss: history.val_loss[i] });
    }
  }

  const comparisonData = Object.keys(metricsData).map(modelName => ({
    name: modelName,
    nse:  metricsData[modelName].final_metrics.nse,
    rmse: metricsData[modelName].final_metrics.rmse,
    mae:  metricsData[modelName].final_metrics.mae,
  }));

  const minRmseModel = comparisonData.reduce(
    (prev, cur) => (prev.rmse < cur.rmse ? prev : cur),
    { rmse: 999 }
  );

  const season = weatherSummary?.t2m_max > 32
    ? 'Summer' : weatherSummary?.rh2m > 75
    ? 'Monsoon' : 'Winter';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-6">

      {/* Chart A — Training Convergence */}
      <div
        id="chart-training-convergence"
        className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6"
      >
        <div className="mb-4">
          <h3 className="text-base font-semibold text-slate-800">Training Convergence</h3>
          <p className="text-xs text-slate-500 mt-0.5">{activeModel} · Loss over Epochs</p>
        </div>
        <div className="h-60">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={epochData} margin={{ top: 5, right: 16, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_COLOR} />
              <XAxis
                dataKey="epoch"
                stroke={CHART_AXIS_COLOR}
                tick={{ fill: CHART_AXIS_COLOR, fontSize: 11 }}
                label={{ value: 'Epoch', position: 'insideBottomRight', offset: -4, fill: CHART_AXIS_COLOR, fontSize: 11 }}
              />
              <YAxis
                stroke={CHART_AXIS_COLOR}
                tick={{ fill: CHART_AXIS_COLOR, fontSize: 11 }}
                tickFormatter={v => v.toFixed(3)}
              />
              <RechartsTooltip
                contentStyle={TOOLTIP_STYLE}
                formatter={(value, name) => [value.toFixed(5), name === 'loss' ? 'Train Loss' : 'Val Loss']}
              />
              <Legend
                wrapperStyle={{ fontSize: '12px', color: CHART_AXIS_COLOR }}
                formatter={(v) => v === 'loss' ? 'Train Loss' : 'Val Loss'}
              />
              <Line type="monotone" dataKey="loss" stroke="#3b82f6" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="val_loss" stroke="#10b981" dot={false} strokeWidth={2} strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Right column: Bar chart + XAI card */}
      <div className="flex flex-col gap-5">
        {/* Chart B — Cross-Model RMSE */}
        <div
          id="chart-model-comparison"
          className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6 flex-1"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-slate-800">Cross-Model Evaluation</h3>
              <p className="text-xs text-slate-500 mt-0.5">RMSE (lower is better)</p>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-500 bg-slate-50 border border-slate-100 rounded-lg px-2 py-1">
              <TrendingDown className="w-3.5 h-3.5 text-emerald-600" />
              Best: {minRmseModel.name}
            </div>
          </div>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 5, right: 16, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID_COLOR} vertical={false} />
                <XAxis
                  dataKey="name"
                  stroke={CHART_AXIS_COLOR}
                  tick={{ fill: CHART_AXIS_COLOR, fontSize: 11 }}
                />
                <YAxis
                  stroke={CHART_AXIS_COLOR}
                  tick={{ fill: CHART_AXIS_COLOR, fontSize: 11 }}
                  tickFormatter={v => v.toFixed(3)}
                />
                <RechartsTooltip
                  cursor={{ fill: CURSOR_FILL }}
                  contentStyle={TOOLTIP_STYLE}
                  formatter={(value) => [value.toFixed(4) + ' mm', 'RMSE']}
                />
                <Bar dataKey="rmse" radius={[5, 5, 0, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.name === minRmseModel.name ? '#059669' : '#93c5fd'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* XAI Insights Card */}
        <div
          id="xai-insights"
          className="bg-blue-50 border border-blue-100 rounded-xl p-4 flex items-start gap-3"
        >
          <div className="p-1.5 bg-blue-100 rounded-lg flex-shrink-0 mt-0.5">
            <Info className="w-4 h-4 text-blue-600" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-blue-700 mb-1">XAI Insights &amp; Weather Context</h4>
            <p className="text-xs text-slate-600 leading-relaxed">
              Based on the simulated features for the requested date, the baseline environment reflects{' '}
              <strong className="text-slate-700">{season}</strong> conditions
              {' '}(RH: {weatherSummary?.rh2m}%, Temp: {weatherSummary?.t2m}°C).
              The model utilized these structural dependencies across the 30-day window to map predictive rainfall curves.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
