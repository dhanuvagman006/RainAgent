import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  Satellite, CloudRain, RefreshCw, CheckCircle2,
  AlertTriangle, Info, WifiOff, Clock, MapPin,
  Database, ArrowRight, Zap, Trophy, Loader2
} from 'lucide-react';

const API_BASE  = 'http://127.0.0.1:8000';
const ALL_MODELS = ['LSTM', 'GRU', 'Bi-LSTM', '1D-CNN', 'CNN-LSTM', 'Transformer'];

/* ─────────────────────────────────────────────
   Variance badge
───────────────────────────────────────────── */
function VarianceBadge({ delta }) {
  if (delta === null || delta === undefined) return null;
  let bg, border, text, icon, label;
  if (delta <= 2) {
    bg = 'bg-emerald-100'; border = 'border-emerald-200'; text = 'text-emerald-700';
    icon = <CheckCircle2 className="w-3.5 h-3.5" />; label = 'High Accuracy Fit';
  } else if (delta <= 10) {
    bg = 'bg-blue-100'; border = 'border-blue-200'; text = 'text-blue-700';
    icon = <Info className="w-3.5 h-3.5" />; label = 'Moderate Deviation';
  } else {
    bg = 'bg-amber-100'; border = 'border-amber-200'; text = 'text-amber-700';
    icon = <AlertTriangle className="w-3.5 h-3.5" />; label = 'Climatic Variance Present';
  }
  return (
    <div className="flex flex-col items-center gap-2">
      <div className="text-center">
        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-1">Δ Error</p>
        <p className="text-3xl font-bold text-slate-800 leading-none tabular-nums">
          {delta.toFixed(2)}<span className="text-sm font-medium text-slate-500 ml-1">mm</span>
        </p>
      </div>
      <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${bg} ${text} border ${border}`}>
        {icon}{label}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Reading column (left / right)
───────────────────────────────────────────── */
function ReadingColumn({ side, label, subLabel, value, isLoading, accent }) {
  const c = {
    blue:    { icon:'text-blue-600',    iconBg:'bg-blue-50 border-blue-100',    label:'text-blue-700',    value:'text-blue-800',    bar:'bg-blue-500',    unit:'text-blue-400'    },
    emerald: { icon:'text-emerald-600', iconBg:'bg-emerald-50 border-emerald-100', label:'text-emerald-700', value:'text-emerald-800', bar:'bg-emerald-500', unit:'text-emerald-400' },
  }[accent];
  const Icon = side === 'model' ? CloudRain : Satellite;
  return (
    <div className="flex-1 flex flex-col items-center text-center px-4 py-5">
      <div className={`p-3 rounded-xl border ${c.iconBg} mb-3`}>
        <Icon className={`w-6 h-6 ${c.icon}`} />
      </div>
      <p className={`text-xs font-bold uppercase tracking-wider mb-0.5 ${c.label}`}>{label}</p>
      <p className="text-xs text-slate-400 mb-4 leading-snug max-w-[160px]">{subLabel}</p>
      {isLoading ? (
        <div className="animate-pulse space-y-2 w-full flex flex-col items-center">
          <div className="h-12 w-32 bg-slate-100 rounded-lg" />
          <div className="h-3 w-20 bg-slate-100 rounded" />
        </div>
      ) : value !== null && value !== undefined ? (
        <>
          <p className={`text-5xl font-black leading-none tabular-nums ${c.value}`}>{value.toFixed(1)}</p>
          <p className={`text-base font-semibold mt-1 ${c.unit}`}>mm</p>
          <div className={`mt-4 h-1 w-16 rounded-full ${c.bar} opacity-40`} />
        </>
      ) : (
        <p className="text-3xl font-bold text-slate-300">—</p>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Centre divider
───────────────────────────────────────────── */
function CenterDivider({ delta, isLoading }) {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-2 flex-shrink-0">
      <div className="w-px flex-1 bg-slate-100 mb-4" />
      {isLoading ? (
        <div className="animate-pulse flex flex-col items-center gap-2">
          <div className="h-8 w-16 bg-slate-100 rounded-lg" />
          <div className="h-5 w-24 bg-slate-100 rounded-full" />
        </div>
      ) : <VarianceBadge delta={delta} />}
      <div className="flex items-center gap-1 mt-4 text-slate-300">
        <ArrowRight className="w-3 h-3 rotate-180" />
        <ArrowRight className="w-3 h-3" />
      </div>
      <div className="w-px flex-1 bg-slate-100 mt-4" />
    </div>
  );
}

/* ─────────────────────────────────────────────
   Status chip
───────────────────────────────────────────── */
function StatusChip({ status }) {
  const map = {
    loading:  { bg:'bg-blue-50 border-blue-100',       text:'text-blue-600',    dot:'bg-blue-400 animate-pulse', label:'Fetching Data…'                    },
    live:     { bg:'bg-emerald-50 border-emerald-100',  text:'text-emerald-700', dot:'bg-emerald-500',            label:'Ground-Truth Available'             },
    live_api: { bg:'bg-violet-50 border-violet-100',   text:'text-violet-700',  dot:'bg-violet-500',             label:'Live NASA POWER · Real-Time'        },
    lag:      { bg:'bg-amber-50 border-amber-100',      text:'text-amber-700',   dot:'bg-amber-400 animate-pulse',label:'Within NASA Processing Lag (~3 days)'},
    future:   { bg:'bg-amber-50 border-amber-100',      text:'text-amber-700',   dot:'bg-amber-400',              label:'Future Date — No Historical Record'  },
    error:    { bg:'bg-red-50 border-red-100',          text:'text-red-600',     dot:'bg-red-400',                label:'Validation Service Unavailable'     },
    idle:     { bg:'bg-slate-50 border-slate-100',      text:'text-slate-500',   dot:'bg-slate-300',              label:'Run a prediction to validate'       },
  };
  const s = map[status] ?? map.idle;
  return (
    <span className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-semibold border ${s.bg} ${s.text}`}>
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${s.dot}`} />
      {s.label}
    </span>
  );
}

/* ─────────────────────────────────────────────
   Best-Model Finder sub-component
───────────────────────────────────────────── */
function BestModelFinder({ imdValue, selectedDate, horizon }) {
  const [scanning, setScanning]   = useState(false);
  const [results,  setResults]    = useState(null);   // [{model, predicted, delta}]
  const [scanErr,  setScanErr]    = useState(null);

  const runScan = useCallback(async () => {
    if (!imdValue || !selectedDate) return;
    setScanning(true);
    setScanErr(null);
    setResults(null);

    const requests = ALL_MODELS.map(m =>
      axios.post(`${API_BASE}/predict`, { date: selectedDate, model_name: m, horizon })
        .then(r => ({
          model:     m,
          predicted: r.data.predicted_rainfall_mm[0],
          delta:     Math.abs(r.data.predicted_rainfall_mm[0] - imdValue),
        }))
        .catch(() => ({ model: m, predicted: null, delta: Infinity }))
    );

    const rows = await Promise.all(requests);
    // Sort ascending by delta
    rows.sort((a, b) => a.delta - b.delta);
    setResults(rows);
    setScanning(false);
  }, [imdValue, selectedDate, horizon]);

  if (!imdValue) return null;

  const winner = results?.[0];
  const maxDelta = results ? Math.max(...results.map(r => r.delta === Infinity ? 0 : r.delta)) : 1;

  return (
    <div className="border-t border-slate-100">
      {/* Trigger bar */}
      <div className="px-5 py-3 bg-slate-50/60 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-blue-50 border border-blue-100 rounded-lg">
            <Zap className="w-4 h-4 text-blue-600" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-800">Least Deviation Finder</p>
            <p className="text-xs text-slate-500">
              Scan all {ALL_MODELS.length} architectures vs IMD actual ({imdValue} mm) and rank by Δ Error
            </p>
          </div>
        </div>
        <button
          id="scan-all-models-btn"
          onClick={runScan}
          disabled={scanning}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold
                     bg-blue-600 hover:bg-blue-700 text-white shadow-sm shadow-blue-200
                     transition-all active:scale-[0.97] disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {scanning
            ? <><Loader2 className="w-4 h-4 animate-spin" />Scanning all models…</>
            : <><Zap className="w-4 h-4" />{results ? 'Re-scan Models' : 'Find Best Model'}</>
          }
        </button>
      </div>

      {/* Results table */}
      {scanning && (
        <div className="px-5 pb-5">
          <div className="animate-pulse space-y-2 pt-4">
            {ALL_MODELS.map(m => (
              <div key={m} className="flex items-center gap-3">
                <div className="h-3 w-20 bg-slate-100 rounded" />
                <div className="h-3 flex-1 bg-slate-100 rounded-full" />
                <div className="h-3 w-14 bg-slate-100 rounded" />
              </div>
            ))}
          </div>
        </div>
      )}

      {scanErr && (
        <div className="px-5 pb-4 pt-3 text-xs text-red-500">{scanErr}</div>
      )}

      {results && !scanning && (
        <div className="px-5 pb-5 pt-4">
          {/* Winner callout */}
          <div className="flex items-center gap-3 mb-4 p-3.5 rounded-xl
                          bg-amber-50 border border-amber-200">
            <div className="p-2 bg-amber-100 rounded-lg flex-shrink-0">
              <Trophy className="w-5 h-5 text-amber-600" />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold text-amber-600 uppercase tracking-wider">
                Least Deviation Model
              </p>
              <p className="text-base font-bold text-slate-800 leading-tight">
                {winner.model}
                <span className="ml-2 text-sm font-semibold text-amber-600">
                  Δ {winner.delta.toFixed(2)} mm
                </span>
              </p>
              {winner.predicted !== null && (
                <p className="text-xs text-slate-500 mt-0.5">
                  Predicted {winner.predicted.toFixed(2)} mm vs IMD {imdValue.toFixed(2)} mm
                </p>
              )}
            </div>
          </div>

          {/* Ranked list */}
          <div className="space-y-2">
            {results.map((row, idx) => {
              const isWinner  = idx === 0;
              const barWidth  = maxDelta > 0 ? Math.max(4, (row.delta / maxDelta) * 100) : 4;
              const barColor  = isWinner ? 'bg-emerald-500' :
                                idx === 1 ? 'bg-blue-400' :
                                idx === 2 ? 'bg-blue-300' : 'bg-slate-200';
              const textDelta = isWinner ? 'text-emerald-700 font-bold' :
                                row.delta === Infinity ? 'text-red-400' : 'text-slate-600';

              return (
                <div
                  key={row.model}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-all ${
                    isWinner
                      ? 'bg-emerald-50/60 border-emerald-200 border-l-4 border-l-emerald-500'
                      : 'bg-white border-slate-100 hover:bg-slate-50/60'
                  }`}
                >
                  {/* Rank */}
                  <span className="w-6 text-center text-xs font-mono flex-shrink-0">
                    {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `#${idx + 1}`}
                  </span>

                  {/* Model name */}
                  <span className={`w-20 text-sm font-semibold flex-shrink-0 ${
                    isWinner ? 'text-emerald-800' : 'text-slate-700'
                  }`}>
                    {row.model}
                  </span>

                  {/* Progress bar */}
                  <div className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${barColor}`}
                      style={{ width: `${row.delta === Infinity ? 100 : barWidth}%` }}
                    />
                  </div>

                  {/* Predicted */}
                  <span className="w-20 text-xs text-slate-400 text-right flex-shrink-0 font-mono">
                    {row.predicted !== null ? `${row.predicted.toFixed(2)} mm` : 'error'}
                  </span>

                  {/* Delta */}
                  <span className={`w-20 text-right text-sm font-mono flex-shrink-0 ${textDelta}`}>
                    {row.delta === Infinity ? '—' : `Δ ${row.delta.toFixed(2)}`}
                  </span>
                </div>
              );
            })}
          </div>

          <p className="text-[11px] text-slate-400 mt-3 text-right italic">
            IMD Ground-Truth: {imdValue.toFixed(2)} mm · {selectedDate} · Ranked by |Predicted − Actual|
          </p>
        </div>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────
   MAIN EXPORT
───────────────────────────────────────────── */
export default function ValidationPanel({ predictionData, selectedDate, selectedModel, horizon = 1 }) {
  const [imdData,   setImdData]   = useState(null);
  const [status,    setStatus]    = useState('idle');
  const [fetchedAt, setFetchedAt] = useState(null);

  const modelValue = predictionData?.predicted_rainfall_mm?.[0] ?? null;

  const fetchValidation = useCallback(async (date) => {
    setStatus('loading');
    setImdData(null);
    try {
      const res = await axios.get(`${API_BASE}/validate-actual-rainfall`, { params: { date } });
      const d = res.data;
      setImdData(d);
      if (d.data_available) {
        setStatus(d.data_source_type === 'nasa_power_live' ? 'live_api' : 'live');
      } else if (d.note?.toLowerCase().includes('future')) {
        setStatus('future');
      } else if (d.note?.toLowerCase().includes('lag') || d.note?.toLowerCase().includes('not yet')) {
        setStatus('lag');
      } else {
        setStatus('future');
      }
      setFetchedAt(new Date().toLocaleTimeString('en-IN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
      }));
    } catch {
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    if (predictionData && selectedDate) fetchValidation(selectedDate);
  }, [predictionData, selectedDate, fetchValidation]);

  const imdValue = imdData?.data_available ? imdData.imd_actual_rainfall_mm : null;
  const delta    = (modelValue !== null && imdValue !== null) ? Math.abs(modelValue - imdValue) : null;

  if (!predictionData) return null;
  const isLoading = status === 'loading';

  return (
    <section id="validation-panel" className="mb-6">
      {/* Section header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="w-1 h-6 bg-emerald-500 rounded-full" />
          <div>
            <h2 className="text-base font-semibold text-slate-800">Live Ground-Truth Validation</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Model forecast vs. IMD observed rainfall · Day-1 comparison
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <StatusChip status={status} />
          <button
            onClick={() => fetchValidation(selectedDate)}
            disabled={isLoading}
            aria-label="Refresh IMD validation"
            className="p-1.5 rounded-lg border border-slate-200 bg-white text-slate-500
                       hover:text-slate-700 hover:border-slate-300 hover:shadow-sm
                       transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Main card */}
      <div className="bg-white border border-slate-100 shadow-sm hover:shadow-md
                      transition-shadow duration-200 rounded-xl overflow-hidden">

        {/* Dual-column comparison */}
        <div className="flex items-stretch divide-x divide-slate-50">
          <ReadingColumn
            side="model" label="Model Forecast"
            subLabel={`${selectedModel} · Day-1 Predicted Rainfall`}
            value={modelValue} isLoading={false} accent="blue"
          />

          <CenterDivider delta={delta} isLoading={isLoading} />

          {status === 'error' ? (
            <div className="flex-1 flex flex-col items-center justify-center px-4 py-8 text-center gap-3">
              <div className="p-3 rounded-xl bg-red-50 border border-red-100">
                <WifiOff className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-600">Validation service offline</p>
                <p className="text-xs text-slate-400 mt-0.5">Check that the backend API is running</p>
              </div>
            </div>
          ) : status === 'lag' ? (
            <div className="flex-1 flex flex-col items-center justify-center px-4 py-8 text-center gap-3">
              <div className="p-3 rounded-xl bg-amber-50 border border-amber-100">
                <Clock className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-600">Data not yet processed</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  NASA POWER has a ~3-day processing lag.{selectedDate && ` (${selectedDate})`}
                </p>
              </div>
            </div>
          ) : status === 'future' ? (
            <div className="flex-1 flex flex-col items-center justify-center px-4 py-8 text-center gap-3">
              <div className="p-3 rounded-xl bg-amber-50 border border-amber-100">
                <Clock className="w-6 h-6 text-amber-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-600">Future date selected</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  Ground-truth is only available for past dates.{selectedDate && ` (${selectedDate})`}
                </p>
              </div>
            </div>
          ) : (
            <ReadingColumn
              side="imd" label="IMD Ground-Truth"
              subLabel="Observed Actual Rainfall · Official Station"
              value={imdValue} isLoading={isLoading} accent="emerald"
            />
          )}
        </div>

        {/* Best-Model Finder — only shown when IMD data is live */}
        {(status === 'live' || status === 'live_api') && imdValue !== null && (
          <BestModelFinder
            imdValue={imdValue}
            selectedDate={selectedDate}
            horizon={horizon}
          />
        )}

        {/* Metadata footer */}
        <div className="border-t border-slate-50 bg-slate-50/50 px-5 py-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
              <span>
                <strong className="text-slate-600">Source:</strong>{' '}
                {imdData?.source ?? 'India Meteorological Department (IMD) · NASA POWER'}
                {imdData?.data_source_type === 'nasa_power_live' && (
                  <span className="ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold bg-violet-100 text-violet-700 border border-violet-200">
                    ⚡ LIVE API
                  </span>
                )}
              </span>
            </span>
            <span className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
              <span>
                <strong className="text-slate-600">Station:</strong>{' '}
                {imdData?.station ?? 'Dakshina Kannada Region (Mangaluru)'}
              </span>
            </span>
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
              <span>
                <strong className="text-slate-600">Date Profile:</strong>{' '}
                {selectedDate ?? '—'}
                {fetchedAt && <span className="text-slate-400 ml-1.5">· fetched {fetchedAt}</span>}
              </span>
            </span>
            {imdData?.dataset_range && (
              <span className="text-slate-400 italic">Dataset window: {imdData.dataset_range}</span>
            )}
          </div>
          {imdData?.note && (
            <p className="text-[11px] text-slate-400 italic mt-1.5 leading-relaxed">{imdData.note}</p>
          )}
        </div>
      </div>
    </section>
  );
}
