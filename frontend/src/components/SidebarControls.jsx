import { Settings, Calendar, Maximize, Target, Database, Zap } from 'lucide-react';

const inputClass =
  "w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-slate-800 text-sm " +
  "placeholder:text-slate-400 " +
  "focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 " +
  "transition-all duration-150";

const labelClass = "flex items-center gap-1.5 text-xs font-medium text-slate-500 mb-1.5";

export default function SidebarControls({ formData, setFormData, onSubmit, isLoading }) {
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ["horizon", "catchment_area", "cultivation_area", "initial_tank_water", "max_tank_capacity"].includes(name)
              ? Number(value) : value
    }));
  };

  return (
    <div
      className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6 w-full md:w-80 flex-shrink-0 text-sm"
    >
      {/* Panel Header */}
      <div className="flex items-center gap-2 mb-6 pb-4 border-b border-slate-100">
        <div className="p-1.5 bg-emerald-50 rounded-lg">
          <Settings className="text-emerald-600 w-4 h-4" />
        </div>
        <h2 className="text-base font-semibold text-slate-800">Control Panel</h2>
      </div>

      <div className="space-y-4">
        {/* Date Picker */}
        <div>
          <label className={labelClass}>
            <Calendar className="w-3.5 h-3.5 text-slate-400" /> Target Date
          </label>
          <input
            type="date"
            name="date"
            value={formData.date}
            onChange={handleChange}
            className={inputClass}
          />
        </div>

        {/* Model Selector */}
        <div>
          <label className={labelClass}>
            <Target className="w-3.5 h-3.5 text-slate-400" /> Model Selector
          </label>
          <select
            name="model_name"
            value={formData.model_name}
            onChange={handleChange}
            className={inputClass + " cursor-pointer"}
          >
            {["LSTM", "GRU", "Bi-LSTM", "1D-CNN", "CNN-LSTM", "Transformer"].map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* Crop Selector */}
        <div>
          <label className={labelClass}>
            <Target className="w-3.5 h-3.5 text-slate-400" /> Target Crop
          </label>
          <select
            name="crop_name"
            value={formData.crop_name}
            onChange={handleChange}
            className={inputClass + " cursor-pointer"}
          >
            {["Arecanut", "Coconut", "Black Pepper"].map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Horizon Tabs */}
        <div>
          <label className={labelClass}>Forecast Horizon (Days)</label>
          <div className="flex bg-slate-50 rounded-lg p-1 border border-slate-200 gap-1">
            {[1, 4, 7].map(h => (
              <button
                key={h}
                type="button"
                onClick={() => setFormData(prev => ({ ...prev, horizon: h }))}
                className={`flex-1 py-1.5 text-center rounded-md text-sm font-medium transition-all duration-150 ${
                  formData.horizon === h
                    ? 'bg-white text-slate-800 shadow-sm border border-slate-200'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {h}D
              </button>
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-slate-100 pt-2">
          <p className="text-[11px] uppercase tracking-widest text-slate-400 font-semibold mb-3">
            Hydrological Parameters
          </p>
        </div>

        {/* Numerical Inputs */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className={labelClass}>
              <Maximize className="w-3 h-3 text-slate-400" /> Catchment (m²)
            </label>
            <input
              type="number" name="catchment_area" value={formData.catchment_area} onChange={handleChange}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>
              <Maximize className="w-3 h-3 text-slate-400" /> Field (m²)
            </label>
            <input
              type="number" name="cultivation_area" value={formData.cultivation_area} onChange={handleChange}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>
              <Database className="w-3 h-3 text-slate-400" /> Init Tank (L)
            </label>
            <input
              type="number" name="initial_tank_water" value={formData.initial_tank_water} onChange={handleChange}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>
              <Database className="w-3 h-3 text-slate-400" /> Max Tank (L)
            </label>
            <input
              type="number" name="max_tank_capacity" value={formData.max_tank_capacity} onChange={handleChange}
              className={inputClass}
            />
          </div>
        </div>

        {/* Submit Button */}
        <button
          id="run-prediction-btn"
          onClick={onSubmit}
          disabled={isLoading}
          className="w-full mt-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed
                     text-white font-semibold py-3 rounded-xl transition-all duration-150
                     active:scale-[0.98] flex justify-center items-center gap-2 shadow-md shadow-emerald-200
                     text-sm tracking-wide"
        >
          {isLoading ? (
            <>
              <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Analysing…
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Run Prediction &amp; Optimization
            </>
          )}
        </button>
      </div>
    </div>
  );
}
