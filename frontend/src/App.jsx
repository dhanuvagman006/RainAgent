import { useState, useEffect } from 'react';
import axios from 'axios';
import SidebarControls from './components/SidebarControls';
import KPIDashboard from './components/KPIDashboard';
import AnalyticsView from './components/AnalyticsView';
import IrrigationTimeline from './components/IrrigationTimeline';
import ValidationPanel from './components/ValidationPanel';
import ModelDeepDive from './components/ModelDeepDive';
import LeaderboardTable from './components/LeaderboardTable';
import { CloudRain, Leaf } from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [formData, setFormData] = useState({
    date: '2026-06-15',
    model_name: 'LSTM',
    horizon: 4,
    crop_name: 'Arecanut',
    catchment_area: 150.0,
    cultivation_area: 1000.0,
    initial_tank_water: 2000.0,
    max_tank_capacity: 10000.0
  });

  const [metricsData, setMetricsData]     = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [scheduleData, setScheduleData]   = useState(null);
  const [isLoading, setIsLoading]         = useState(false);
  const [error, setError]                 = useState(null);

  // Load metrics on mount
  useEffect(() => {
    axios.get(`${API_BASE}/metrics`)
      .then(res => setMetricsData(res.data))
      .catch(err => console.error("Error loading metrics:", err));
  }, []);

  const handleRunPrediction = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const predRes = await axios.post(`${API_BASE}/predict`, {
        date:       formData.date,
        model_name: formData.model_name,
        horizon:    formData.horizon
      });
      setPredictionData(predRes.data);

      const irrRes = await axios.post(`${API_BASE}/calculate-irrigation-plan`, {
        predicted_rainfall:    predRes.data.predicted_rainfall_mm,
        simulated_temperatures: Array(formData.horizon).fill(predRes.data.simulated_weather_summary.t2m),
        crop_name:             formData.crop_name,
        catchment_area:        formData.catchment_area,
        cultivation_area:      formData.cultivation_area,
        initial_tank_water:    formData.initial_tank_water,
        max_tank_capacity:     formData.max_tank_capacity
      });
      setScheduleData(irrRes.data.schedule);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred connecting to the backend.");
    } finally {
      setIsLoading(false);
    }
  };

  const finalTankStatus = scheduleData ? scheduleData[scheduleData.length - 1] : null;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">

      {/* ── Sticky Top Header ── */}
      <header className="bg-white border-b border-slate-100 shadow-sm sticky top-0 z-40">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-emerald-600 rounded-xl shadow-md shadow-emerald-200">
              <CloudRain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-800 leading-tight">
                AgriSense Forecasting
              </h1>
              <p className="text-xs text-slate-500 leading-tight">
                Deep Learning Rain Prediction &amp; Irrigation Optimizer
              </p>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-2 text-xs text-slate-500
                          bg-slate-50 border border-slate-100 rounded-lg px-3 py-2">
            <Leaf className="w-3.5 h-3.5 text-emerald-600" />
            <span>Dakshina Kannada · AgTech Dashboard</span>
          </div>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div className="max-w-screen-2xl mx-auto px-4 md:px-6 lg:px-8 py-6">
        <div className="flex flex-col md:flex-row gap-6">

          {/* Sidebar */}
          <SidebarControls
            formData={formData}
            setFormData={setFormData}
            onSubmit={handleRunPrediction}
            isLoading={isLoading}
          />

          {/* Main content column */}
          <main className="flex-1 min-w-0">

            {/* Error Banner */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3
                              rounded-xl mb-6 text-sm flex items-center gap-2">
                <span className="font-semibold">Error:</span> {error}
              </div>
            )}

            {/* ── Section A: Prediction Results (shown after run) ── */}
            {predictionData && scheduleData ? (
              <>
                <KPIDashboard
                  predictionData={predictionData}
                  finalTankStatus={finalTankStatus}
                />
                <AnalyticsView
                  metricsData={metricsData}
                  selectedModel={formData.model_name}
                  weatherSummary={predictionData.simulated_weather_summary}
                />
                <IrrigationTimeline schedule={scheduleData} />

                {/* ── Validation Panel — auto-fetches IMD ground-truth ── */}
                <ValidationPanel
                  predictionData={predictionData}
                  selectedDate={formData.date}
                  selectedModel={formData.model_name}
                  horizon={formData.horizon}
                />
              </>
            ) : (
              <div className="flex flex-col items-center justify-center pt-20 pb-10">
                <div className="p-6 bg-slate-100 rounded-full mb-5">
                  <CloudRain className="w-16 h-16 text-slate-300" />
                </div>
                <p className="text-slate-400 text-base font-medium">
                  Configure the control panel and run prediction to view insights.
                </p>
                <p className="text-slate-300 text-sm mt-1">
                  Your forecasting results will appear here.
                </p>
              </div>
            )}

            {/* ── Section B: Model Deep-Dive (always visible) ── */}
            <ModelDeepDive
              metricsData={metricsData}
              selectedModel={formData.model_name}
            />

            {/* ── Section C: Global Leaderboard (always visible) ── */}
            <LeaderboardTable metricsData={metricsData} />

          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
