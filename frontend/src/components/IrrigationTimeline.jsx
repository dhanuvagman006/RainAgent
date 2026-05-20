import { Sun, CloudRain, Cloud, AlertTriangle, Droplet } from 'lucide-react';

export default function IrrigationTimeline({ schedule }) {
  if (!schedule || schedule.length === 0) return null;

  const getWeatherIcon = (rain) => {
    if (rain === 0)  return <Sun className="w-7 h-7 text-amber-400" />;
    if (rain < 10)   return <Cloud className="w-7 h-7 text-slate-400" />;
    return                  <CloudRain className="w-7 h-7 text-blue-500" />;
  };

  return (
    <div
      id="irrigation-timeline"
      className="bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow duration-200 rounded-xl p-6 mb-6"
    >
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-5">
        <div className="p-1.5 bg-blue-50 rounded-lg">
          <Droplet className="w-4 h-4 text-blue-600" />
        </div>
        <div>
          <h3 className="text-base font-semibold text-slate-800">Optimised N-Day Irrigation Planner</h3>
          <p className="text-xs text-slate-500 mt-0.5">Hover any card to see detailed water balance</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {schedule.map((dayData, idx) => {
          const isWarning     = dayData.action.includes("Partial");
          const noIrrigation  = dayData.action === "No Irrigation";

          return (
            <div
              key={idx}
              className="bg-white border border-slate-100 rounded-xl p-4 flex flex-col relative overflow-hidden
                         shadow-sm hover:shadow-md transition-all duration-200 group"
            >
              {/* Warning stripe */}
              {isWarning && (
                <div className="absolute top-0 left-0 w-full bg-amber-400 text-white text-[10px] font-bold
                                py-1 px-2 flex items-center justify-center gap-1 tracking-wide z-10">
                  <AlertTriangle className="w-3 h-3" /> TANK DEFICIT
                </div>
              )}

              {/* Day Header */}
              <div className={`flex justify-between items-start ${isWarning ? 'mt-5' : ''} mb-4`}>
                <span className="bg-slate-100 text-slate-600 text-xs px-2.5 py-1 rounded-md font-mono font-semibold">
                  DAY {dayData.day_index}
                </span>
                {getWeatherIcon(dayData.predicted_rainfall_mm)}
              </div>

              {/* Climate Stats */}
              <div className="mb-4">
                <p className="text-2xl font-bold text-slate-800 leading-tight mb-0.5">
                  {dayData.predicted_rainfall_mm}
                  <span className="text-sm font-normal text-slate-400 ml-1">mm</span>
                </p>
                <p className="text-xs text-slate-500">{dayData.simulated_temperature_c}°C avg temp</p>
              </div>

              {/* Action Badge */}
              <div className="mt-auto">
                <div className={`w-full py-2 px-3 rounded-lg text-center text-xs font-semibold tracking-wide ${
                  noIrrigation
                    ? 'bg-slate-100 text-slate-700'
                    : 'bg-emerald-100 text-emerald-800'
                }`}>
                  {noIrrigation
                    ? 'No Irrigation Required'
                    : `Irrigate: ${dayData.crop_water_needed_liters} L`}
                </div>
              </div>

              {/* Hover Details Overlay */}
              <div className="absolute inset-0 bg-white/97 backdrop-blur-sm p-4 flex flex-col justify-center
                              opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-xl border border-slate-100">
                <p className="text-xs font-semibold text-slate-700 mb-3">Water Balance Detail</p>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Harvested</span>
                    <span className="font-medium text-slate-800">{dayData.harvested_water_liters} L</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Tank (EOD)</span>
                    <span className="font-medium text-slate-800">{dayData.tank_status_liters} L</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Capacity</span>
                    <span className="font-medium text-slate-800">{dayData.tank_status_percentage}%</span>
                  </div>
                </div>
                <p className="text-[10px] text-slate-400 italic mt-3 border-t border-slate-100 pt-2 leading-relaxed">
                  {dayData.status_message}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
