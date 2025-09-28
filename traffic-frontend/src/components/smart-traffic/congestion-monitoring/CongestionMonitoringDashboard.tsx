'use client';

import { useState, useEffect } from 'react';
import SummaryCards from './SummaryCards';
import AreaMonitor from './AreaMonitor';
import RouteFinder from './RouteFinder';
import AIPredictions from './AIPredictions';
import { initialAreas } from '@/lib/data';
import type { Area } from '@/lib/types';

const CongestionMonitoringDashboard = () => {
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [selectedArea, setSelectedArea] = useState<Area | null>(initialAreas[0]);

  useEffect(() => {
    const interval = setInterval(() => setLastUpdate(new Date()), 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 md:p-6 lg:p-8">
      <div className="mb-6 flex flex-col items-center text-center">
        <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
          Congestion Monitoring Dashboard
        </h1>
        <p className="mt-2 max-w-2xl text-gray-600">
          Real-time traffic analysis and smart route optimization powered by AI.
        </p>
        <div className="mt-4 flex items-center gap-2 text-sm text-gray-500">
          <div className="relative flex h-3 w-3">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex h-3 w-3 rounded-full bg-green-500"></span>
          </div>
          Live Data • Updated {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      <div className="space-y-6">
        <SummaryCards areas={initialAreas} />
        <AreaMonitor areas={initialAreas} selectedArea={selectedArea} onAreaSelect={setSelectedArea} />
        
        <div className="grid gap-6 lg:grid-cols-2">
          <RouteFinder />
          <AIPredictions selectedArea={selectedArea} />
        </div>
      </div>
    </div>
  );
};

export default CongestionMonitoringDashboard;