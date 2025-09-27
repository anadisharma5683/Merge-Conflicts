'use client';

import { useState, useEffect, useCallback } from 'react';
import { TrafficStats, BackendVehicleResponse } from '@/types/smart-traffic';

const BACKEND_URL = 'http://127.0.0.1:5000';
const POLLING_INTERVAL = 2000; // 2 seconds

export const useVehicleCounter = () => {
  const [vehicleStats, setVehicleStats] = useState<TrafficStats>({
    cars: 0,
    trucks: 0,
    buses: 0,
    motorcycles: 0,
    total: 0
  });
  
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Fetch vehicle counts from backend
  const fetchVehicleCounts = useCallback(async () => {
    console.log('🔄 Fetching vehicle counts from:', `${BACKEND_URL}/get_counts`);
    try {
      const response = await fetch(`${BACKEND_URL}/get_counts`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('📡 Response status:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: BackendVehicleResponse = await response.json();
      console.log('📊 Received data:', data);
      
      // Handle different response formats from the backend
      const newStats = {
        cars: typeof data.cars === 'number' ? data.cars : (parseInt(data.cars as string) || 0),
        trucks: typeof data.trucks === 'number' ? data.trucks : (parseInt(data.trucks as string) || 0), 
        buses: typeof data.buses === 'number' ? data.buses : (parseInt(data.buses as string) || 0),
        motorcycles: typeof data.motorcycles === 'number' ? data.motorcycles : (parseInt(data.motorcycles as string) || 0),
        total: typeof data.total === 'number' ? data.total : (parseInt(data.total as string) || 0)
      };
      
      console.log('📈 Processed stats:', newStats);
      setVehicleStats(newStats);
      
      setIsConnected(true);
      setError(null);
      setLastUpdate(new Date());
      console.log('✅ Vehicle stats updated successfully');
      
    } catch (err) {
      console.error('❌ Failed to fetch vehicle counts:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect to backend');
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reset counters
  const resetCounters = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/reset_counters`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Reset local state immediately for responsive UI
      setVehicleStats({
        cars: 0,
        trucks: 0,
        buses: 0,
        motorcycles: 0,
        total: 0
      });
      
      setLastUpdate(new Date());
      
    } catch (err) {
      console.error('Failed to reset counters:', err);
      setError(err instanceof Error ? err.message : 'Failed to reset counters');
    }
  }, []);

  // Control video playback
  const togglePlayPause = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/play_pause`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.is_playing;
      
    } catch (err) {
      console.error('Failed to toggle playback:', err);
      setError(err instanceof Error ? err.message : 'Failed to control playback');
      return false;
    }
  }, []);

  // Set up polling for real-time updates
  useEffect(() => {
    // Initial fetch
    fetchVehicleCounts();

    // Set up polling interval
    const interval = setInterval(fetchVehicleCounts, POLLING_INTERVAL);

    return () => clearInterval(interval);
  }, [fetchVehicleCounts]);

  return {
    vehicleStats,
    isConnected,
    isLoading,
    error,
    lastUpdate,
    resetCounters,
    togglePlayPause,
    fetchVehicleCounts
  };
};