'use client';

import { useState, useEffect } from 'react';
import { VehicleCounts, BackendStatus } from '@/types/traffic';

export const useTrafficData = (videoStarted: boolean) => {
  const [counts, setCounts] = useState<VehicleCounts>({
    car: 0,
    motorcycle: 0,
    bus: 0,
    truck: 0,
    total: 0,
  });
  const [isPlaying, setIsPlaying] = useState(true);
  const [frameSkip, setFrameSkip] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:5000';

  const togglePlayPause = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/play_pause`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setIsPlaying(data.is_playing);
        setError(null);
      } else {
        throw new Error('Failed to toggle play/pause');
      }
    } catch (error) {
      console.error('Error toggling play/pause:', error);
      setError('Failed to toggle playback');
    } finally {
      setIsLoading(false);
    }
  };

  const resetCounters = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/reset_counters`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setCounts(data.counters);
        setCurrentFrame(0);
        setError(null);
      } else {
        throw new Error('Failed to reset counters');
      }
    } catch (error) {
      console.error('Error resetting counters:', error);
      setError('Failed to reset counters');
    } finally {
      setIsLoading(false);
    }
  };

  const updateFrameSkip = async (newSkipValue: number) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/set_frame_skip`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ skip_value: newSkipValue }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setFrameSkip(data.frame_skip);
        setError(null);
      } else {
        throw new Error('Failed to update frame skip');
      }
    } catch (error) {
      console.error('Error updating frame skip:', error);
      setError('Failed to update frame skip');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (videoStarted) {
      const intervalId = setInterval(() => {
        fetch(`${backendUrl}/get_counts`)
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
            return response.json();
          })
          .then((data: BackendStatus) => {
            setCounts(data.counters);
            setIsPlaying(data.is_playing);
            setFrameSkip(data.frame_skip);
            setCurrentFrame(data.current_frame);
            setError(null);
          })
          .catch(error => {
            console.error('Error fetching counts:', error);
            setError('Connection error. Please check if backend is running.');
          });
      }, 1000);

      return () => clearInterval(intervalId);
    }
  }, [videoStarted, backendUrl]);

  return {
    counts,
    isPlaying,
    frameSkip,
    currentFrame,
    isLoading,
    error,
    backendUrl,
    togglePlayPause,
    resetCounters,
    updateFrameSkip,
  };
};