'use client';

import { TrafficStats, Theme } from '@/types/smart-traffic';
import { useVehicleCounter } from '@/hooks/useVehicleCounter';
import VideoPlayer from './VideoPlayer';
import VehicleStats from './VehicleStats';
import LiveClock from './LiveClock';

interface VideoAnalysisSectionProps {
  isVideoPlaying: boolean;
  videoVolume: number;
  isFullscreen: boolean;
  trafficStats: TrafficStats;
  theme: Theme;
  onPlayPause: () => void;
  onVolumeChange: (volume: number) => void;
  onFullscreen: () => void;
}

export default function VideoAnalysisSection({
  isVideoPlaying,
  videoVolume,
  isFullscreen,
  trafficStats,
  theme,
  onPlayPause,
  onVolumeChange,
  onFullscreen
}: VideoAnalysisSectionProps) {
  // Use the vehicle counter hook for real-time backend data
  const {
    vehicleStats,
    isConnected,
    isLoading,
    error,
    lastUpdate,
    resetCounters,
    togglePlayPause,
    fetchVehicleCounts
  } = useVehicleCounter();

  // Debug logging
  console.log('📊 VideoAnalysisSection - Vehicle Stats:', vehicleStats);
  console.log('🔌 Connection Status:', isConnected);
  console.log('⏱️ Last Update:', lastUpdate);

  const handleResetCounters = async () => {
    await resetCounters();
  };

  const handlePlayPause = async () => {
    const newPlayingState = await togglePlayPause();
    onPlayPause(); // Also call the parent callback for UI updates
    return newPlayingState;
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: theme.primary, margin: 0 }}>Live Traffic Video Analysis : Bhubaneshwar</h2>
        
        {/* Debug Controls */}
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => fetchVehicleCounts()}
            style={{
              padding: '8px 15px',
              background: theme.primary,
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            🔄 Manual Refresh
          </button>
          <div style={{
            padding: '8px 15px',
            background: isConnected ? '#4caf5010' : '#f4433620',
            borderRadius: '5px',
            fontSize: '12px',
            color: isConnected ? '#4caf50' : '#f44336'
          }}>
            {isConnected ? '✅ Connected' : '❌ Disconnected'}
          </div>
        </div>
      </div>
      
      <div style={{
        background: theme.background,
        borderRadius: '15px',
        padding: '30px',
        border: `2px solid ${theme.primary}10`,
        position: 'relative'
      }}>
        <LiveClock theme={theme} />
        
        <VideoPlayer
          isPlaying={isVideoPlaying}
          volume={videoVolume}
          isFullscreen={isFullscreen}
          theme={theme}
          onPlayPause={handlePlayPause}
          onVolumeChange={onVolumeChange}
          onFullscreen={onFullscreen}
          onResetCounters={handleResetCounters}
        />

        <VehicleStats
          stats={vehicleStats}
          theme={theme}
          isConnected={isConnected}
          lastUpdate={lastUpdate}
          error={error}
        />
      </div>
    </div>
  );
}