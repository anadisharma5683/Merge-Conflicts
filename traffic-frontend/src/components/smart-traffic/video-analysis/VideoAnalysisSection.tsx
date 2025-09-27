'use client';

import { TrafficStats, Theme } from '@/types/smart-traffic';
import VideoPlayer from './VideoPlayer';
import VehicleStats from './VehicleStats';

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
  const handleResetCounters = () => {
    // This will be handled by the backend integration
    console.log('Counters reset');
  };

  return (
    <div>
      <h2 style={{ color: theme.primary, marginBottom: '20px' }}>
        Live Traffic Video Analysis : Bhubaneshwar
      </h2>
      
      <div style={{
        background: theme.background,
        borderRadius: '15px',
        padding: '30px',
        border: `2px solid ${theme.primary}10`
      }}>
        <VideoPlayer
          isPlaying={isVideoPlaying}
          volume={videoVolume}
          isFullscreen={isFullscreen}
          theme={theme}
          onPlayPause={onPlayPause}
          onVolumeChange={onVolumeChange}
          onFullscreen={onFullscreen}
          onResetCounters={handleResetCounters}
        />

        <VehicleStats
          stats={trafficStats}
          theme={theme}
        />
      </div>
    </div>
  );
}