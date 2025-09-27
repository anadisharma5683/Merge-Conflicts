'use client';

import { Play, Pause, Volume2, Maximize } from 'lucide-react';
import { Theme } from '@/types/smart-traffic';

interface VideoPlayerProps {
  isPlaying: boolean;
  volume: number;
  isFullscreen: boolean;
  theme: Theme;
  onPlayPause: () => Promise<boolean> | void;
  onVolumeChange: (volume: number) => void;
  onFullscreen: () => void;
  onResetCounters: () => Promise<void> | void;
}

export default function VideoPlayer({
  isPlaying,
  volume,
  isFullscreen,
  theme,
  onPlayPause,
  onVolumeChange,
  onFullscreen,
  onResetCounters
}: VideoPlayerProps) {
  const handlePlayPause = async () => {
    await onPlayPause();
  };

  const handleResetCounters = async () => {
    await onResetCounters();
  };

  return (
    <div style={{
      position: 'relative',
      background: '#000',
      borderRadius: '10px',
      marginBottom: '20px',
      overflow: 'hidden'
    }}>
      {/* Real Video Stream from Backend */}
      <img 
        src="http://127.0.0.1:5000/video_feed"
        alt="Live Traffic Feed"
        style={{
          width: '100%',
          height: '500px',
          objectFit: 'cover',
          display: 'block'
        }}
        onError={(e) => {
          // Fallback to demo content if backend is not available
          const target = e.target as HTMLImageElement;
          target.style.display = 'none';
          const fallback = target.parentElement?.querySelector('.video-fallback') as HTMLElement;
          if (fallback) fallback.style.display = 'flex';
        }}
      />
      
      {/* Fallback content when backend is not available */}
      <div className="video-fallback" style={{
        width: '100%',
        height: '400px',
        display: 'none',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontSize: '18px',
        position: 'relative',
        flexDirection: 'column'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '60px', marginBottom: '20px' }}>🎥</div>
          <div>Backend Connection Required</div>
          <div style={{ fontSize: '14px', opacity: 0.7, marginTop: '10px' }}>
            Start the Python backend at http://127.0.0.1:5000
          </div>
        </div>
      </div>

      {/* Video Controls */}
      <div style={{
        position: 'absolute',
        bottom: '0',
        left: '0',
        right: '0',
        background: 'rgba(0,0,0,0.8)',
        padding: '15px',
        display: 'flex',
        alignItems: 'center',
        gap: '15px'
      }}>
        <button
          onClick={handlePlayPause}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            fontSize: '20px'
          }}
        >
          {isPlaying ? <Pause /> : <Play />}
        </button>

        <button
          onClick={handleResetCounters}
          style={{
            background: theme.primary,
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            padding: '5px 10px',
            borderRadius: '3px',
            fontSize: '12px'
          }}
        >
          Reset Count
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Volume2 size={18} color="white" />
          <input
            type="range"
            min="0"
            max="100"
            value={volume}
            onChange={(e) => onVolumeChange(Number(e.target.value))}
            style={{
              width: '100px',
              height: '5px'
            }}
          />
        </div>

        <div style={{ flex: 1 }} />

        <button
          onClick={onFullscreen}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          <Maximize size={18} />
        </button>
      </div>
    </div>
  );
}