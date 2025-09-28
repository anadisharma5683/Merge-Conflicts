'use client';

import { CrossPath, Theme } from '@/types/smart-traffic';
import { getCongestionColor } from '@/lib/smart-traffic-theme';
import MiniVideoPlayer from './MiniVideoPlayer';

interface CrossPathDetailsProps {
  crossPath: CrossPath;
  theme: Theme;
  onClose: () => void;
}

export default function CrossPathDetails({ crossPath, theme, onClose }: CrossPathDetailsProps) {
  return (
    <div style={{
      flex: 1,
      background: theme.background,
      borderRadius: '15px',
      padding: '25px',
      border: `2px solid ${theme.primary}10`,
      maxHeight: '500px',
      overflow: 'auto'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 style={{ margin: 0, color: theme.primary }}>{crossPath.name}</h3>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            fontSize: '20px',
            cursor: 'pointer',
            color: theme.neutral
          }}
        >
          ×
        </button>
      </div>

      {/* Mini Video Player */}
      <div style={{ marginBottom: '20px' }}>
        <MiniVideoPlayer
          videoUrl={crossPath.videoUrl}
          liveStreamUrl={crossPath.liveStreamUrl}
          crossPathId={crossPath.id}
          theme={theme}
          isEnabled={crossPath.isVideoEnabled}
        />
      </div>

      {/* Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '15px',
        marginBottom: '20px'
      }}>
        <div style={{
          padding: '15px',
          background: theme.accent,
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: theme.primary }}>
            {crossPath.vehicles}
          </div>
          <div style={{ fontSize: '14px', color: theme.neutral }}>Active Vehicles</div>
        </div>

        <div style={{
          padding: '15px',
          background: theme.accent,
          borderRadius: '8px',
          textAlign: 'center'
        }}>
          <div style={{
            fontSize: '16px',
            fontWeight: 'bold',
            color: getCongestionColor(crossPath.congestion, theme)
          }}>
            {crossPath.congestion}
          </div>
          <div style={{ fontSize: '14px', color: theme.neutral }}>Congestion</div>
        </div>
      </div>

      <div style={{
        padding: '15px',
        background: `${theme.primary}05`,
        borderRadius: '8px',
        borderLeft: `4px solid ${theme.primary}`
      }}>
        <h4 style={{ margin: '0 0 10px 0', color: theme.primary }}>Current Status</h4>
        <p style={{ margin: 0, fontSize: '14px', color: theme.darkText }}>
          Traffic flowing normally with {crossPath.congestion.toLowerCase()} congestion levels. 
          Signal timing optimized for current conditions.
        </p>
      </div>
    </div>
  );
}