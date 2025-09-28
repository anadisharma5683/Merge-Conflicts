'use client';

import { CrossPath, Theme } from '@/types/smart-traffic';
import { getCongestionColor } from '@/lib/smart-traffic-theme';

interface InteractiveMapProps {
  crossPaths: CrossPath[];
  theme: Theme;
  onCrossPathSelect: (path: CrossPath) => void;
  backgroundImage?: string;
  showOverlay?: boolean;
  overlayOpacity?: number;
}

export default function InteractiveMap({ 
  crossPaths, 
  theme, 
  onCrossPathSelect, 
  backgroundImage,
  showOverlay = true,
  overlayOpacity = 0.3
}: InteractiveMapProps) {
  return (
    <div style={{
      flex: 2,
      background: theme.background,
      borderRadius: '15px',
      padding: '20px',
      position: 'relative',
      minHeight: '500px',
      border: `2px solid ${theme.primary}10`
    }}>
      {/* Mock City Map */}
      <div style={{
        width: '100%',
        height: '500px',
        background: backgroundImage 
          ? `url(${backgroundImage})` 
          : `linear-gradient(45deg, ${theme.accent} 0%, #e8f4f8 100%)`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        borderRadius: '10px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Background Overlay for better visibility */}
        {backgroundImage && showOverlay && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `rgba(${theme.primary === '#1976d2' ? '25, 118, 210' : '0, 0, 0'}, ${overlayOpacity})`,
            pointerEvents: 'none'
          }} />
        )}
        {/* Map Grid Lines */}
        <svg width="100%" height="100%" style={{ position: 'absolute' }}>
          {Array.from({ length: 10 }, (_, i) => (
            <g key={i}>
              <line 
                x1={`${(i + 1) * 10}%`} 
                y1="0" 
                x2={`${(i + 1) * 10}%`} 
                y2="100%" 
                stroke={`${theme.primary}20`} 
                strokeWidth="1" 
              />
              <line 
                x1="0" 
                y1={`${(i + 1) * 10}%`} 
                x2="100%" 
                y2={`${(i + 1) * 10}%`} 
                stroke={`${theme.primary}20`} 
                strokeWidth="1" 
              />
            </g>
          ))}
        </svg>

        {/* Cross Path Markers */}
        {crossPaths.map(path => (
          <button
            key={path.id}
            onClick={() => onCrossPathSelect(path)}
            style={{
              position: 'absolute',
              left: `${path.x}%`,
              top: `${path.y}%`,
              transform: 'translate(-50%, -50%)',
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              border: 'none',
              background: getCongestionColor(path.congestion, theme),
              color: 'white',
              cursor: 'pointer',
              fontSize: '18px',
              boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => {
              (e.target as HTMLElement).style.transform = 'translate(-50%, -50%) scale(1.1)';
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLElement).style.transform = 'translate(-50%, -50%) scale(1)';
            }}
          >
            🚦
          </button>
        ))}
      </div>

      {/* Map Legend */}
      <div style={{
        position: 'absolute',
        bottom: '30px',
        left: '30px',
        background: 'rgba(255,255,255,0.95)',
        padding: '15px',
        borderRadius: '10px',
        boxShadow: '0 5px 20px rgba(0,0,0,0.1)'
      }}>
        <h4 style={{ margin: '0 0 10px 0', color: theme.primary }}>Congestion Levels</h4>
        {[
          { color: '#4caf50', label: 'Low' },
          { color: '#ffa726', label: 'Medium' },
          { color: theme.secondary, label: 'High' }
        ].map(item => (
          <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' }}>
            <div style={{
              width: '15px',
              height: '15px',
              borderRadius: '50%',
              background: item.color
            }} />
            <span style={{ fontSize: '14px', color: theme.darkText }}>{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}