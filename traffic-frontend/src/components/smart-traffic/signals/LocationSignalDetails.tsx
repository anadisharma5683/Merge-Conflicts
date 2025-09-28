'use client';

import { Clock, Settings, MapPin } from 'lucide-react';
import { LocationSignalData, Theme, TrafficSignals } from '@/types/smart-traffic';
import { useState } from 'react';

interface LocationSignalDetailsProps {
  location: LocationSignalData;
  overrideMode: boolean;
  theme: Theme;
  onSignalOverride: (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => void;
}

export default function LocationSignalDetails({
  location,
  overrideMode,
  theme,
  onSignalOverride
}: LocationSignalDetailsProps) {
  const [lastAction, setLastAction] = useState<string>('');

  const handleOverride = (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => {
    // Call the original override function
    onSignalOverride(direction, newState);
    
    // Set visual feedback
    setLastAction(`${direction.toUpperCase()} set to ${newState.toUpperCase()}`);
    
    // Clear feedback after 3 seconds
    setTimeout(() => setLastAction(''), 3000);
  };
  return (
    <div style={{
      background: theme.background,
      borderRadius: '15px',
      padding: '25px',
      border: `2px solid ${theme.primary}10`,
      marginBottom: '20px'
    }}>
      {/* Location Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '15px',
        marginBottom: '25px',
        padding: '15px',
        background: `${theme.primary}08`,
        borderRadius: '10px',
        border: `1px solid ${theme.primary}20`
      }}>
        <div style={{
          width: '50px',
          height: '50px',
          borderRadius: '50%',
          background: theme.primary,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'white'
        }}>
          <MapPin size={24} />
        </div>
        
        <div style={{ flex: 1 }}>
          <h3 style={{ 
            margin: 0, 
            color: theme.primary,
            fontSize: '20px',
            fontWeight: 'bold'
          }}>
            {location.name}
          </h3>
          <div style={{
            fontSize: '14px',
            color: theme.neutral,
            marginTop: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <span>🆔 Location ID: {location.id}</span>
            {location.isActive && (
              <span style={{
                background: '#4caf50',
                color: 'white',
                padding: '4px 8px',
                borderRadius: '12px',
                fontSize: '12px',
                fontWeight: 'bold'
              }}>
                🟢 ACTIVE
              </span>
            )}
          </div>
        </div>
        
        {location.lastUpdated && (
          <div style={{
            fontSize: '12px',
            color: theme.neutral,
            textAlign: 'right'
          }}>
            <div>Last Updated</div>
            <div style={{ fontWeight: 'bold' }}>{location.lastUpdated}</div>
          </div>
        )}
      </div>

      {/* Traffic Signals Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '20px'
      }}>
        {Object.entries(location.signals).map(([direction, signal]) => (
          <div key={direction} style={{
            background: theme.accent,
            borderRadius: '12px',
            padding: '20px',
            border: `2px solid ${theme.primary}10`,
            textAlign: 'center',
            transition: 'transform 0.3s ease',
            boxShadow: '0 2px 10px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{ 
              color: theme.primary, 
              marginBottom: '15px',
              textTransform: 'capitalize',
              fontSize: '16px',
              fontWeight: 'bold'
            }}>
              {direction} Lane
            </h4>

            {/* Traffic Light Visual */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '15px'
            }}>
              {['red', 'yellow', 'green'].map(color => (
                <div key={color} style={{
                  width: '35px',
                  height: '35px',
                  borderRadius: '50%',
                  background: signal.state === color ? 
                    (color === 'red' ? '#f44336' : color === 'yellow' ? '#ffc107' : '#4caf50') :
                    '#e0e0e0',
                  boxShadow: signal.state === color ? 
                    `0 0 20px ${color === 'red' ? '#f44336' : color === 'yellow' ? '#ffc107' : '#4caf50'}` : 
                    'none',
                  transition: 'all 0.3s ease',
                  border: '2px solid #ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {signal.state === color && (
                    <div style={{
                      width: '15px',
                      height: '15px',
                      borderRadius: '50%',
                      background: 'rgba(255,255,255,0.9)'
                    }} />
                  )}
                </div>
              ))}
            </div>

            {/* Current State Display */}
            <div style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: signal.state === 'red' ? '#f44336' : 
                     signal.state === 'yellow' ? '#ffc107' : '#4caf50',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              {signal.state}
            </div>

            {/* Countdown Timer */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              marginBottom: '15px',
              padding: '8px',
              background: 'rgba(255,255,255,0.7)',
              borderRadius: '6px'
            }}>
              <Clock size={16} color={theme.primary} />
              <span style={{ 
                color: theme.darkText, 
                fontSize: '16px',
                fontWeight: 'bold'
              }}>
                {signal.countdown}s
              </span>
            </div>

            {/* Manual Override Controls */}
            {overrideMode && (
              <div style={{
                display: 'flex',
                gap: '6px',
                justifyContent: 'center',
                marginTop: '12px'
              }}>
                {(['red', 'yellow', 'green'] as const).map(color => (
                  <button
                    key={color}
                    onClick={() => {
                      console.log(`Button clicked: ${color} for ${direction} at location ${location.id}`);
                      onSignalOverride(direction as keyof TrafficSignals, color);
                    }}
                    style={{
                      padding: '8px 12px',
                      background: color === 'red' ? '#f44336' : 
                                 color === 'yellow' ? '#ffc107' : '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '5px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                      transition: 'transform 0.2s ease',
                      boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                    }}
                    onMouseEnter={(e) => {
                      (e.target as HTMLElement).style.transform = 'scale(1.05)';
                    }}
                    onMouseLeave={(e) => {
                      (e.target as HTMLElement).style.transform = 'scale(1)';
                    }}
                  >
                    {color}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Status Info */}
      <div style={{
        marginTop: '20px',
        padding: '15px',
        background: `${theme.primary}05`,
        borderRadius: '8px',
        borderLeft: `4px solid ${theme.primary}`
      }}>
        <h4 style={{ margin: '0 0 8px 0', color: theme.primary, fontSize: '14px' }}>
          🚦 Signal Status Information
        </h4>
        <p style={{ margin: 0, fontSize: '13px', color: theme.darkText }}>
          Traffic signals are operating in automatic mode with 60-second cycles. 
          Each direction gets 15 seconds of green time in sequence: North → East → South → West.
        </p>
      </div>
    </div>
  );
}