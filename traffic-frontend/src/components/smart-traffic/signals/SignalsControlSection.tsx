import React from 'react';
import { Clock, Settings } from 'lucide-react';
import { TrafficSignals, Theme, OverrideLog } from '@/types/smart-traffic';

interface SignalsControlSectionProps {
  trafficSignals: TrafficSignals;
  overrideMode: boolean;
  overrideLogs: OverrideLog[];
  theme: Theme;
  onOverrideModeToggle: () => void;
  onSignalOverride: (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => void;
}

export default function SignalsControlSection({
  trafficSignals,
  overrideMode,
  overrideLogs,
  theme,
  onOverrideModeToggle,
  onSignalOverride
}: SignalsControlSectionProps) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ color: theme.primary, margin: 0 }}>4-Way Traffic Signal Control</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{
            padding: '8px 15px',
            background: '#4caf5010',
            borderRadius: '20px',
            color: '#4caf50',
            fontSize: '14px',
            fontWeight: 'bold'
          }}>
            🔄 60s Cycle • 15s Green Each
          </div>
          <button
            onClick={onOverrideModeToggle}
            style={{
              padding: '10px 20px',
              background: overrideMode ? theme.secondary : theme.primary,
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <Settings size={16} />
            {overrideMode ? 'Exit Manual' : 'Manual Override'}
          </button>
        </div>
      </div>

      {/* 4-Way Traffic Signals Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '25px',
        marginBottom: '30px'
      }}>
        {Object.entries(trafficSignals).map(([direction, signal]) => (
          <div key={direction} style={{
            background: theme.background,
            borderRadius: '15px',
            padding: '25px',
            border: `2px solid ${theme.primary}10`,
            textAlign: 'center',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
          }}>
            <h3 style={{ 
              color: theme.primary, 
              marginBottom: '20px',
              textTransform: 'capitalize',
              fontSize: '18px',
              fontWeight: 'bold'
            }}>
              {direction} Lane
            </h3>

            {/* Traffic Light Visual */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '10px',
              marginBottom: '20px'
            }}>
              {['red', 'yellow', 'green'].map(color => (
                <div key={color} style={{
                  width: '45px',
                  height: '45px',
                  borderRadius: '50%',
                  background: signal.state === color ? 
                    (color === 'red' ? '#f44336' : color === 'yellow' ? '#ffc107' : '#4caf50') :
                    '#e0e0e0',
                  boxShadow: signal.state === color ? 
                    `0 0 25px ${color === 'red' ? '#f44336' : color === 'yellow' ? '#ffc107' : '#4caf50'}` : 
                    'none',
                  transition: 'all 0.3s ease',
                  border: '3px solid #ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  {signal.state === color && (
                    <div style={{
                      width: '20px',
                      height: '20px',
                      borderRadius: '50%',
                      background: 'rgba(255,255,255,0.8)'
                    }} />
                  )}
                </div>
              ))}
            </div>

            {/* Current State Display */}
            <div style={{
              fontSize: '20px',
              fontWeight: 'bold',
              color: signal.state === 'red' ? '#f44336' : 
                     signal.state === 'yellow' ? '#ffc107' : '#4caf50',
              marginBottom: '15px',
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
              gap: '8px',
              marginBottom: '20px',
              padding: '10px',
              background: theme.accent,
              borderRadius: '8px'
            }}>
              <Clock size={18} color={theme.primary} />
              <span style={{ 
                color: theme.darkText, 
                fontSize: '18px',
                fontWeight: 'bold'
              }}>
                {signal.countdown}s
              </span>
            </div>

            {/* Manual Override Controls */}
            {overrideMode && (
              <div style={{
                display: 'flex',
                gap: '8px',
                justifyContent: 'center',
                marginTop: '15px'
              }}>
                {(['red', 'yellow', 'green'] as const).map(color => (
                  <button
                    key={color}
                    onClick={() => onSignalOverride(direction as keyof TrafficSignals, color)}
                    style={{
                      padding: '10px 15px',
                      background: color === 'red' ? '#f44336' : 
                                 color === 'yellow' ? '#ffc107' : '#4caf50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: 'bold',
                      textTransform: 'uppercase',
                      transition: 'transform 0.2s ease',
                      boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'scale(1.05)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'scale(1)';
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

      {/* Synchronization Status */}
      <div style={{
        background: theme.background,
        borderRadius: '15px',
        padding: '20px',
        border: `2px solid ${theme.primary}10`,
        marginBottom: '20px'
      }}>
        <h3 style={{ color: theme.primary, marginBottom: '15px' }}>
          🚦 4-Way Intersection Status
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px'
        }}>
          <div style={{
            padding: '15px',
            background: '#4caf5010',
            borderRadius: '8px',
            borderLeft: '4px solid #4caf50'
          }}>
            <div style={{ fontSize: '14px', color: '#4caf50', fontWeight: 'bold' }}>
              ✅ One Green at a Time
            </div>
            <div style={{ fontSize: '12px', color: theme.darkText, marginTop: '5px' }}>
              Only one direction has green light
            </div>
          </div>
          <div style={{
            padding: '15px',
            background: `${theme.primary}10`,
            borderRadius: '8px',
            borderLeft: `4px solid ${theme.primary}`
          }}>
            <div style={{ fontSize: '14px', color: theme.primary, fontWeight: 'bold' }}>
              ⏱️ 60-Second Cycle
            </div>
            <div style={{ fontSize: '12px', color: theme.darkText, marginTop: '5px' }}>
              15 seconds green per direction
            </div>
          </div>
          <div style={{
            padding: '15px',
            background: '#ffa72610',
            borderRadius: '8px',
            borderLeft: '4px solid #ffa726'
          }}>
            <div style={{ fontSize: '14px', color: '#ffa726', fontWeight: 'bold' }}>
              🔄 Auto Cycling
            </div>
            <div style={{ fontSize: '12px', color: theme.darkText, marginTop: '5px' }}>
              N → E → S → W → Repeat
            </div>
          </div>
        </div>
      </div>

      {/* Override History */}
      {overrideLogs.length > 0 && (
        <div style={{
          background: theme.background,
          borderRadius: '15px',
          padding: '25px',
          border: `2px solid ${theme.primary}10`
        }}>
          <h3 style={{ color: theme.primary, marginBottom: '20px' }}>
            📋 Manual Override History
          </h3>
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {overrideLogs.map(log => (
              <div key={log.id} style={{
                padding: '12px',
                background: theme.accent,
                borderRadius: '8px',
                marginBottom: '10px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                border: '1px solid rgba(0,0,0,0.05)'
              }}>
                <div>
                  <span style={{ 
                    fontWeight: 'bold', 
                    color: theme.primary,
                    textTransform: 'uppercase'
                  }}>
                    {log.direction}
                  </span>
                  <span style={{ margin: '0 10px', color: theme.darkText }}>→</span>
                  <span style={{
                    color: log.state === 'red' ? '#f44336' : 
                           log.state === 'yellow' ? '#ffc107' : '#4caf50',
                    fontWeight: 'bold',
                    textTransform: 'uppercase'
                  }}>
                    {log.state}
                  </span>
                </div>
                <div style={{ 
                  fontSize: '12px', 
                  color: theme.neutral,
                  textAlign: 'right'
                }}>
                  <div>{log.time}</div>
                  <div style={{ fontStyle: 'italic' }}>by {log.user}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}