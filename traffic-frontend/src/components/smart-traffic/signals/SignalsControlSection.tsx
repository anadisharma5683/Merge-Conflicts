import React from 'react';
import { Settings } from 'lucide-react';
import { TrafficSignals, Theme, OverrideLog, LocationSignalData } from '@/types/smart-traffic';
import LocationSelector from './LocationSelector';
import LocationSignalDetails from './LocationSignalDetails';

interface SignalsControlSectionProps {
  trafficSignals: TrafficSignals;
  overrideMode: boolean;
  overrideLogs: OverrideLog[];
  locationSignalData: LocationSignalData[];
  selectedSignalLocation: LocationSignalData | null;
  theme: Theme;
  onOverrideModeToggle: () => void;
  onSignalOverride: (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => void;
  onLocationSelect: (location: LocationSignalData) => void;
  onLocationSignalOverride: (locationId: number, direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => void;
}

export default function SignalsControlSection({
  trafficSignals,
  overrideMode,
  overrideLogs,
  locationSignalData,
  selectedSignalLocation,
  theme,
  onOverrideModeToggle,
  onSignalOverride,
  onLocationSelect,
  onLocationSignalOverride
}: SignalsControlSectionProps) {
  return (
    <div>
      {/* Header Section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
        <div>
          <h2 style={{ color: theme.primary, margin: '0 0 8px 0', fontSize: '24px', fontWeight: 'bold' }}>
            🚦 Traffic Signal Control Center
          </h2>
          <p style={{ color: theme.neutral, margin: 0, fontSize: '14px' }}>
            Monitor and control traffic signals at major intersections in Bhubaneswar
          </p>
        </div>
        
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
              padding: '12px 20px',
              background: overrideMode ? theme.secondary : theme.primary,
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '14px',
              fontWeight: 'bold',
              transition: 'all 0.3s ease'
            }}
          >
            <Settings size={16} />
            {overrideMode ? 'Exit Manual Control' : 'Enable Manual Override'}
          </button>
        </div>
      </div>

      {/* Location Selector */}
      <LocationSelector
        locations={locationSignalData}
        selectedLocation={selectedSignalLocation}
        theme={theme}
        onLocationSelect={onLocationSelect}
      />

      {/* Selected Location Signal Details */}
      {selectedSignalLocation && (
        <LocationSignalDetails
          location={selectedSignalLocation}
          overrideMode={overrideMode}
          theme={theme}
          onSignalOverride={(direction, newState) => 
            onLocationSignalOverride(selectedSignalLocation.id, direction, newState)
          }
        />
      )}

      {/* Synchronization Status */}
      <div style={{
        background: theme.background,
        borderRadius: '15px',
        padding: '20px',
        border: `2px solid ${theme.primary}10`,
        marginBottom: '20px'
      }}>
        <h3 style={{ color: theme.primary, marginBottom: '15px', fontSize: '22px', fontWeight: 'bold' }}>
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
            <div style={{ fontSize: '18px', color: '#4caf50', fontWeight: 'bold' }}>
              ✅ One Green at a Time
            </div>
            <div style={{ fontSize: '16px', color: theme.darkText, marginTop: '5px' }}>
              Only one direction has green light
            </div>
          </div>
          <div style={{
            padding: '15px',
            background: `${theme.primary}10`,
            borderRadius: '8px',
            borderLeft: `4px solid ${theme.primary}`
          }}>
            <div style={{ fontSize: '18px', color: theme.primary, fontWeight: 'bold' }}>
              ⏱️ 60-Second Cycle
            </div>
            <div style={{ fontSize: '16px', color: theme.darkText, marginTop: '5px' }}>
              15 seconds green per direction
            </div>
          </div>
          <div style={{
            padding: '15px',
            background: '#ffa72610',
            borderRadius: '8px',
            borderLeft: '4px solid #ffa726'
          }}>
            <div style={{ fontSize: '18px', color: '#ffa726', fontWeight: 'bold' }}>
              🔄 Auto Cycling
            </div>
            <div style={{ fontSize: '16px', color: theme.darkText, marginTop: '5px' }}>
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