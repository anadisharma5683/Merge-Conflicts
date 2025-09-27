'use client';

import { TrafficStats, Theme } from '@/types/smart-traffic';

interface VehicleStatsProps {
  stats: TrafficStats;
  theme: Theme;
  isConnected?: boolean;
  lastUpdate?: Date | null;
  error?: string | null;
}

export default function VehicleStats({ 
  stats, 
  theme, 
  isConnected = true, 
  lastUpdate = null, 
  error = null 
}: VehicleStatsProps) {
  const vehicleTypes = [
    { type: 'Cars', count: stats.cars, icon: '🚗', color: '#4caf50' },
    { type: 'Trucks', count: stats.trucks, icon: '🚛', color: theme.secondary },
    { type: 'Buses', count: stats.buses, icon: '🚌', color: '#ffa726' },
    { type: 'Motorcycles', count: stats.motorcycles, icon: '🏍️', color: '#2196f3' }
  ];

  return (
    <div>
      {/* Connection Status */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px',
        padding: '10px 15px',
        background: isConnected ? '#4caf5010' : '#f4433620',
        borderRadius: '8px',
        border: `1px solid ${isConnected ? '#4caf50' : '#f44336'}20`
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: isConnected ? '#4caf50' : '#f44336'
          }} />
          <span style={{
            fontSize: '14px',
            color: isConnected ? '#4caf50' : '#f44336',
            fontWeight: 'bold'
          }}>
            {isConnected ? 'Backend Connected' : 'Backend Disconnected'}
          </span>
        </div>
        {lastUpdate && (
          <span style={{
            fontSize: '12px',
            color: theme.neutral
          }}>
            Last updated: {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </div>

      {error && (
        <div style={{
          padding: '10px 15px',
          background: '#f4433610',
          borderRadius: '8px',
          border: '1px solid #f4433620',
          marginBottom: '15px',
          color: '#f44336',
          fontSize: '14px'
        }}>
          Error: {error}
        </div>
      )}

      {/* Vehicle Count Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '20px'
      }}>
        {vehicleTypes.map(stat => (
          <div key={stat.type} style={{
            background: theme.accent,
            borderRadius: '10px',
            padding: '20px',
            textAlign: 'center',
            border: `2px solid ${stat.color}20`
          }}>
            <div style={{ fontSize: '30px', marginBottom: '10px' }}>{stat.icon}</div>
            <div style={{
              fontSize: '28px',
              fontWeight: 'bold',
              color: stat.color,
              marginBottom: '5px',
              fontFamily: 'monospace'
            }}>
              {stat.count.toLocaleString()}
            </div>
            <div style={{ fontSize: '14px', color: theme.neutral }}>{stat.type}</div>
          </div>
        ))}
      </div>

      {/* Total Count */}
      <div style={{
        marginTop: '20px',
        padding: '20px',
        background: `${theme.primary}10`,
        borderRadius: '10px',
        border: `2px solid ${theme.primary}20`,
        textAlign: 'center'
      }}>
        <div style={{
          fontSize: '18px',
          fontWeight: 'bold',
          color: theme.primary,
          marginBottom: '5px'
        }}>
          Total Vehicles Detected
        </div>
        <div style={{
          fontSize: '36px',
          fontWeight: 'bold',
          color: theme.primary,
          fontFamily: 'monospace'
        }}>
          {stats.total.toLocaleString()}
        </div>
      </div>
    </div>
  );
}