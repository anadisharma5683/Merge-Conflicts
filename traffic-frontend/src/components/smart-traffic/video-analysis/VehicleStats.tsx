'use client';

import { TrafficStats, Theme } from '@/types/smart-traffic';

interface VehicleStatsProps {
  stats: TrafficStats;
  theme: Theme;
}

export default function VehicleStats({ stats, theme }: VehicleStatsProps) {
  const vehicleTypes = [
    { type: 'Cars', count: stats.cars, icon: '🚗', color: '#4caf50' },
    { type: 'Trucks', count: stats.trucks, icon: '🚛', color: theme.secondary },
    { type: 'Buses', count: stats.buses, icon: '🚌', color: '#ffa726' },
    { type: 'Motorcycles', count: stats.motorcycles, icon: '🏍️', color: '#2196f3' }
  ];

  return (
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
            fontSize: '24px',
            fontWeight: 'bold',
            color: stat.color,
            marginBottom: '5px'
          }}>
            {stat.count}
          </div>
          <div style={{ fontSize: '14px', color: theme.neutral }}>{stat.type}</div>
        </div>
      ))}
    </div>
  );
}