'use client';

import { VehicleCounts, VehicleDisplayItem } from '@/types/traffic';

interface VehicleCountsDisplayProps {
  counts: VehicleCounts;
}

export default function VehicleCountsDisplay({ counts }: VehicleCountsDisplayProps) {
  const vehicleItems: VehicleDisplayItem[] = [
    { label: 'Total', count: counts.total, emoji: '🚗', color: '#ffa726' },
    { label: 'Cars', count: counts.car, emoji: '🚙', color: '#4caf50' },
    { label: 'Motorcycles', count: counts.motorcycle, emoji: '🏍️', color: '#2196f3' },
    { label: 'Buses', count: counts.bus, emoji: '🚌', color: '#ff9800' },
    { label: 'Trucks', count: counts.truck, emoji: '🚛', color: '#f44336' }
  ];

  return (
    <div style={{ marginTop: '30px' }}>
      <h3 style={{ 
        marginBottom: '20px',
        textAlign: 'center',
        fontSize: '1.8em'
      }}>
        🚙 Vehicle Count Statistics
      </h3>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '20px'
      }}>
        {vehicleItems.map((item, index) => (
          <div
            key={index}
            style={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              transition: 'transform 0.3s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-5px)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <div style={{ fontSize: '2em', marginBottom: '10px' }}>
              {item.emoji}
            </div>
            <div style={{
              fontSize: '2.5em',
              fontWeight: 'bold',
              color: item.color,
              marginBottom: '5px'
            }}>
              {item.count.toLocaleString()}
            </div>
            <div style={{
              fontSize: '1em',
              textTransform: 'capitalize',
              color: '#ccc'
            }}>
              {item.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}