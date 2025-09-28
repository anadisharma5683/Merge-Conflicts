'use client';

import React, { useState, useEffect } from 'react';

// Simple, clean styling
const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    backgroundColor: '#f8f9fa',
    minHeight: '100vh'
  },
  header: {
    textAlign: 'center' as const,
    marginBottom: '40px',
    padding: '20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    border: '1px solid #ddd'
  },
  title: {
    fontSize: '28px',
    fontWeight: 'bold',
    color: '#333',
    marginBottom: '10px'
  },
  subtitle: {
    fontSize: '16px',
    color: '#666'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginBottom: '30px'
  },
  card: {
    backgroundColor: 'white',
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '20px'
  },
  cardTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#333',
    marginBottom: '15px'
  },
  statNumber: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#007bff',
    marginBottom: '5px'
  },
  statLabel: {
    fontSize: '14px',
    color: '#666'
  },
  chart: {
    width: '100%',
    height: '200px',
    backgroundColor: '#f8f9fa',
    border: '1px solid #ddd',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#666'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
    marginTop: '15px'
  },
  th: {
    backgroundColor: '#f8f9fa',
    padding: '12px',
    textAlign: 'left' as const,
    borderBottom: '2px solid #ddd',
    fontWeight: '600',
    color: '#333'
  },
  td: {
    padding: '12px',
    borderBottom: '1px solid #eee'
  },
  statusGood: {
    color: '#28a745',
    fontWeight: '600'
  },
  statusWarning: {
    color: '#ffc107',
    fontWeight: '600'
  },
  statusDanger: {
    color: '#dc3545',
    fontWeight: '600'
  },
  progressBar: {
    width: '100%',
    height: '20px',
    backgroundColor: '#e9ecef',
    borderRadius: '10px',
    overflow: 'hidden',
    marginTop: '10px'
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007bff',
    transition: 'width 0.3s ease'
  }
};

// Traffic data type
interface TrafficArea {
  id: number;
  name: string;
  vehicles: number;
  speed: number;
  congestion: number;
  status: 'Good' | 'Warning' | 'Critical';
}

// Simple Traffic Dashboard Component
export default function SimpleTrafficSystem() {
  const [trafficData, setTrafficData] = useState<TrafficArea[]>([
    { id: 1, name: 'Vijay Nagar', vehicles: 350, speed: 25, congestion: 85, status: 'Critical' },
    { id: 2, name: 'Bhawarkua', vehicles: 280, speed: 35, congestion: 60, status: 'Warning' },
    { id: 3, name: 'Palasia', vehicles: 150, speed: 45, congestion: 35, status: 'Good' },
    { id: 4, name: 'AB Road', vehicles: 120, speed: 55, congestion: 20, status: 'Good' }
  ]);

  const [totalVehicles, setTotalVehicles] = useState(0);
  const [avgSpeed, setAvgSpeed] = useState(0);
  const [avgCongestion, setAvgCongestion] = useState(0);

  // Calculate summary statistics
  useEffect(() => {
    const total = trafficData.reduce((sum, area) => sum + area.vehicles, 0);
    const speedAvg = Math.round(trafficData.reduce((sum, area) => sum + area.speed, 0) / trafficData.length);
    const congestionAvg = Math.round(trafficData.reduce((sum, area) => sum + area.congestion, 0) / trafficData.length);
    
    setTotalVehicles(total);
    setAvgSpeed(speedAvg);
    setAvgCongestion(congestionAvg);
  }, [trafficData]);

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'Good': return styles.statusGood;
      case 'Warning': return styles.statusWarning;
      case 'Critical': return styles.statusDanger;
      default: return {};
    }
  };

  const getCongestionColor = (congestion: number) => {
    if (congestion < 40) return '#28a745';
    if (congestion < 70) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>Traffic Management Dashboard</h1>
        <p style={styles.subtitle}>Real-time traffic monitoring and analysis</p>
      </div>

      {/* Summary Statistics */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Total Vehicles</h3>
          <div style={styles.statNumber}>{totalVehicles}</div>
          <div style={styles.statLabel}>Active on roads</div>
        </div>

        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Average Speed</h3>
          <div style={styles.statNumber}>{avgSpeed} km/h</div>
          <div style={styles.statLabel}>Across all areas</div>
        </div>

        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Average Congestion</h3>
          <div style={styles.statNumber}>{avgCongestion}%</div>
          <div style={styles.statLabel}>Traffic density</div>
        </div>
      </div>

      {/* Traffic Areas Table */}
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Traffic Areas Status</h3>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Area Name</th>
              <th style={styles.th}>Vehicles</th>
              <th style={styles.th}>Speed (km/h)</th>
              <th style={styles.th}>Congestion</th>
              <th style={styles.th}>Status</th>
            </tr>
          </thead>
          <tbody>
            {trafficData.map(area => (
              <tr key={area.id}>
                <td style={styles.td}><strong>{area.name}</strong></td>
                <td style={styles.td}>{area.vehicles}</td>
                <td style={styles.td}>{area.speed}</td>
                <td style={styles.td}>
                  {area.congestion}%
                  <div style={styles.progressBar}>
                    <div 
                      style={{
                        ...styles.progressFill,
                        width: `${area.congestion}%`,
                        backgroundColor: getCongestionColor(area.congestion)
                      }}
                    />
                  </div>
                </td>
                <td style={{...styles.td, ...getStatusStyle(area.status)}}>{area.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts Section */}
      <div style={styles.grid}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Congestion Levels Chart</h3>
          <div style={styles.chart}>
            <div>
              <p>Bar Chart: Congestion by Area</p>
              <p style={{fontSize: '12px', color: '#999', marginTop: '10px'}}>
                Chart visualization would go here
              </p>
            </div>
          </div>
        </div>

        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Speed Analysis</h3>
          <div style={styles.chart}>
            <div>
              <p>Line Chart: Speed Trends</p>
              <p style={{fontSize: '12px', color: '#999', marginTop: '10px'}}>
                Chart visualization would go here
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}