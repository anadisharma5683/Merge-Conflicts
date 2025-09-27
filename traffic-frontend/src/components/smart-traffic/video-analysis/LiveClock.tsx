'use client';

import React, { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';
import { Theme } from '@/types/smart-traffic';

interface LiveClockProps {
  theme: Theme;
}

export default function LiveClock({ theme }: LiveClockProps) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: true,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div style={{
      position: 'absolute',
      top: '20px',
      right: '20px',
      background: 'rgba(0, 0, 0, 0.85)',
      color: 'white',
      padding: '15px 20px',
      borderRadius: '12px',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(76, 175, 80, 0.3)',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4)',
      minWidth: '200px',
      zIndex: 10
    }}>
      {/* Live Status Indicator */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '5px'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: '#4caf50',
          animation: 'pulse 2s infinite'
        }} />
        <span style={{
          fontSize: '12px',
          color: '#4caf50',
          fontWeight: '600',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          LIVE FEED
        </span>
      </div>
      
      {/* Time Display */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <Clock size={18} color="#4caf50" />
        <div>
          <div style={{
            fontSize: '16px',
            fontWeight: 'bold',
            color: '#ffffff',
            letterSpacing: '0.5px',
            fontFamily: 'monospace'
          }}>
            {formatTime(currentTime)}
          </div>
          <div style={{
            fontSize: '11px',
            color: 'rgba(255, 255, 255, 0.7)',
            marginTop: '2px'
          }}>
            {formatDate(currentTime)}
          </div>
        </div>
      </div>
      
      {/* Location Info */}
      <div style={{
        fontSize: '10px',
        color: 'rgba(255, 255, 255, 0.6)',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        paddingTop: '6px',
        textAlign: 'center'
      }}>
        Bhubaneswar Traffic Control
      </div>
      
      <style jsx>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.5; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}