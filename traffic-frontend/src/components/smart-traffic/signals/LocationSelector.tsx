'use client';

import { MapPin } from 'lucide-react';
import { LocationSignalData, Theme } from '@/types/smart-traffic';

interface LocationSelectorProps {
  locations: LocationSignalData[];
  selectedLocation: LocationSignalData | null;
  theme: Theme;
  onLocationSelect: (location: LocationSignalData) => void;
}

export default function LocationSelector({
  locations,
  selectedLocation,
  theme,
  onLocationSelect
}: LocationSelectorProps) {
  return (
    <div style={{
      background: theme.background,
      borderRadius: '15px',
      padding: '20px',
      border: `2px solid ${theme.primary}10`,
      marginBottom: '20px'
    }}>
      <h3 style={{ 
        color: theme.primary, 
        marginBottom: '15px',
        fontSize: '18px',
        fontWeight: 'bold'
      }}>
        📍 Select Traffic Signal Location
      </h3>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
        gap: '15px'
      }}>
        {locations.map(location => (
          <button
            key={location.id}
            onClick={() => onLocationSelect(location)}
            style={{
              padding: '15px',
              background: selectedLocation?.id === location.id 
                ? `${theme.primary}15` 
                : theme.accent,
              border: selectedLocation?.id === location.id 
                ? `2px solid ${theme.primary}` 
                : '1px solid rgba(0,0,0,0.1)',
              borderRadius: '10px',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '12px'
            }}
            onMouseEnter={(e) => {
              if (selectedLocation?.id !== location.id) {
                (e.target as HTMLElement).style.background = `${theme.primary}08`;
              }
            }}
            onMouseLeave={(e) => {
              if (selectedLocation?.id !== location.id) {
                (e.target as HTMLElement).style.background = theme.accent;
              }
            }}
          >
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '50%',
              background: selectedLocation?.id === location.id 
                ? theme.primary 
                : theme.neutral,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white'
            }}>
              <MapPin size={20} />
            </div>
            
            <div style={{ flex: 1 }}>
              <div style={{
                fontWeight: 'bold',
                color: selectedLocation?.id === location.id 
                  ? theme.primary 
                  : theme.darkText,
                marginBottom: '4px'
              }}>
                {location.name}
              </div>
              
              <div style={{
                fontSize: '12px',
                color: theme.neutral,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span>📍 ID: {location.id}</span>
                {location.isActive && (
                  <span style={{
                    background: '#4caf50',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '10px',
                    fontSize: '10px',
                    fontWeight: 'bold'
                  }}>
                    ACTIVE
                  </span>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>
      
      {!selectedLocation && (
        <div style={{
          marginTop: '15px',
          padding: '12px',
          background: `${theme.neutral}10`,
          borderRadius: '8px',
          textAlign: 'center',
          color: theme.neutral,
          fontSize: '14px'
        }}>
          👆 Select a location above to view its traffic signal status
        </div>
      )}
    </div>
  );
}