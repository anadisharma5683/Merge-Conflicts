import { Theme } from '@/types/smart-traffic';

export const smartTrafficTheme: Theme = {
  primary: '#12355b',
  secondary: '#c16e70',
  darkText: '#3d1308',
  neutral: '#4a442d',
  background: '#ffffff',
  accent: '#f8f9fa'
};

// Map Configuration
export const mapConfig = {
  backgroundImage: '/images/traffic-map-bg.jpg',
  showOverlay: true,
  overlayOpacity: 0.4,
  fallbackGradient: true
};

export const getSeverityColor = (severity: 'Low' | 'Medium' | 'High', theme: Theme): string => {
  switch (severity) {
    case 'High':
      return theme.secondary;
    case 'Medium':
      return '#ffa726';
    case 'Low':
      return '#4caf50';
    default:
      return theme.neutral;
  }
};

export const getCongestionColor = (congestion: 'Low' | 'Medium' | 'High', theme: Theme): string => {
  switch (congestion) {
    case 'High':
      return theme.secondary;
    case 'Medium':
      return '#ffa726';
    case 'Low':
      return '#4caf50';
    default:
      return theme.neutral;
  }
};