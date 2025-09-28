'use client';

import { useState, useEffect, useCallback } from 'react';
import { 
  AreaCongestion, 
  Route, 
  AITrafficPrediction, 
  CongestionAnalytics,
  CongestionSettings,
  CongestionPrediction
} from '@/types/smart-traffic';

export const useCongestionMonitoring = () => {
  // State management
  const [areas, setAreas] = useState<AreaCongestion[]>([]);
  const [selectedArea, setSelectedArea] = useState<AreaCongestion | null>(null);
  const [routes, setRoutes] = useState<Route[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [predictions, setPredictions] = useState<AITrafficPrediction[]>([]);
  const [analytics, setAnalytics] = useState<CongestionAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Settings
  const [settings, setSettings] = useState<CongestionSettings>({
    refreshInterval: 30,
    predictionHours: 24,
    alertThresholds: {
      medium: 40,
      high: 70,
      critical: 90
    },
    routePreferences: {
      prioritizeFastest: true,
      avoidHighTraffic: true,
      considerWeather: false
    }
  });

  // Sample area data - in real app this would come from backend
  const initializeAreas = useCallback(() => {
    const sampleAreas: AreaCongestion[] = [
      {
        id: 1,
        name: 'Rajmahal Square',
        congestionLevel: 75,
        congestionStatus: 'High',
        vehicleCount: 145,
        averageSpeed: 12,
        waitTime: 8.5,
        coordinates: { lat: 20.2961, lng: 85.8245 },
        lastUpdated: new Date(),
        trend: 'increasing'
      },
      {
        id: 2,
        name: 'Kalpana Square',
        congestionLevel: 45,
        congestionStatus: 'Medium',
        vehicleCount: 87,
        averageSpeed: 25,
        waitTime: 4.2,
        coordinates: { lat: 20.2921, lng: 85.8312 },
        lastUpdated: new Date(),
        trend: 'stable'
      },
      {
        id: 3,
        name: 'Shastri Nagar Square',
        congestionLevel: 25,
        congestionStatus: 'Low',
        vehicleCount: 52,
        averageSpeed: 35,
        waitTime: 2.1,
        coordinates: { lat: 20.2882, lng: 85.8189 },
        lastUpdated: new Date(),
        trend: 'decreasing'
      },
      {
        id: 4,
        name: 'Acharya Vihar Square',
        congestionLevel: 85,
        congestionStatus: 'Critical',
        vehicleCount: 198,
        averageSpeed: 8,
        waitTime: 12.3,
        coordinates: { lat: 20.3015, lng: 85.8398 },
        lastUpdated: new Date(),
        trend: 'increasing'
      },
      {
        id: 5,
        name: 'Maharishi College Square',
        congestionLevel: 55,
        congestionStatus: 'Medium',
        vehicleCount: 103,
        averageSpeed: 20,
        waitTime: 5.8,
        coordinates: { lat: 20.2845, lng: 85.8156 },
        lastUpdated: new Date(),
        trend: 'stable'
      }
    ];
    setAreas(sampleAreas);
  }, []);

  // Initialize sample routes
  const initializeRoutes = useCallback(() => {
    const sampleRoutes: Route[] = [
      {
        id: 'route-1',
        name: 'Route A: Via Rajmahal Square',
        origin: 'Patia',
        destination: 'Bhubaneswar Station',
        segments: [
          {
            id: 'seg-1',
            name: 'Patia to Rajmahal Square',
            distance: 5.2,
            estimatedTime: 18,
            congestionLevel: 75,
            coordinates: [
              { lat: 20.3498, lng: 85.8050 },
              { lat: 20.2961, lng: 85.8245 }
            ]
          },
          {
            id: 'seg-2',
            name: 'Rajmahal Square to Station',
            distance: 8.1,
            estimatedTime: 25,
            congestionLevel: 60,
            coordinates: [
              { lat: 20.2961, lng: 85.8245 },
              { lat: 20.2618, lng: 85.8352 }
            ]
          }
        ],
        totalDistance: 13.3,
        totalTime: 43,
        averageCongestion: 67,
        isOptimal: false
      },
      {
        id: 'route-2',
        name: 'Route B: Via Shastri Nagar (Recommended)',
        origin: 'Patia',
        destination: 'Bhubaneswar Station',
        segments: [
          {
            id: 'seg-3',
            name: 'Patia to Shastri Nagar',
            distance: 6.8,
            estimatedTime: 15,
            congestionLevel: 25,
            coordinates: [
              { lat: 20.3498, lng: 85.8050 },
              { lat: 20.2882, lng: 85.8189 }
            ]
          },
          {
            id: 'seg-4',
            name: 'Shastri Nagar to Station',
            distance: 7.5,
            estimatedTime: 18,
            congestionLevel: 35,
            coordinates: [
              { lat: 20.2882, lng: 85.8189 },
              { lat: 20.2618, lng: 85.8352 }
            ]
          }
        ],
        totalDistance: 14.3,
        totalTime: 33,
        averageCongestion: 30,
        isOptimal: true
      },
      {
        id: 'route-3',
        name: 'Route C: Via Kalpana Square',
        origin: 'Patia',
        destination: 'Bhubaneswar Station',
        segments: [
          {
            id: 'seg-5',
            name: 'Patia to Kalpana Square',
            distance: 7.2,
            estimatedTime: 20,
            congestionLevel: 45,
            coordinates: [
              { lat: 20.3498, lng: 85.8050 },
              { lat: 20.2921, lng: 85.8312 }
            ]
          },
          {
            id: 'seg-6',
            name: 'Kalpana Square to Station',
            distance: 6.8,
            estimatedTime: 16,
            congestionLevel: 40,
            coordinates: [
              { lat: 20.2921, lng: 85.8312 },
              { lat: 20.2618, lng: 85.8352 }
            ]
          }
        ],
        totalDistance: 14.0,
        totalTime: 36,
        averageCongestion: 42,
        isOptimal: false
      }
    ];
    setRoutes(sampleRoutes);
    setSelectedRoute(sampleRoutes[1]); // Set optimal route as default
  }, []);

  // Initialize AI predictions
  const initializePredictions = useCallback(() => {
    const samplePredictions: AITrafficPrediction[] = areas.map(area => {
      const hourlyPredictions: CongestionPrediction[] = [];
      const currentHour = new Date().getHours();
      
      for (let i = 1; i <= 24; i++) {
        const hour = (currentHour + i) % 24;
        let baseCongestion = area.congestionLevel;
        
        // Simulate rush hour patterns
        if (hour >= 7 && hour <= 10) baseCongestion += 20;
        else if (hour >= 17 && hour <= 20) baseCongestion += 25;
        else if (hour >= 22 || hour <= 6) baseCongestion -= 30;
        
        // Add some randomness
        baseCongestion += Math.random() * 20 - 10;
        baseCongestion = Math.max(0, Math.min(100, baseCongestion));
        
        hourlyPredictions.push({
          timeSlot: `${hour.toString().padStart(2, '0')}:00`,
          hour,
          predictedCongestion: Math.round(baseCongestion),
          confidence: Math.round(75 + Math.random() * 20),
          factors: ['Historical patterns', 'Weather conditions', 'Events'],
          weatherImpact: Math.round(Math.random() * 10 - 5),
          eventImpact: Math.round(Math.random() * 15)
        });
      }
      
      return {
        areaId: area.id,
        areaName: area.name,
        currentCongestion: area.congestionLevel,
        predictions: hourlyPredictions,
        recommendations: [
          'Avoid during 8-10 AM peak hours',
          'Best time: 2-4 PM',
          'Consider alternative routes during evening rush'
        ],
        alerts: [
          {
            type: area.congestionLevel > 80 ? 'critical' : area.congestionLevel > 60 ? 'warning' : 'info',
            message: `Current congestion level: ${area.congestionLevel}%`,
            time: new Date().toLocaleTimeString()
          }
        ]
      };
    });
    
    setPredictions(samplePredictions);
  }, [areas]);

  // Initialize analytics data
  const initializeAnalytics = useCallback(() => {
    const analyticsData: CongestionAnalytics = {
      hourlyData: Array.from({ length: 24 }, (_, i) => {
        const hour = i.toString().padStart(2, '0') + ':00';
        let congestion = 30;
        
        if (i >= 7 && i <= 10) congestion = 75;
        else if (i >= 17 && i <= 20) congestion = 80;
        else if (i >= 22 || i <= 6) congestion = 15;
        
        return {
          hour,
          congestion: congestion + Math.random() * 20 - 10,
          vehicles: Math.round(congestion * 2.5 + Math.random() * 50),
          speed: Math.round(45 - (congestion * 0.4) + Math.random() * 10)
        };
      }),
      dailyTrends: [
        { day: 'Monday', averageCongestion: 65, peakHour: '18:00', minCongestion: 15, maxCongestion: 85 },
        { day: 'Tuesday', averageCongestion: 58, peakHour: '17:30', minCongestion: 12, maxCongestion: 82 },
        { day: 'Wednesday', averageCongestion: 62, peakHour: '18:15', minCongestion: 18, maxCongestion: 88 },
        { day: 'Thursday', averageCongestion: 67, peakHour: '17:45', minCongestion: 20, maxCongestion: 90 },
        { day: 'Friday', averageCongestion: 72, peakHour: '18:30', minCongestion: 25, maxCongestion: 95 },
        { day: 'Saturday', averageCongestion: 45, peakHour: '14:00', minCongestion: 10, maxCongestion: 70 },
        { day: 'Sunday', averageCongestion: 35, peakHour: '16:00', minCongestion: 8, maxCongestion: 60 }
      ],
      monthlyComparison: [
        { month: 'Jan', congestion: 55, improvement: -3 },
        { month: 'Feb', congestion: 52, improvement: 3 },
        { month: 'Mar', congestion: 58, improvement: -6 },
        { month: 'Apr', congestion: 62, improvement: -4 },
        { month: 'May', congestion: 65, improvement: -3 },
        { month: 'Jun', congestion: 68, improvement: -3 }
      ],
      heatmapData: areas.map(area => ({
        x: area.coordinates.lng,
        y: area.coordinates.lat,
        congestion: area.congestionLevel,
        areaName: area.name
      }))
    };
    
    setAnalytics(analyticsData);
  }, [areas]);

  // Simulate real-time updates
  const updateCongestionData = useCallback(() => {
    setAreas(prev => prev.map(area => {
      const variation = (Math.random() - 0.5) * 10;
      const newLevel = Math.max(0, Math.min(100, area.congestionLevel + variation));
      
      let status: 'Low' | 'Medium' | 'High' | 'Critical';
      if (newLevel < settings.alertThresholds.medium) status = 'Low';
      else if (newLevel < settings.alertThresholds.high) status = 'Medium';
      else if (newLevel < settings.alertThresholds.critical) status = 'High';
      else status = 'Critical';
      
      return {
        ...area,
        congestionLevel: Math.round(newLevel),
        congestionStatus: status,
        vehicleCount: Math.round(area.vehicleCount + (Math.random() - 0.5) * 20),
        averageSpeed: Math.round(45 - (newLevel * 0.4) + Math.random() * 5),
        waitTime: Math.round((newLevel * 0.15) + Math.random() * 2),
        lastUpdated: new Date(),
        trend: variation > 2 ? 'increasing' : variation < -2 ? 'decreasing' : 'stable'
      };
    }));
    
    setLastUpdate(new Date());
  }, [settings.alertThresholds]);

  // Find optimal route
  const findOptimalRoute = useCallback((origin: string, destination: string) => {
    const optimalRoute = routes.reduce((best, current) => {
      if (settings.routePreferences.prioritizeFastest) {
        return current.totalTime < best.totalTime ? current : best;
      }
      if (settings.routePreferences.avoidHighTraffic) {
        return current.averageCongestion < best.averageCongestion ? current : best;
      }
      return current.totalDistance < best.totalDistance ? current : best;
    });
    
    setSelectedRoute(optimalRoute);
    return optimalRoute;
  }, [routes, settings.routePreferences]);

  // Initialize data on mount
  useEffect(() => {
    initializeAreas();
  }, [initializeAreas]);

  useEffect(() => {
    if (areas.length > 0) {
      initializeRoutes();
      initializePredictions();
      initializeAnalytics();
    }
  }, [areas, initializeRoutes, initializePredictions, initializeAnalytics]);

  // Set up real-time updates
  useEffect(() => {
    const interval = setInterval(updateCongestionData, settings.refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [updateCongestionData, settings.refreshInterval]);

  // Update predictions when areas change
  useEffect(() => {
    if (areas.length > 0) {
      initializePredictions();
    }
  }, [areas, initializePredictions]);

  return {
    // Data
    areas,
    selectedArea,
    setSelectedArea,
    routes,
    selectedRoute,
    setSelectedRoute,
    predictions,
    analytics,
    
    // State
    isLoading,
    error,
    lastUpdate,
    
    // Settings
    settings,
    setSettings,
    
    // Actions
    updateCongestionData,
    findOptimalRoute,
    
    // Utilities
    getCongestionColor: (level: number) => {
      if (level < settings.alertThresholds.medium) return '#4ade80'; // green
      if (level < settings.alertThresholds.high) return '#fbbf24'; // yellow
      if (level < settings.alertThresholds.critical) return '#f97316'; // orange
      return '#ef4444'; // red
    },
    
    getCongestionStatus: (level: number) => {
      if (level < settings.alertThresholds.medium) return 'Low';
      if (level < settings.alertThresholds.high) return 'Medium';
      if (level < settings.alertThresholds.critical) return 'High';
      return 'Critical';
    }
  };
};