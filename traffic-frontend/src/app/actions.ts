import type { Route, Area, TrafficPrediction } from '@/lib/types';

export async function getOptimalRoutes(routes: Route[]) {
  // Simulate API call with delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  // Simulate finding optimal route
  const optimizedRoutes = routes.map((route, index) => ({
    ...route,
    isOptimal: index === 0, // First route is optimal
    averageCongestion: Math.max(20, route.averageCongestion - Math.floor(Math.random() * 15))
  }));
  
  return {
    success: true,
    data: optimizedRoutes
  };
}

export async function getCongestionPrediction(area: Area) {
  // Simulate API call with delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Generate mock prediction data
  const predictions = [
    { timeSlot: '10 PM', predictedCongestion: Math.max(20, area.congestionLevel - 5), confidence: 95 },
    { timeSlot: '11 PM', predictedCongestion: Math.max(15, area.congestionLevel - 15), confidence: 90 },
    { timeSlot: '12 AM', predictedCongestion: Math.max(10, area.congestionLevel - 25), confidence: 88 },
  ];
  
  const prediction: TrafficPrediction = {
    areaId: area.id,
    areaName: area.name,
    predictions,
    recommendations: [
      'Optimize signal timing based on predicted patterns.',
      'Consider rerouting traffic during peak hours.',
    ],
    reasoning: `Based on current congestion level of ${area.congestionLevel}% and historical patterns, traffic is expected to decrease gradually over the next few hours.`
  };
  
  return {
    success: true,
    data: prediction
  };
}