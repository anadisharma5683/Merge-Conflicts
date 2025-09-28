import type { Area, Route, TrafficPrediction } from './types';

export const initialAreas: Area[] = [
  {
    id: '1',
    name: 'Vijay Nagar',
    congestionLevel: 85,
    congestionStatus: 'Critical',
    vehicleCount: 350,
    averageSpeed: 15,
    waitTime: 8,
  },
  {
    id: '2',
    name: 'Bhawarkua',
    congestionLevel: 72,
    congestionStatus: 'High',
    vehicleCount: 280,
    averageSpeed: 22,
    waitTime: 5,
  },
  {
    id: '3',
    name: 'Palasia',
    congestionLevel: 45,
    congestionStatus: 'Medium',
    vehicleCount: 150,
    averageSpeed: 40,
    waitTime: 2,
  },
  {
    id: '4',
    name: 'AB Road',
    congestionLevel: 25,
    congestionStatus: 'Low',
    vehicleCount: 120,
    averageSpeed: 55,
    waitTime: 1,
  },
];

export const initialRoutes: Route[] = [
  {
    id: '1',
    name: 'NH-16',
    origin: 'Patia',
    destination: 'Bhubaneswar Station',
    totalDistance: 12,
    totalTime: 25,
    averageCongestion: 65,
    isOptimal: true,
  },
  {
    id: '2',
    name: 'City Road',
    origin: 'Patia',
    destination: 'Bhubaneswar Station',
    totalDistance: 10,
    totalTime: 35,
    averageCongestion: 80,
    isOptimal: false,
  },
];

export const initialPredictions: TrafficPrediction[] = [
  {
    areaId: '1',
    areaName: 'Vijay Nagar',
    predictions: [
      { timeSlot: '10 PM', predictedCongestion: 92, confidence: 95 },
      { timeSlot: '11 PM', predictedCongestion: 75, confidence: 90 },
      { timeSlot: '12 AM', predictedCongestion: 50, confidence: 88 },
    ],
    recommendations: [
      'Increase signal time for North-South corridor.',
      'Divert traffic via Ring Road.',
    ],
    reasoning: 'Based on historical traffic patterns and current conditions, we predict increased congestion during peak hours. The AI model suggests optimizing signal timing to improve traffic flow.',
  },
  {
    areaId: '2',
    areaName: 'Bhawarkua',
    predictions: [
      { timeSlot: '10 PM', predictedCongestion: 78, confidence: 92 },
      { timeSlot: '11 PM', predictedCongestion: 65, confidence: 89 },
      { timeSlot: '12 AM', predictedCongestion: 45, confidence: 85 },
    ],
    recommendations: [
      'Optimize traffic light timings.',
      'Consider alternative routes for heavy vehicles.',
    ],
    reasoning: 'Current traffic patterns show moderate congestion. AI analysis suggests minor optimizations to improve flow efficiency.',
  },
  {
    areaId: '3',
    areaName: 'Palasia',
    predictions: [
      { timeSlot: '10 PM', predictedCongestion: 52, confidence: 88 },
      { timeSlot: '11 PM', predictedCongestion: 38, confidence: 85 },
      { timeSlot: '12 AM', predictedCongestion: 25, confidence: 82 },
    ],
    recommendations: [
      'Maintain current signal settings.',
      'Monitor for any unusual patterns.',
    ],
    reasoning: 'Traffic flow is currently optimal with low congestion levels. No immediate interventions required.',
  },
  {
    areaId: '4',
    areaName: 'AB Road',
    predictions: [
      { timeSlot: '10 PM', predictedCongestion: 30, confidence: 90 },
      { timeSlot: '11 PM', predictedCongestion: 22, confidence: 87 },
      { timeSlot: '12 AM', predictedCongestion: 15, confidence: 85 },
    ],
    recommendations: [
      'Continue current traffic management.',
      'Use as alternate route for diverted traffic.',
    ],
    reasoning: 'Excellent traffic flow with minimal congestion. This route can serve as a backup for other congested areas.',
  },
];

export function getCongestionColor(level: number): string {
  if (level >= 80) return '#dc2626'; // red-600
  if (level >= 60) return '#ea580c'; // orange-600
  if (level >= 40) return '#ca8a04'; // yellow-600
  return '#16a34a'; // green-600
}