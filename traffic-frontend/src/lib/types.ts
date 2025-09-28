export type CongestionStatus = 'Low' | 'Medium' | 'High' | 'Critical';

export interface Area {
  id: string;
  name: string;
  congestionLevel: number;
  congestionStatus: CongestionStatus;
  vehicleCount: number;
  averageSpeed: number;
  waitTime: number;
}

export interface Route {
  id: string;
  name: string;
  origin: string;
  destination: string;
  totalDistance: number;
  totalTime: number;
  averageCongestion: number;
  isOptimal: boolean;
}

export interface PredictionTimeSlot {
  timeSlot: string;
  predictedCongestion: number;
  confidence: number;
}

export interface TrafficPrediction {
  areaId: string;
  areaName: string;
  predictions: PredictionTimeSlot[];
  recommendations: string[];
  reasoning: string;
}

export interface SystemAlert {
  id: string;
  type: 'info' | 'warning' | 'success';
  message: string;
  timestamp: string;
  severity?: 'low' | 'medium' | 'high';
}