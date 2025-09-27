// Smart Traffic System Type Definitions

export interface CrossPath {
  id: number;
  name: string;
  x: number;
  y: number;
  congestion: 'Low' | 'Medium' | 'High';
  vehicles: number;
}

export interface TrafficSignal {
  state: 'red' | 'yellow' | 'green';
  countdown: number;
}

export interface TrafficSignals {
  north: TrafficSignal;
  south: TrafficSignal;
  east: TrafficSignal;
  west: TrafficSignal;
}

export interface OverrideLog {
  id: number;
  direction: string;
  state: string;
  time: string;
  user: string;
}

export interface TrafficStats {
  cars: number;
  trucks: number;
  buses: number;
  motorcycles: number;
  total: number;
}

export interface BackendVehicleResponse {
  cars: number | string;
  trucks: number | string;
  buses: number | string;
  motorcycles: number | string;
  total: number | string;
  is_playing?: boolean;
  error?: string;
}

export interface Accident {
  id: number;
  location: string;
  time: string;
  severity: 'Low' | 'Medium' | 'High';
  status: 'Active' | 'Resolved';
  notes: string;
}

export interface NewAccident {
  location: string;
  severity: 'Low' | 'Medium' | 'High';
  notes: string;
}

export interface TrafficTrend {
  hour: string;
  vehicles: number;
}

export interface Theme {
  primary: string;
  secondary: string;
  darkText: string;
  neutral: string;
  background: string;
  accent: string;
}

export interface VehicleDistribution {
  type: string;
  percentage: number;
  color: string;
}

export interface NavItem {
  id: string;
  icon: any;
  label: string;
}

export interface TruckDetection {
  id: string;
  timestamp: Date;
  location: string;
  section: string;
  isRestricted: boolean;
  status: 'active' | 'acknowledged' | 'permitted';
  notes?: string;
}

export interface TruckWarning {
  id: string;
  detection: TruckDetection;
  message: string;
  actions: TruckWarningAction[];
  createdAt: Date;
  resolvedAt?: Date;
  resolvedBy?: string;
}

export interface TruckWarningAction {
  id: string;
  label: string;
  type: 'inform' | 'permit' | 'dismiss';
  color: 'blue' | 'green' | 'red';
}