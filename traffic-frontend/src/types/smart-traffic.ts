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
  cars: string;
  trucks: string;
  buses: string;
  motorcycles: string;
  total: string;
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