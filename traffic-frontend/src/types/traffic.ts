// Define the type for the vehicle counts data
export interface VehicleCounts {
  car: number;
  motorcycle: number;
  bus: number;
  truck: number;
  total: number;
}

// Define the type for the backend status response
export interface BackendStatus {
  counters: VehicleCounts;
  is_playing: boolean;
  frame_skip: number;
  current_frame: number;
}

// Define the type for vehicle display items
export interface VehicleDisplayItem {
  label: string;
  count: number;
  emoji: string;
  color: string;
}