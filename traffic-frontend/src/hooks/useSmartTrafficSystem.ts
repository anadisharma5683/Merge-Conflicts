'use client';

import { useState, useEffect } from 'react';
import { 
  TrafficSignals, 
  OverrideLog, 
  TrafficStats, 
  Accident, 
  NewAccident,
  CrossPath,
  TrafficTrend
} from '@/types/smart-traffic';

export const useSmartTrafficSystem = () => {
  // Authentication & Navigation
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [activeSection, setActiveSection] = useState('map');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  // Map & Cross Path Selection
  const [selectedCrossPath, setSelectedCrossPath] = useState<CrossPath | null>(null);
  const [showPathDetails, setShowPathDetails] = useState(false);

  // Video Controls
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [videoVolume, setVideoVolume] = useState(50);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Traffic Signal States - 4-way intersection with proper timing
  // 60-second cycle: North(15s) -> East(15s) -> South(15s) -> West(15s)
  const [trafficSignals, setTrafficSignals] = useState<TrafficSignals>({
    north: { state: 'green', countdown: 15 },
    south: { state: 'red', countdown: 45 },
    east: { state: 'red', countdown: 30 },
    west: { state: 'red', countdown: 60 }
  });
  const [currentCycle, setCurrentCycle] = useState(0); // 0=North, 1=East, 2=South, 3=West

  // Manual Override
  const [overrideMode, setOverrideMode] = useState(false);
  const [overrideLogs, setOverrideLogs] = useState<OverrideLog[]>([]);

  // Congestion & Analytics
  const [congestionLevel] = useState(65);
  const [trafficStats] = useState<TrafficStats>({
    cars: 0,
    trucks: 0,
    buses: 0,
    motorcycles: 0,
    total: 0
  });

  // Accident Reporting
  const [accidents, setAccidents] = useState<Accident[]>([
    { id: 1, location: 'Cross Path 1', time: '2025-01-15 14:30', severity: 'Medium', status: 'Active', notes: 'Minor collision, traffic blocked' },
    { id: 2, location: 'Cross Path 3', time: '2025-01-15 12:15', severity: 'Low', status: 'Resolved', notes: 'Vehicle breakdown cleared' }
  ]);
  const [showAccidentForm, setShowAccidentForm] = useState(false);
  const [newAccident, setNewAccident] = useState<NewAccident>({
    location: '',
    severity: 'Low',
    notes: ''
  });

  // Sample cross paths data
  const crossPaths: CrossPath[] = [
    { id: 1, name: 'Rajmahal Square', x: 25, y: 30, congestion: 'High', vehicles: 45 },
    { id: 2, name: 'Kalpana Square', x: 60, y: 45, congestion: 'Medium', vehicles: 32 },
    { id: 3, name: 'Shastri Nagar Square', x: 40, y: 70, congestion: 'Low', vehicles: 18 },
    { id: 4, name: 'Acharya Vihar Square', x: 75, y: 25, congestion: 'High', vehicles: 52 },
    { id: 5, name: 'Maharishi College Square', x: 20, y: 80, congestion: 'Medium', vehicles: 28 }
  ];

  // Traffic trends data
  const trafficTrends: TrafficTrend[] = [
    { hour: '6 AM', vehicles: 120 },
    { hour: '7 AM', vehicles: 280 },
    { hour: '8 AM', vehicles: 450 },
    { hour: '9 AM', vehicles: 320 },
    { hour: '10 AM', vehicles: 250 },
    { hour: '11 AM', vehicles: 300 },
    { hour: '12 PM', vehicles: 380 }
  ];

  // Authentication
  const handleLogin = (e: { preventDefault: () => void; }) => {
    e.preventDefault();
    if (username === 'admin' && password === 'admin') {
      setIsLoggedIn(true);
      setLoginError('');
    } else {
      setLoginError('Invalid credentials. Use admin/admin');
    }
  };

  // Traffic signal countdown timer - 4-way intersection logic
  useEffect(() => {
    if (!isLoggedIn || overrideMode) return;
    
    const interval = setInterval(() => {
      setTrafficSignals(prev => {
        const newSignals = { ...prev };
        
        // Decrease all countdowns
        (Object.keys(newSignals) as Array<keyof typeof newSignals>).forEach(direction => {
          if (newSignals[direction].countdown > 0) {
            newSignals[direction].countdown -= 1;
          }
        });
        
        // Check if current green phase is ending
        const directions = ['north', 'east', 'south', 'west'] as const;
        const currentGreenDirection = directions[currentCycle % 4];
        
        if (newSignals[currentGreenDirection].countdown === 0) {
          // Move to next phase
          setCurrentCycle(prev => prev + 1);
          
          const nextCycle = (currentCycle + 1) % 4;
          const nextGreenDirection = directions[nextCycle];
          
          // Set all signals to red first
          directions.forEach(dir => {
            newSignals[dir].state = 'red';
          });
          
          // Set next direction to green with 15-second timer
          newSignals[nextGreenDirection].state = 'green';
          newSignals[nextGreenDirection].countdown = 15;
          
          // Set countdown for other directions based on when they'll be green
          directions.forEach((dir, index) => {
            if (dir !== nextGreenDirection) {
              const cyclesUntilGreen = (index - nextCycle + 4) % 4;
              newSignals[dir].countdown = cyclesUntilGreen === 0 ? 60 : cyclesUntilGreen * 15;
            }
          });
        }
        
        return newSignals;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isLoggedIn, overrideMode, currentCycle]);

  // Manual signal override - Updated for 4-way intersection
  const handleSignalOverride = (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => {
    if (newState === 'green') {
      // When setting a direction to green, set all others to red
      const directions = ['north', 'east', 'south', 'west'] as const;
      const newSignals = { ...trafficSignals };
      
      directions.forEach(dir => {
        if (dir === direction) {
          newSignals[dir] = { state: 'green', countdown: 15 };
          // Update current cycle to match the manually set direction
          setCurrentCycle(directions.indexOf(direction));
        } else {
          newSignals[dir] = { state: 'red', countdown: Math.floor(Math.random() * 45) + 15 };
        }
      });
      
      setTrafficSignals(newSignals);
    } else {
      // For red or yellow, just update that specific signal
      setTrafficSignals(prev => ({
        ...prev,
        [direction]: { 
          state: newState, 
          countdown: newState === 'red' ? 45 : 5 
        }
      }));
    }
    
    const log: OverrideLog = {
      id: Date.now(),
      direction,
      state: newState,
      time: new Date().toLocaleString(),
      user: 'Admin'
    };
    setOverrideLogs(prev => [log, ...prev.slice(0, 9)]);
  };

  // Accident form submission
  const handleAccidentSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const accident: Accident = {
      id: Date.now(),
      ...newAccident,
      time: new Date().toLocaleString(),
      status: 'Active'
    };
    setAccidents(prev => [accident, ...prev]);
    setNewAccident({ location: '', severity: 'Low', notes: '' });
    setShowAccidentForm(false);
  };

  return {
    // Authentication
    isLoggedIn,
    setIsLoggedIn,
    username,
    setUsername,
    password,
    setPassword,
    loginError,
    handleLogin,
    
    // Navigation
    activeSection,
    setActiveSection,
    
    // Map
    selectedCrossPath,
    setSelectedCrossPath,
    showPathDetails,
    setShowPathDetails,
    crossPaths,
    
    // Video
    isVideoPlaying,
    setIsVideoPlaying,
    videoVolume,
    setVideoVolume,
    isFullscreen,
    setIsFullscreen,
    
    // Traffic Signals
    trafficSignals,
    overrideMode,
    setOverrideMode,
    overrideLogs,
    handleSignalOverride,
    
    // Analytics
    congestionLevel,
    trafficStats,
    trafficTrends,
    
    // Accidents
    accidents,
    setAccidents,
    showAccidentForm,
    setShowAccidentForm,
    newAccident,
    setNewAccident,
    handleAccidentSubmit
  };
};