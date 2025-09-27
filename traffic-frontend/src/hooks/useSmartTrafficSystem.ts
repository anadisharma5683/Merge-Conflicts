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

  // Traffic Signal States
  const [trafficSignals, setTrafficSignals] = useState<TrafficSignals>({
    north: { state: 'red', countdown: 45 },
    south: { state: 'green', countdown: 30 },
    east: { state: 'yellow', countdown: 5 },
    west: { state: 'red', countdown: 50 }
  });

  // Manual Override
  const [overrideMode, setOverrideMode] = useState(false);
  const [overrideLogs, setOverrideLogs] = useState<OverrideLog[]>([]);

  // Congestion & Analytics
  const [congestionLevel] = useState(65);
  const [trafficStats] = useState<TrafficStats>({
    cars: "Will be updated",
    trucks: "Will be updated",
    buses: "Will be updated",
    motorcycles: "Will be updated",
    total: "Will be updated"
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

  // Traffic signal countdown timer
  useEffect(() => {
    if (!isLoggedIn) return;
    
    const interval = setInterval(() => {
      setTrafficSignals(prev => {
        const newSignals = { ...prev };
        (Object.keys(newSignals) as Array<keyof typeof newSignals>).forEach(direction => {
          if (newSignals[direction].countdown > 0) {
            newSignals[direction].countdown -= 1;
          } else {
            // Cycle through states
            const states = ['red', 'yellow', 'green'] as const;
            const currentIndex = states.indexOf(newSignals[direction].state);
            const nextIndex = (currentIndex + 1) % states.length;
            newSignals[direction].state = states[nextIndex];
            newSignals[direction].countdown = nextIndex === 0 ? 60 : nextIndex === 1 ? 5 : 30;
          }
        });
        return newSignals;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isLoggedIn]);

  // Manual signal override
  const handleSignalOverride = (direction: keyof TrafficSignals, newState: 'red' | 'yellow' | 'green') => {
    setTrafficSignals(prev => ({
      ...prev,
      [direction]: { state: newState, countdown: newState === 'red' ? 60 : newState === 'yellow' ? 5 : 30 }
    }));
    
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