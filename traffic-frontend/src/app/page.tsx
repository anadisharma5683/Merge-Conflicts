"use client";
import React, { useState, useEffect } from 'react';
import { 
  MapPin, 
  Play, 
  Pause, 
  Volume2, 
  Maximize, 
  Settings, 
  AlertTriangle, 
  FileText,
  Clock,
  Activity,
  BarChart3,
  Download,
  Plus,
  Eye,
  Edit3,
  CheckCircle,
  XCircle
} from 'lucide-react';

const SmartTrafficSystem = () => {
  // Authentication & Navigation
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [activeSection, setActiveSection] = useState('map');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  // Map & Cross Path Selection
  interface CrossPath {
    id: number;
    name: string;
    x: number;
    y: number;
    congestion: string;
    vehicles: number;
  }
  
  const [selectedCrossPath, setSelectedCrossPath] = useState<CrossPath | null>(null);
  const [showPathDetails, setShowPathDetails] = useState(false);

  // Video Controls
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [videoVolume, setVideoVolume] = useState(50);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Traffic Signal States
  const [trafficSignals, setTrafficSignals] = useState({
    north: { state: 'red', countdown: 45 },
    south: { state: 'green', countdown: 30 },
    east: { state: 'yellow', countdown: 5 },
    west: { state: 'red', countdown: 50 }
  });

  // Manual Override
  const [overrideMode, setOverrideMode] = useState(false);
  interface OverrideLog {
    id: number;
    direction: string;
    state: string;
    time: string;
    user: string;
  }
  const [overrideLogs, setOverrideLogs] = useState<OverrideLog[]>([]);

  // Congestion & Analytics
  const [congestionLevel, ] = useState(65);
  const [trafficStats, ] = useState({
    cars: 234,
    trucks: 45,
    buses: 12,
    motorcycles: 89,
    total: 380
  });

  // Accident Reporting
  const [accidents, setAccidents] = useState([
    { id: 1, location: 'Cross Path 1', time: '2025-01-15 14:30', severity: 'Medium', status: 'Active', notes: 'Minor collision, traffic blocked' },
    { id: 2, location: 'Cross Path 3', time: '2025-01-15 12:15', severity: 'Low', status: 'Resolved', notes: 'Vehicle breakdown cleared' }
  ]);
  const [showAccidentForm, setShowAccidentForm] = useState(false);
  const [newAccident, setNewAccident] = useState({
    location: '',
    severity: 'Low',
    notes: ''
  });

  // Sample cross paths data
  const crossPaths = [
    { id: 1, name: 'Rajmahal Square', x: 25, y: 30, congestion: 'High', vehicles: 45 },
    { id: 2, name: 'Kalpana Square', x: 60, y: 45, congestion: 'Medium', vehicles: 32 },
    { id: 3, name: 'Shastri Nagar Square', x: 40, y: 70, congestion: 'Low', vehicles: 18 },
    { id: 4, name: 'Acharya Vihar Square', x: 75, y: 25, congestion: 'High', vehicles: 52 },
    { id: 5, name: 'Maharishi College Square', x: 20, y: 80, congestion: 'Medium', vehicles: 28 }
  ];

  // Traffic trends data
  const trafficTrends = [
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
              const states = ['red', 'yellow', 'green'];
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
  const handleSignalOverride = (direction: 'north' | 'south' | 'east' | 'west', newState: 'red' | 'yellow' | 'green') => {
    setTrafficSignals(prev => ({
      ...prev,
      [direction]: { state: newState, countdown: newState === 'red' ? 60 : newState === 'yellow' ? 5 : 30 }
    }));
    
    const log = {
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
    const accident = {
      id: Date.now(),
      ...newAccident,
      time: new Date().toLocaleString(),
      status: 'Active'
    };
    setAccidents(prev => [accident, ...prev]);
    setNewAccident({ location: '', severity: 'Low', notes: '' });
    setShowAccidentForm(false);
  };

  // Theme colors
  const theme = {
    primary: '#12355b',
    secondary: '#c16e70',
    darkText: '#3d1308',
    neutral: '#4a442d',
    background: '#ffffff',
    accent: '#f8f9fa'
  };

  // Login Screen
  if (!isLoggedIn) {
    return (
      <div style={{
        minHeight: '100vh',
        background: `linear-gradient(135deg, ${theme.primary} 0%, ${theme.secondary} 100%)`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}>
        <div style={{
          background: theme.background,
          padding: '40px',
          borderRadius: '15px',
          boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
          width: '100%',
          maxWidth: '400px'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '30px' }}>
            <div style={{
              width: '80px',
              height: '80px',
              background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
              borderRadius: '50%',
              margin: '0 auto 20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px'
            }}>
              🚦
            </div>
            <h1 style={{ color: theme.primary, margin: '0', fontSize: '24px' }}>
              Smart Traffic Management
            </h1>
            <p style={{ color: theme.neutral, margin: '10px 0 0 0' }}>
              Login to access the system
            </p>
          </div>

          <form onSubmit={handleLogin}>
            <div style={{ marginBottom: '20px' }}>
              <input
                type="text"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                style={{
                  width: '100%',
                  padding: '15px',
                  border: `2px solid ${theme.primary}20`,
                  borderRadius: '8px',
                  fontSize: '16px',
                  boxSizing: 'border-box'
                }}
                required
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  width: '100%',
                  padding: '15px',
                  border: `2px solid ${theme.primary}20`,
                  borderRadius: '8px',
                  fontSize: '16px',
                  boxSizing: 'border-box'
                }}
                required
              />
            </div>

            {loginError && (
              <div style={{
                color: theme.secondary,
                background: `${theme.secondary}10`,
                padding: '10px',
                borderRadius: '5px',
                marginBottom: '20px',
                fontSize: '14px'
              }}>
                {loginError}
              </div>
            )}

            <button
              type="submit"
              style={{
                width: '100%',
                padding: '15px',
                background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: 'pointer'
              }}
            >
              Login to Dashboard
            </button>
          </form>

          <div style={{
            marginTop: '30px',
            padding: '15px',
            background: theme.accent,
            borderRadius: '8px',
            fontSize: '14px',
            color: theme.neutral
          }}>
            <strong>Demo Credentials:</strong><br />
            Username: admin<br />
            Password: admin
          </div>
        </div>
      </div>
    );
  }

  // Main Dashboard
  return (
    <div style={{
      minHeight: '100vh',
      background: theme.accent,
      fontFamily: 'system-ui, -apple-system, sans-serif',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Top Navbar */}
      <header style={{
        background: theme.background,
        padding: '15px 30px',
        borderBottom: `2px solid ${theme.primary}10`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: '0 2px 10px rgba(0,0,0,0.05)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            background: `linear-gradient(45deg, ${theme.primary}, ${theme.secondary})`,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '20px'
          }}>
            🚦
          </div>
          <h1 style={{ color: theme.primary, margin: 0, fontSize: '20px' }}>
            Smart Traffic Management System
          </h1>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <div style={{
            padding: '8px 15px',
            background: `${theme.primary}10`,
            borderRadius: '20px',
            color: theme.primary,
            fontSize: '14px'
          }}>
            Welcome, Admin
          </div>
          <button
            onClick={() => setIsLoggedIn(false)}
            style={{
              padding: '8px 15px',
              background: theme.secondary,
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Logout
          </button>
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1 }}>
        {/* Left Sidebar */}
        <nav style={{
          width: '250px',
          background: theme.background,
          borderRight: `2px solid ${theme.primary}10`,
          padding: '20px'
        }}>
          {[
            { id: 'map', icon: MapPin, label: 'Interactive Map' },
            { id: 'video', icon: Play, label: 'Live Video Feed' },
            { id: 'signals', icon: Settings, label: 'Signal Status' },
            { id: 'congestion', icon: Activity, label: 'Congestion Monitor' },
            { id: 'analytics', icon: BarChart3, label: 'Traffic Analytics' },
            { id: 'accidents', icon: AlertTriangle, label: 'Accident Reports' }
          ].map(item => (
            <button
              key={item.id}
              onClick={() => setActiveSection(item.id)}
              style={{
                width: '100%',
                padding: '15px',
                background: activeSection === item.id ? `${theme.primary}15` : 'transparent',
                border: activeSection === item.id ? `2px solid ${theme.primary}` : 'none',
                borderRadius: '8px',
                color: activeSection === item.id ? theme.primary : theme.darkText,
                cursor: 'pointer',
                marginBottom: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                fontSize: '14px',
                textAlign: 'left'
              }}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>

        {/* Main Content */}
        <main style={{ flex: 1, padding: '30px', overflow: 'auto' }}>
          {activeSection === 'map' && (
            <div>
              <h2 style={{ color: theme.primary, marginBottom: '20px' }}>City Traffic Map</h2>
              
              <div style={{ display: 'flex', gap: '30px' }}>
                {/* Map Area */}
                <div style={{
                  flex: 2,
                  background: theme.background,
                  borderRadius: '15px',
                  padding: '20px',
                  position: 'relative',
                  minHeight: '500px',
                  border: `2px solid ${theme.primary}10`
                }}>
                  {/* Mock City Map */}
                  <div style={{
                    width: '100%',
                    height: '500px',
                    background: `linear-gradient(45deg, ${theme.accent} 0%, #e8f4f8 100%)`,
                    borderRadius: '10px',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    {/* Map Grid Lines */}
                    <svg width="100%" height="100%" style={{ position: 'absolute' }}>
                      {Array.from({ length: 10 }, (_, i) => (
                        <g key={i}>
                          <line x1={`${(i + 1) * 10}%`} y1="0" x2={`${(i + 1) * 10}%`} y2="100%" stroke={`${theme.primary}20`} strokeWidth="1" />
                          <line x1="0" y1={`${(i + 1) * 10}%`} x2="100%" y2={`${(i + 1) * 10}%`} stroke={`${theme.primary}20`} strokeWidth="1" />
                        </g>
                      ))}
                    </svg>

                    {/* Cross Path Markers */}
                    {crossPaths.map(path => (
                      <button
                        key={path.id}
                        onClick={() => {
                          setSelectedCrossPath(path);
                          setShowPathDetails(true);
                        }}
                        style={{
                          position: 'absolute',
                          left: `${path.x}%`,
                          top: `${path.y}%`,
                          transform: 'translate(-50%, -50%)',
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          border: 'none',
                          background: path.congestion === 'High' ? theme.secondary : 
                                     path.congestion === 'Medium' ? '#ffa726' : '#4caf50',
                          color: 'white',
                          cursor: 'pointer',
                          fontSize: '18px',
                          boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          (e.target as HTMLElement).style.transform = 'translate(-50%, -50%) scale(1.1)';
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLElement).style.transform = 'translate(-50%, -50%) scale(1)';
                        }}
                      >
                        🚦
                      </button>
                    ))}
                  </div>

                  {/* Map Legend */}
                  <div style={{
                    position: 'absolute',
                    bottom: '30px',
                    left: '30px',
                    background: 'rgba(255,255,255,0.95)',
                    padding: '15px',
                    borderRadius: '10px',
                    boxShadow: '0 5px 20px rgba(0,0,0,0.1)'
                  }}>
                    <h4 style={{ margin: '0 0 10px 0', color: theme.primary }}>Congestion Levels</h4>
                    {[
                      { color: '#4caf50', label: 'Low' },
                      { color: '#ffa726', label: 'Medium' },
                      { color: theme.secondary, label: 'High' }
                    ].map(item => (
                      <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '5px' }}>
                        <div style={{
                          width: '15px',
                          height: '15px',
                          borderRadius: '50%',
                          background: item.color
                        }} />
                        <span style={{ fontSize: '14px', color: theme.darkText }}>{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Side Panel for Selected Cross Path */}
                {showPathDetails && selectedCrossPath && (
                  <div style={{
                    flex: 1,
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '25px',
                    border: `2px solid ${theme.primary}10`,
                    maxHeight: '500px',
                    overflow: 'auto'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                      <h3 style={{ margin: 0, color: theme.primary }}>{selectedCrossPath.name}</h3>
                      <button
                        onClick={() => setShowPathDetails(false)}
                        style={{
                          background: 'none',
                          border: 'none',
                          fontSize: '20px',
                          cursor: 'pointer',
                          color: theme.neutral
                        }}
                      >
                        ×
                      </button>
                    </div>

                    {/* Live Feed Placeholder */}
                    <div style={{
                      width: '100%',
                      height: '200px',
                      background: '#000',
                      borderRadius: '10px',
                      marginBottom: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white'
                    }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '30px', marginBottom: '10px' }}>📹</div>
                        <div>Live Traffic Feed</div>
                        <div style={{ fontSize: '12px', opacity: 0.7 }}>Cross Path {selectedCrossPath.id}</div>
                      </div>
                    </div>

                    {/* Stats */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '15px',
                      marginBottom: '20px'
                    }}>
                      <div style={{
                        padding: '15px',
                        background: theme.accent,
                        borderRadius: '8px',
                        textAlign: 'center'
                      }}>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: theme.primary }}>
                          {selectedCrossPath.vehicles}
                        </div>
                        <div style={{ fontSize: '14px', color: theme.neutral }}>Active Vehicles</div>
                      </div>

                      <div style={{
                        padding: '15px',
                        background: theme.accent,
                        borderRadius: '8px',
                        textAlign: 'center'
                      }}>
                        <div style={{
                          fontSize: '16px',
                          fontWeight: 'bold',
                          color: selectedCrossPath.congestion === 'High' ? theme.secondary :
                                selectedCrossPath.congestion === 'Medium' ? '#ffa726' : '#4caf50'
                        }}>
                          {selectedCrossPath.congestion}
                        </div>
                        <div style={{ fontSize: '14px', color: theme.neutral }}>Congestion</div>
                      </div>
                    </div>

                    <div style={{
                      padding: '15px',
                      background: `${theme.primary}05`,
                      borderRadius: '8px',
                      borderLeft: `4px solid ${theme.primary}`
                    }}>
                      <h4 style={{ margin: '0 0 10px 0', color: theme.primary }}>Current Status</h4>
                      <p style={{ margin: 0, fontSize: '14px', color: theme.darkText }}>
                        Traffic flowing normally with {selectedCrossPath.congestion.toLowerCase()} congestion levels. 
                        Signal timing optimized for current conditions.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Video Feed Section */}
          {/* Video Feed Section */}
          {activeSection === 'video' && (
            <div>
              <h2 style={{ color: theme.primary, marginBottom: '20px' }}>Live Traffic Video Analysis</h2>
              
              <div style={{
                background: theme.background,
                borderRadius: '15px',
                padding: '30px',
                border: `2px solid ${theme.primary}10`
              }}>
                {/* Video Player */}
                <div style={{
                  position: 'relative',
                  background: '#000',
                  borderRadius: '10px',
                  marginBottom: '20px',
                  overflow: 'hidden'
                }}>
                  {/* Real Video Stream from Backend */}
                  <img 
                    src="http://127.0.0.1:5000/video_feed"
                    alt="Live Traffic Feed"
                    style={{
                      width: '100%',
                      height: '500px',
                      objectFit: 'none',
                      display: 'block'
                    }}
                    onError={(e) => {
                      // Fallback to demo content if backend is not available
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                      const fallback = target.parentElement?.querySelector('.video-fallback') as HTMLElement;
                      if (fallback) fallback.style.display = 'flex';
                    }}
                  />
                  
                  {/* Fallback content when backend is not available */}
                  <div className="video-fallback" style={{
                    width: '100%',
                    height: '400px',
                    display: 'none',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    fontSize: '18px',
                    position: 'relative',
                    flexDirection: 'column'
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '60px', marginBottom: '20px' }}>🎥</div>
                      <div>Backend Connection Required</div>
                      <div style={{ fontSize: '14px', opacity: 0.7, marginTop: '10px' }}>
                        Start the Python backend at http://127.0.0.1:5000
                      </div>
                    </div>
                  </div>

                  {/* Video Controls */}
                  <div style={{
                    position: 'absolute',
                    bottom: '0',
                    left: '0',
                    right: '0',
                    background: 'rgba(0,0,0,0.8)',
                    padding: '15px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '15px'
                  }}>
                    <button
                      onClick={async () => {
                        try {
                          const response = await fetch('http://127.0.0.1:5000/play_pause', {
                            method: 'POST'
                          });
                          const data = await response.json();
                          setIsVideoPlaying(data.is_playing);
                        } catch (error) {
                          console.error('Error controlling playback:', error);
                          setIsVideoPlaying(!isVideoPlaying);
                        }
                      }}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '20px'
                      }}
                    >
                      {isVideoPlaying ? <Pause /> : <Play />}
                    </button>

                    <button
                      onClick={async () => {
                        try {
                          await fetch('http://127.0.0.1:5000/reset_counters', {
                            method: 'POST'
                          });
                        } catch (error) {
                          console.error('Error resetting counters:', error);
                        }
                      }}
                      style={{
                        background: theme.primary,
                        border: 'none',
                        color: 'white',
                        cursor: 'pointer',
                        padding: '5px 10px',
                        borderRadius: '3px',
                        fontSize: '12px'
                      }}
                    >
                      Reset Count
                    </button>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <Volume2 size={18} color="white" />
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={videoVolume}
                        onChange={(e) => setVideoVolume(Number(e.target.value))}
                        style={{
                          width: '100px',
                          height: '5px'
                        }}
                      />
                    </div>

                    <div style={{ flex: 1 }} />

                    <button
                      onClick={() => setIsFullscreen(!isFullscreen)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'white',
                        cursor: 'pointer'
                      }}
                    >
                      <Maximize size={18} />
                    </button>
                  </div>
                </div>

                {/* Vehicle Detection Stats */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                  gap: '20px'
                }}>
                  {[
                    { type: 'Cars', count: trafficStats.cars, icon: '🚗', color: '#4caf50' },
                    { type: 'Trucks', count: trafficStats.trucks, icon: '🚚', color: theme.secondary },
                    { type: 'Buses', count: trafficStats.buses, icon: '🚌', color: '#ffa726' },
                    { type: 'Motorcycles', count: trafficStats.motorcycles, icon: '🏍️', color: '#2196f3' }
                  ].map(stat => (
                    <div key={stat.type} style={{
                      background: theme.accent,
                      borderRadius: '10px',
                      padding: '20px',
                      textAlign: 'center',
                      border: `2px solid ${stat.color}20`
                    }}>
                      <div style={{ fontSize: '30px', marginBottom: '10px' }}>{stat.icon}</div>
                      <div style={{
                        fontSize: '24px',
                        fontWeight: 'bold',
                        color: stat.color,
                        marginBottom: '5px'
                      }}>
                        {stat.count}
                      </div>
                      <div style={{ fontSize: '14px', color: theme.neutral }}>{stat.type}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            )}

          {/* Traffic Signals Section */}
          {activeSection === 'signals' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ color: theme.primary, margin: 0 }}>Traffic Signal Status</h2>
                <button
                  onClick={() => setOverrideMode(!overrideMode)}
                  style={{
                    padding: '10px 20px',
                    background: overrideMode ? theme.secondary : theme.primary,
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer'
                  }}
                >
                  {overrideMode ? 'Exit Override' : 'Manual Override'}
                </button>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '25px',
                marginBottom: '30px'
              }}>
                {Object.entries(trafficSignals).map(([direction, signal]) => (
                  <div key={direction} style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '25px',
                    border: `2px solid ${theme.primary}10`,
                    textAlign: 'center'
                  }}>
                    <h3 style={{ 
                      color: theme.primary, 
                      marginBottom: '20px',
                      textTransform: 'capitalize'
                    }}>
                      {direction} Lane
                    </h3>

                    {/* Traffic Light Visual */}
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '10px',
                      marginBottom: '20px'
                    }}>
                      {['red', 'yellow', 'green'].map(color => (
                        <div key={color} style={{
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          background: signal.state === color ? 
                            (color === 'red' ? '#f44336' : color === 'yellow' ? '#ffc107' : '#4caf50') :
                            '#e0e0e0',
                          boxShadow: signal.state === color ? '0 0 20px rgba(255,255,255,0.8)' : 'none',
                          transition: 'all 0.3s ease'
                        }} />
                      ))}
                    </div>

                    {/* Current State */}
                    <div style={{
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: signal.state === 'red' ? '#f44336' : 
                             signal.state === 'yellow' ? '#ffc107' : '#4caf50',
                      marginBottom: '15px',
                      textTransform: 'uppercase'
                    }}>
                      {signal.state}
                    </div>

                    {/* Countdown */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      marginBottom: '20px'
                    }}>
                      <Clock size={16} color={theme.neutral} />
                      <span style={{ color: theme.darkText, fontSize: '16px' }}>
                        {signal.countdown}s remaining
                      </span>
                    </div>

                    {/* Manual Override Controls */}
                    {overrideMode && (
                      <div style={{
                        display: 'flex',
                        gap: '5px',
                        justifyContent: 'center'
                      }}>
                        {['red', 'yellow', 'green'].map(color => (
                          <button
                            key={color}
                            onClick={() => handleSignalOverride(direction as 'north' | 'south' | 'east' | 'west', color as 'red' | 'yellow' | 'green')}
                            style={{
                              padding: '8px 12px',
                              background: color === 'red' ? '#f44336' : 
                                         color === 'yellow' ? '#ffc107' : '#4caf50',
                              color: 'white',
                              border: 'none',
                              borderRadius: '5px',
                              cursor: 'pointer',
                              fontSize: '12px',
                              textTransform: 'uppercase'
                            }}
                          >
                            {color}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {/* Override Logs */}
              {overrideLogs.length > 0 && (
                <div style={{
                  background: theme.background,
                  borderRadius: '15px',
                  padding: '25px',
                  border: `2px solid ${theme.primary}10`
                }}>
                  <h3 style={{ color: theme.primary, marginBottom: '20px' }}>Override History</h3>
                  <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                    {overrideLogs.map(log => (
                      <div key={log.id} style={{
                        padding: '10px',
                        background: theme.accent,
                        borderRadius: '8px',
                        marginBottom: '10px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div>
                          <span style={{ fontWeight: 'bold', color: theme.primary }}>
                            {log.direction.toUpperCase()}
                          </span>
                          <span style={{ margin: '0 10px', color: theme.darkText }}>→</span>
                          <span style={{
                            color: log.state === 'red' ? '#f44336' : 
                                   log.state === 'yellow' ? '#ffc107' : '#4caf50',
                            fontWeight: 'bold',
                            textTransform: 'uppercase'
                          }}>
                            {log.state}
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: theme.neutral }}>
                          {log.time} by {log.user}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Congestion Monitor Section */}
          {activeSection === 'congestion' && (
            <div>
              <h2 style={{ color: theme.primary, marginBottom: '20px' }}>Real-time Congestion Monitor</h2>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 300px',
                gap: '30px'
              }}>
                {/* Main Congestion Display */}
                <div style={{
                  background: theme.background,
                  borderRadius: '15px',
                  padding: '30px',
                  border: `2px solid ${theme.primary}10`
                }}>
                  <div style={{ textAlign: 'center', marginBottom: '30px' }}>
                    <h3 style={{ color: theme.primary, marginBottom: '20px' }}>
                      City-wide Traffic Congestion
                    </h3>
                    
                    {/* Circular Progress Indicator */}
                    <div style={{
                      width: '200px',
                      height: '200px',
                      margin: '0 auto',
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <svg width="200" height="200" style={{ position: 'absolute' }}>
                        <circle
                          cx="100"
                          cy="100"
                          r="80"
                          fill="none"
                          stroke={theme.accent}
                          strokeWidth="20"
                        />
                        <circle
                          cx="100"
                          cy="100"
                          r="80"
                          fill="none"
                          stroke={congestionLevel > 70 ? theme.secondary : congestionLevel > 40 ? '#ffa726' : '#4caf50'}
                          strokeWidth="20"
                          strokeDasharray={`${(congestionLevel / 100) * 502.65} 502.65`}
                          strokeDashoffset="125.66"
                          style={{ transition: 'all 1s ease' }}
                        />
                      </svg>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{
                          fontSize: '36px',
                          fontWeight: 'bold',
                          color: congestionLevel > 70 ? theme.secondary : congestionLevel > 40 ? '#ffa726' : '#4caf50'
                        }}>
                          {congestionLevel}%
                        </div>
                        <div style={{ color: theme.neutral, fontSize: '14px' }}>
                          Congestion Level
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Cross Path Congestion Details */}
                  <div>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      Individual Cross Path Status
                    </h4>
                    <div style={{ display: 'grid', gap: '10px' }}>
                      {crossPaths.map(path => (
                        <div key={path.id} style={{
                          display: 'flex',
                          alignItems: 'center',
                          padding: '15px',
                          background: theme.accent,
                          borderRadius: '8px',
                          border: `1px solid ${theme.primary}10`
                        }}>
                          <div style={{
                            width: '15px',
                            height: '15px',
                            borderRadius: '50%',
                            background: path.congestion === 'High' ? theme.secondary :
                                       path.congestion === 'Medium' ? '#ffa726' : '#4caf50',
                            marginRight: '15px'
                          }} />
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 'bold', color: theme.darkText }}>
                              {path.name}
                            </div>
                            <div style={{ fontSize: '12px', color: theme.neutral }}>
                              {path.vehicles} vehicles • {path.congestion} congestion
                            </div>
                          </div>
                          <div style={{
                            padding: '5px 10px',
                            borderRadius: '15px',
                            background: path.congestion === 'High' ? `${theme.secondary}20` :
                                       path.congestion === 'Medium' ? '#ffa72620' : '#4caf5020',
                            color: path.congestion === 'High' ? theme.secondary :
                                   path.congestion === 'Medium' ? '#ffa726' : '#4caf50',
                            fontSize: '12px',
                            fontWeight: 'bold'
                          }}>
                            {path.congestion}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Side Panel - AI Predictions */}
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '20px'
                }}>
                  {/* Current Status */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      Current Status
                    </h4>
                    <div style={{
                      padding: '15px',
                      background: `${theme.secondary}10`,
                      borderRadius: '8px',
                      borderLeft: `4px solid ${theme.secondary}`
                    }}>
                      <div style={{ fontWeight: 'bold', color: theme.secondary, marginBottom: '5px' }}>
                        Moderate Traffic
                      </div>
                      <div style={{ fontSize: '14px', color: theme.darkText }}>
                        Traffic is flowing with some delays expected at main intersections.
                      </div>
                    </div>
                  </div>

                  {/* AI Predictions */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      AI Predictions
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{
                        padding: '10px',
                        background: theme.accent,
                        borderRadius: '6px',
                        fontSize: '14px'
                      }}>
                        <div style={{ fontWeight: 'bold', color: theme.darkText }}>Next 30 mins:</div>
                        <div style={{ color: '#ffa726' }}>↗ Congestion increasing</div>
                      </div>
                      <div style={{
                        padding: '10px',
                        background: theme.accent,
                        borderRadius: '6px',
                        fontSize: '14px'
                      }}>
                        <div style={{ fontWeight: 'bold', color: theme.darkText }}>Peak at:</div>
                        <div style={{ color: theme.secondary }}>6:30 PM (82% congestion)</div>
                      </div>
                      <div style={{
                        padding: '10px',
                        background: theme.accent,
                        borderRadius: '6px',
                        fontSize: '14px'
                      }}>
                        <div style={{ fontWeight: 'bold', color: theme.darkText }}>Recommended:</div>
                        <div style={{ color: '#4caf50' }}>Optimize signal timing</div>
                      </div>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      Live Metrics
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Avg Speed:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>25 mph</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Active Vehicles:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>1,247</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Wait Time:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>2.3 min</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Analytics Section */}
          {activeSection === 'analytics' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ color: theme.primary, margin: 0 }}>Traffic Analytics & Trends</h2>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button style={{
                    padding: '8px 15px',
                    background: theme.primary,
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px'
                  }}>
                    <Download size={16} />
                    Export CSV
                  </button>
                  <button style={{
                    padding: '8px 15px',
                    background: theme.secondary,
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px'
                  }}>
                    <FileText size={16} />
                    Generate Report
                  </button>
                </div>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: '2fr 1fr',
                gap: '30px'
              }}>
                {/* Chart Area */}
                <div style={{
                  background: theme.background,
                  borderRadius: '15px',
                  padding: '30px',
                  border: `2px solid ${theme.primary}10`
                }}>
                  <h3 style={{ color: theme.primary, marginBottom: '20px' }}>
                     Todays Traffic Flow
                  </h3>
                  
                  {/* Mock Chart */}
                  <div style={{ position: 'relative', height: '300px' }}>
                    <svg width="100%" height="100%" viewBox="0 0 400 300">
                      {/* Grid lines */}
                      {Array.from({ length: 6 }, (_, i) => (
                        <line
                          key={i}
                          x1="50"
                          y1={50 + i * 40}
                          x2="380"
                          y2={50 + i * 40}
                          stroke={`${theme.primary}15`}
                          strokeWidth="1"
                        />
                      ))}
                      
                      {/* Chart line */}
                      <path
                        d="M50,250 Q100,200 150,180 T250,120 Q300,100 350,80"
                        fill="none"
                        stroke={theme.primary}
                        strokeWidth="3"
                      />
                      
                      {/* Data points */}
                      {trafficTrends.map((point, index) => (
                        <circle
                          key={index}
                          cx={50 + index * 50}
                          cy={250 - (point.vehicles / 5)}
                          r="4"
                          fill={theme.secondary}
                        />
                      ))}
                    </svg>
                    
                    {/* X-axis labels */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      paddingLeft: '50px',
                      paddingRight: '20px',
                      marginTop: '10px'
                    }}>
                      {trafficTrends.map((point, index) => (
                        <span key={index} style={{
                          fontSize: '12px',
                          color: theme.neutral
                        }}>
                          {point.hour}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Filter Options */}
                  <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                    {['Hour', 'Day', 'Week', 'Month'].map(period => (
                      <button
                        key={period}
                        style={{
                          padding: '8px 15px',
                          background: period === 'Hour' ? theme.primary : theme.accent,
                          color: period === 'Hour' ? 'white' : theme.darkText,
                          border: 'none',
                          borderRadius: '5px',
                          cursor: 'pointer',
                          fontSize: '14px'
                        }}
                      >
                        {period}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Statistics Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {/* Summary Stats */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      Today&apos;s Summary
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Total Vehicles:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>15,432</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Peak Hour:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>8:00 AM</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Avg Congestion:</span>
                        <span style={{ fontWeight: 'bold', color: theme.primary }}>45%</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: theme.darkText }}>Incidents:</span>
                        <span style={{ fontWeight: 'bold', color: theme.secondary }}>3</span>
                      </div>
                    </div>
                  </div>

                  {/* Vehicle Distribution */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      Vehicle Distribution
                    </h4>
                    {[
                      { type: 'Cars', percentage: 65, color: '#4caf50' },
                      { type: 'Motorcycles', percentage: 25, color: '#2196f3' },
                      { type: 'Trucks', percentage: 8, color: theme.secondary },
                      { type: 'Buses', percentage: 2, color: '#ffa726' }
                    ].map(vehicle => (
                      <div key={vehicle.type} style={{ marginBottom: '15px' }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          marginBottom: '5px'
                        }}>
                          <span style={{ color: theme.darkText, fontSize: '14px' }}>
                            {vehicle.type}
                          </span>
                          <span style={{ color: vehicle.color, fontWeight: 'bold' }}>
                            {vehicle.percentage}%
                          </span>
                        </div>
                        <div style={{
                          height: '6px',
                          background: theme.accent,
                          borderRadius: '3px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            height: '100%',
                            width: `${vehicle.percentage}%`,
                            background: vehicle.color,
                            transition: 'width 1s ease'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Performance Metrics */}
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '20px',
                    border: `2px solid ${theme.primary}10`
                  }}>
                    <h4 style={{ color: theme.primary, marginBottom: '15px' }}>
                      System Performance
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div style={{
                        padding: '10px',
                        background: '#4caf5020',
                        borderRadius: '6px',
                        borderLeft: '4px solid #4caf50'
                      }}>
                        <div style={{ fontSize: '12px', color: '#4caf50', fontWeight: 'bold' }}>
                          ↑ 12% Traffic Efficiency
                        </div>
                      </div>
                      <div style={{
                        padding: '10px',
                        background: `${theme.primary}15`,
                        borderRadius: '6px',
                        borderLeft: `4px solid ${theme.primary}`
                      }}>
                        <div style={{ fontSize: '12px', color: theme.primary, fontWeight: 'bold' }}>
                          99.8% System Uptime
                        </div>
                      </div>
                      <div style={{
                        padding: '10px',
                        background: '#ffa72620',
                        borderRadius: '6px',
                        borderLeft: '4px solid #ffa726'
                      }}>
                        <div style={{ fontSize: '12px', color: '#ffa726', fontWeight: 'bold' }}>
                          ↓ 8% Average Wait Time
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Accidents Section */}
          {activeSection === 'accidents' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ color: theme.primary, margin: 0 }}>Accident Reports & Management</h2>
                <button
                  onClick={() => setShowAccidentForm(true)}
                  style={{
                    padding: '10px 20px',
                    background: theme.primary,
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <Plus size={18} />
                  Report Accident
                </button>
              </div>

              {/* Accidents List */}
              <div style={{
                background: theme.background,
                borderRadius: '15px',
                padding: '25px',
                border: `2px solid ${theme.primary}10`
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                  {accidents.map(accident => (
                    <div key={accident.id} style={{
                      padding: '20px',
                      background: theme.accent,
                      borderRadius: '10px',
                      border: `1px solid ${theme.primary}10`,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '20px'
                    }}>
                      {/* Severity Indicator */}
                      <div style={{
                        width: '50px',
                        height: '50px',
                        borderRadius: '50%',
                        background: accident.severity === 'High' ? theme.secondary :
                                   accident.severity === 'Medium' ? '#ffa726' : '#4caf50',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontWeight: 'bold'
                      }}>
                        !
                      </div>

                      {/* Accident Details */}
                      <div style={{ flex: 1 }}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginBottom: '8px'
                        }}>
                          <h4 style={{ margin: 0, color: theme.primary }}>
                            {accident.location}
                          </h4>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span style={{
                              padding: '4px 12px',
                              borderRadius: '12px',
                              background: accident.status === 'Active' ? `${theme.secondary}20` : '#4caf5020',
                              color: accident.status === 'Active' ? theme.secondary : '#4caf50',
                              fontSize: '12px',
                              fontWeight: 'bold'
                            }}>
                              {accident.status}
                            </span>
                            <span style={{
                              padding: '4px 12px',
                              borderRadius: '12px',
                              background: accident.severity === 'High' ? `${theme.secondary}20` :
                                         accident.severity === 'Medium' ? '#ffa72620' : '#4caf5020',
                              color: accident.severity === 'High' ? theme.secondary :
                                     accident.severity === 'Medium' ? '#ffa726' : '#4caf50',
                              fontSize: '12px',
                              fontWeight: 'bold'
                            }}>
                              {accident.severity} Severity
                            </span>
                          </div>
                        </div>
                        
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                          marginBottom: '10px',
                          color: theme.neutral,
                          fontSize: '14px'
                        }}>
                          <Clock size={14} />
                          {accident.time}
                        </div>
                        
                        <p style={{
                          margin: 0,
                          color: theme.darkText,
                          fontSize: '14px',
                          lineHeight: '1.4'
                        }}>
                          {accident.notes}
                        </p>
                      </div>

                      {/* Action Buttons */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <button style={{
                          padding: '8px 12px',
                          background: theme.primary,
                          color: 'white',
                          border: 'none',
                          borderRadius: '5px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                          fontSize: '12px'
                        }}>
                          <Eye size={14} />
                          View
                        </button>
                        <button style={{
                          padding: '8px 12px',
                          background: '#ffa726',
                          color: 'white',
                          border: 'none',
                          borderRadius: '5px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '5px',
                          fontSize: '12px'
                        }}>
                          <Edit3 size={14} />
                          Edit
                        </button>
                        {accident.status === 'Active' && (
                          <button
                            onClick={() => {
                              setAccidents(prev => 
                                prev.map(acc => 
                                  acc.id === accident.id 
                                    ? { ...acc, status: 'Resolved' } 
                                    : acc
                                )
                              );
                            }}
                            style={{
                              padding: '8px 12px',
                              background: '#4caf50',
                              color: 'white',
                              border: 'none',
                              borderRadius: '5px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '5px',
                              fontSize: '12px'
                            }}
                          >
                            <CheckCircle size={14} />
                            Resolve
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {accidents.length === 0 && (
                  <div style={{
                    textAlign: 'center',
                    padding: '50px',
                    color: theme.neutral
                  }}>
                    <AlertTriangle size={48} style={{ marginBottom: '20px', opacity: 0.3 }} />
                    <p>No accidents reported today</p>
                  </div>
                )}
              </div>

              {/* Accident Report Form Modal */}
              {showAccidentForm && (
                <div style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'rgba(0,0,0,0.5)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  zIndex: 1000
                }}>
                  <div style={{
                    background: theme.background,
                    borderRadius: '15px',
                    padding: '30px',
                    width: '100%',
                    maxWidth: '500px',
                    boxShadow: '0 20px 60px rgba(0,0,0,0.3)'
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '25px'
                    }}>
                      <h3 style={{ color: theme.primary, margin: 0 }}>Report New Accident</h3>
                      <button
                        onClick={() => setShowAccidentForm(false)}
                        style={{
                          background: 'none',
                          border: 'none',
                          fontSize: '24px',
                          cursor: 'pointer',
                          color: theme.neutral
                        }}
                      >
                        <XCircle size={24} />
                      </button>
                    </div>

                    <form onSubmit={handleAccidentSubmit}>
                      <div style={{ marginBottom: '20px' }}>
                        <label style={{
                          display: 'block',
                          marginBottom: '8px',
                          color: theme.darkText,
                          fontWeight: 'bold'
                        }}>
                          Location / Cross Path
                        </label>
                        <select
                          value={newAccident.location}
                          onChange={(e) => setNewAccident(prev => ({...prev, location: e.target.value}))}
                          style={{
                            width: '100%',
                            padding: '12px',
                            border: `2px solid ${theme.primary}20`,
                            borderRadius: '8px',
                            fontSize: '16px',
                            boxSizing: 'border-box'
                          }}
                          required
                        >
                          <option value="">Select Location</option>
                          {crossPaths.map(path => (
                            <option key={path.id} value={path.name}>
                              {path.name}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div style={{ marginBottom: '20px' }}>
                        <label style={{
                          display: 'block',
                          marginBottom: '8px',
                          color: theme.darkText,
                          fontWeight: 'bold'
                        }}>
                          Severity Level
                        </label>
                        <select
                          value={newAccident.severity}
                          onChange={(e) => setNewAccident(prev => ({...prev, severity: e.target.value}))}
                          style={{
                            width: '100%',
                            padding: '12px',
                            border: `2px solid ${theme.primary}20`,
                            borderRadius: '8px',
                            fontSize: '16px',
                            boxSizing: 'border-box'
                          }}
                        >
                          <option value="Low">Low - Minor incident</option>
                          <option value="Medium">Medium - Traffic disruption</option>
                          <option value="High">High - Major accident</option>
                        </select>
                      </div>

                      <div style={{ marginBottom: '25px' }}>
                        <label style={{
                          display: 'block',
                          marginBottom: '8px',
                          color: theme.darkText,
                          fontWeight: 'bold'
                        }}>
                          Notes & Description
                        </label>
                        <textarea
                          value={newAccident.notes}
                          onChange={(e) => setNewAccident(prev => ({...prev, notes: e.target.value}))}
                          placeholder="Describe the accident details, vehicles involved, current status, etc."
                          style={{
                            width: '100%',
                            minHeight: '100px',
                            padding: '12px',
                            border: `2px solid ${theme.primary}20`,
                            borderRadius: '8px',
                            fontSize: '16px',
                            boxSizing: 'border-box',
                            resize: 'vertical'
                          }}
                          required
                        />
                      </div>

                      <div style={{ display: 'flex', gap: '15px', justifyContent: 'flex-end' }}>
                        <button
                          type="button"
                          onClick={() => setShowAccidentForm(false)}
                          style={{
                            padding: '12px 20px',
                            background: theme.neutral,
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: 'pointer'
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          type="submit"
                          style={{
                            padding: '12px 20px',
                            background: theme.secondary,
                            color: 'white',
                            border: 'none',
                            borderRadius: '8px',
                            cursor: 'pointer'
                          }}
                        >
                          Submit Report
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default SmartTrafficSystem;

