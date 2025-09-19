'use client';

import { useState, useEffect } from 'react';

// Define the type for the vehicle counts data
interface VehicleCounts {
  car: number;
  motorcycle: number;
  bus: number;
  truck: number;
  total: number;
}

// Define the type for the backend status response
interface BackendStatus {
  counters: VehicleCounts;
  is_playing: boolean;
  frame_skip: number;
  current_frame: number;
}

export default function TrafficAnalysisApp() {
  // Authentication state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentPage, setCurrentPage] = useState('home');
  
  // Login form state
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  // Traffic analysis state
  const [videoStarted, setVideoStarted] = useState(false);
  const [counts, setCounts] = useState<VehicleCounts>({
    car: 0,
    motorcycle: 0,
    bus: 0,
    truck: 0,
    total: 0,
  });
  const [isPlaying, setIsPlaying] = useState(true);
  const [frameSkip, setFrameSkip] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const backendUrl = 'http://127.0.0.1:5000';

  // Authentication functions
  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (username === '1' && password === '1') {
      setIsLoggedIn(true);
      setCurrentPage('dashboard');
      setLoginError('');
    } else {
      setLoginError('Invalid credentials. Use username: 1, password: 1');
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setCurrentPage('home');
    setVideoStarted(false);
    setUsername('');
    setPassword('');
  };

  // Traffic analysis functions
  const startAnalysis = () => {
    setVideoStarted(true);
    setError(null);
  };

  const playNewVideo = async () => {
    try {
      setIsLoading(true);
      await resetCounters();
      setVideoStarted(true);
      setError(null);
    } catch (error) {
      console.error('Error starting new video:', error);
      setError('Failed to start new video');
    } finally {
      setIsLoading(false);
    }
  };

  const restartVideo = async () => {
    try {
      setIsLoading(true);
      await resetCounters();
      setCurrentFrame(0);
      setError(null);
    } catch (error) {
      console.error('Error restarting video:', error);
      setError('Failed to restart video');
    } finally {
      setIsLoading(false);
    }
  };

  const togglePlayPause = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/play_pause`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (response.ok) {
        const data = await response.json();
        setIsPlaying(data.is_playing);
        setError(null);
      } else {
        throw new Error('Failed to toggle play/pause');
      }
    } catch (error) {
      console.error('Error toggling play/pause:', error);
      setError('Failed to toggle playback');
    } finally {
      setIsLoading(false);
    }
  };

  const resetCounters = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/reset_counters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (response.ok) {
        const data = await response.json();
        setCounts(data.counters);
        setCurrentFrame(0);
        setError(null);
      } else {
        throw new Error('Failed to reset counters');
      }
    } catch (error) {
      console.error('Error resetting counters:', error);
      setError('Failed to reset counters');
    } finally {
      setIsLoading(false);
    }
  };

  const updateFrameSkip = async (newSkipValue: number) => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/set_frame_skip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_value: newSkipValue }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setFrameSkip(data.frame_skip);
        setError(null);
      } else {
        throw new Error('Failed to update frame skip');
      }
    } catch (error) {
      console.error('Error updating frame skip:', error);
      setError('Failed to update frame skip');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (videoStarted && isLoggedIn) {
      const intervalId = setInterval(() => {
        fetch(`${backendUrl}/get_counts`)
          .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
          })
          .then((data: BackendStatus) => {
            setCounts(data.counters);
            setIsPlaying(data.is_playing);
            setFrameSkip(data.frame_skip);
            setCurrentFrame(data.current_frame);
            setError(null);
          })
          .catch(error => {
            console.error('Error fetching counts:', error);
            setError('Connection error. Please check if backend is running.');
          });
      }, 1000);

      return () => clearInterval(intervalId);
    }
  }, [videoStarted, isLoggedIn, backendUrl]);

  // Style objects
  const containerStyle: React.CSSProperties = {
    padding: '20px',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    minHeight: '100vh',
    color: 'white'
  };

  const cardStyle: React.CSSProperties = {
    maxWidth: '1200px',
    margin: '0 auto',
    background: 'rgba(255, 255, 255, 0.1)',
    borderRadius: '20px',
    padding: '30px',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255, 255, 255, 0.2)'
  };

  const buttonStyle: React.CSSProperties = {
    background: 'linear-gradient(45deg, #4caf50, #8bc34a)',
    border: 'none',
    borderRadius: '25px',
    padding: '12px 24px',
    fontSize: '16px',
    color: 'white',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
    margin: '5px'
  };

  // Header Component
  const Header = () => (
    <header style={{
      background: 'rgba(0, 0, 0, 0.3)',
      padding: '15px 30px',
      borderRadius: '15px',
      marginBottom: '30px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap'
    }}>
      <h1 style={{ margin: '0', fontSize: '1.8em' }}>🚗 Traffic Analyzer Pro</h1>
      {isLoggedIn && (
        <nav style={{ display: 'flex', gap: '15px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button 
            onClick={() => setCurrentPage('dashboard')}
            style={{
              ...buttonStyle,
              background: currentPage === 'dashboard' ? 'linear-gradient(45deg, #2196f3, #21cbf3)' : 'rgba(255, 255, 255, 0.2)',
              fontSize: '14px',
              padding: '8px 16px'
            }}
          >
            Dashboard
          </button>
          <button 
            onClick={() => setCurrentPage('about')}
            style={{
              ...buttonStyle,
              background: currentPage === 'about' ? 'linear-gradient(45deg, #2196f3, #21cbf3)' : 'rgba(255, 255, 255, 0.2)',
              fontSize: '14px',
              padding: '8px 16px'
            }}
          >
            About
          </button>
          <button 
            onClick={() => setCurrentPage('howto')}
            style={{
              ...buttonStyle,
              background: currentPage === 'howto' ? 'linear-gradient(45deg, #2196f3, #21cbf3)' : 'rgba(255, 255, 255, 0.2)',
              fontSize: '14px',
              padding: '8px 16px'
            }}
          >
            How to Use
          </button>
          <button 
            onClick={handleLogout}
            style={{
              ...buttonStyle,
              background: 'linear-gradient(45deg, #f44336, #e91e63)',
              fontSize: '14px',
              padding: '8px 16px'
            }}
          >
            Logout
          </button>
        </nav>
      )}
    </header>
  );

  // Footer Component
  const Footer = () => (
    <footer style={{
      background: 'rgba(0, 0, 0, 0.3)',
      padding: '20px',
      borderRadius: '15px',
      marginTop: '30px',
      textAlign: 'center',
      borderTop: '1px solid rgba(255, 255, 255, 0.2)'
    }}>
      <p style={{ margin: '0', color: '#ccc' }}>
        © 2025 Traffic Analyzer Pro | Real-time Vehicle Detection & Counting
      </p>
      <p style={{ margin: '10px 0 0 0', fontSize: '14px', color: '#aaa' }}>
        Powered by AI Computer Vision Technology
      </p>
    </footer>
  );

  // Home Page (Landing Page)
  const HomePage = () => (
    <div style={{ textAlign: 'center', padding: '40px 20px' }}>
      <div style={{ fontSize: '4em', marginBottom: '20px' }}>🚗🚙🚌</div>
      <h1 style={{ fontSize: '3em', marginBottom: '30px', textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)' }}>
        Welcome to Traffic Analyzer Pro
      </h1>
      <p style={{ fontSize: '1.3em', marginBottom: '40px', maxWidth: '800px', margin: '0 auto 40px' }}>
        Advanced real-time traffic monitoring and vehicle counting system powered by computer vision and AI. 
        Monitor traffic flow, count different vehicle types, and analyze patterns with precision and ease.
      </p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '30px', marginBottom: '40px' }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '15px',
          padding: '30px',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <div style={{ fontSize: '3em', marginBottom: '15px' }}>🎯</div>
          <h3>Accurate Detection</h3>
          <p>Advanced AI algorithms for precise vehicle identification and counting</p>
        </div>
        
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '15px',
          padding: '30px',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <div style={{ fontSize: '3em', marginBottom: '15px' }}>⚡</div>
          <h3>Real-time Processing</h3>
          <p>Live video analysis with instant results and responsive controls</p>
        </div>
        
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '15px',
          padding: '30px',
          border: '1px solid rgba(255, 255, 255, 0.2)'
        }}>
          <div style={{ fontSize: '3em', marginBottom: '15px' }}>📊</div>
          <h3>Detailed Analytics</h3>
          <p>Comprehensive statistics for cars, motorcycles, buses, and trucks</p>
        </div>
      </div>

      <button 
        onClick={() => setCurrentPage('login')}
        style={{
          ...buttonStyle,
          fontSize: '20px',
          padding: '20px 40px',
          background: 'linear-gradient(45deg, #ff6b6b, #ffa726)'
        }}
      >
        🚀 Get Started - Login
      </button>
    </div>
  );

  // Login Page
  const LoginPage = () => (
    <div style={{ maxWidth: '400px', margin: '0 auto', textAlign: 'center', padding: '40px 20px' }}>
      <div style={{ fontSize: '4em', marginBottom: '30px' }}>🔐</div>
      <h2 style={{ marginBottom: '30px', fontSize: '2.5em' }}>Login to Continue</h2>
      
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            style={{
              width: '100%',
              padding: '15px',
              borderRadius: '10px',
              border: '1px solid rgba(255, 255, 255, 0.3)',
              background: 'rgba(255, 255, 255, 0.1)',
              color: 'white',
              fontSize: '16px',
              boxSizing: 'border-box'
            }}
            required
          />
        </div>
        
        <div>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{
              width: '100%',
              padding: '15px',
              borderRadius: '10px',
              border: '1px solid rgba(255, 255, 255, 0.3)',
              background: 'rgba(255, 255, 255, 0.1)',
              color: 'white',
              fontSize: '16px',
              boxSizing: 'border-box'
            }}
            required
          />
        </div>
        
        {loginError && (
          <div style={{ color: '#ffcdd2', background: 'rgba(244, 67, 54, 0.2)', padding: '10px', borderRadius: '5px' }}>
            {loginError}
          </div>
        )}
        
        <button type="submit" style={{...buttonStyle, width: '100%', padding: '15px'}}>
          🚀 Login
        </button>
      </form>
      
      <div style={{ marginTop: '30px', padding: '20px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '10px' }}>
        <p style={{ margin: '0', fontSize: '14px', color: '#ccc' }}>
          <strong>Demo Credentials:</strong><br/>
          Username: 1<br/>
          Password: 1
        </p>
      </div>
      
      <button 
        onClick={() => setCurrentPage('home')}
        style={{
          ...buttonStyle,
          background: 'rgba(255, 255, 255, 0.2)',
          marginTop: '20px'
        }}
      >
        ← Back to Home
      </button>
    </div>
  );

  // About Page
  const AboutPage = () => (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', fontSize: '2.5em', marginBottom: '30px' }}>
        📋 About Traffic Analyzer Pro
      </h2>
      
      <div style={{ fontSize: '1.1em', lineHeight: '1.6' }}>
        <p>
          Traffic Analyzer Pro is a cutting-edge computer vision application that provides real-time traffic monitoring 
          and vehicle counting capabilities. Built with advanced AI algorithms, it can accurately detect and classify 
          different types of vehicles in live video streams.
        </p>
        
        <h3 style={{ color: '#ffa726', marginTop: '30px' }}>🎯 Key Features</h3>
        <p>
          • Real-time vehicle detection and counting<br/>
          • Support for multiple vehicle types (Cars, Motorcycles, Buses, Trucks)<br/>
          • Live video streaming with analysis overlay<br/>
          • Performance optimization controls<br/>
          • Interactive playback controls (Play/Pause/Reset)<br/>
          • Detailed statistics and counters
        </p>
        
        <h3 style={{ color: '#ffa726', marginTop: '30px' }}>🔧 Technology Stack</h3>
        <p>
          • Frontend: React with TypeScript<br/>
          • Backend: Flask Python server<br/>
          • Computer Vision: OpenCV and AI models<br/>
          • Real-time Processing: Optimized frame processing<br/>
          • Modern UI: Responsive design with glassmorphism
        </p>
        
        <h3 style={{ color: '#ffa726', marginTop: '30px' }}>🎯 Use Cases</h3>
        <p>
          • Traffic flow monitoring<br/>
          • Road usage analysis<br/>
          • Vehicle counting for statistics<br/>
          • Smart city applications<br/>
          • Research and development
        </p>
      </div>
    </div>
  );

  // How to Use Page
  const HowToUsePage = () => (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h2 style={{ textAlign: 'center', fontSize: '2.5em', marginBottom: '30px' }}>
        📖 How to Use
      </h2>
      
      <div style={{ fontSize: '1.1em', lineHeight: '1.6' }}>
        <div style={{ background: 'rgba(255, 255, 255, 0.1)', padding: '20px', borderRadius: '15px', marginBottom: '30px' }}>
          <h3 style={{ color: '#4caf50', marginTop: '0' }}>🚀 Getting Started</h3>
          <p>
            1. <strong>Login:</strong> Use username 1 and password 1 to access the system<br/>
            2. <strong>Navigate:</strong> Go to Dashboard to start analyzing traffic<br/>
            3. <strong>Start Analysis:</strong> 
          </p>
        </div>
        
        <div style={{ background: 'rgba(255, 255, 255, 0.1)', padding: '20px', borderRadius: '15px', marginBottom: '30px' }}>
          <h3 style={{ color: '#2196f3', marginTop: '0' }}>🎮 Controls Guide</h3>
          <p>
            <strong>▶ Play/Pause:</strong> Control video playback<br/>
            <strong>🔄 Reset:</strong> Reset all counters to zero<br/>
            <strong>🎬 Play New Video:</strong> Start analysis with fresh counters<br/>
            <strong>🔄 Restart Video:</strong> Restart current video from beginning<br/>
            <strong>⚡ Frame Skip:</strong> Adjust processing speed (1-10 frames)
          </p>
        </div>
        
        <div style={{ background: 'rgba(255, 255, 255, 0.1)', padding: '20px', borderRadius: '15px', marginBottom: '30px' }}>
          <h3 style={{ color: '#ffa726', marginTop: '0' }}>📊 Understanding the Interface</h3>
          <p>
            <strong>Live Feed:</strong> Shows processed video with detection boxes<br/>
            <strong>Vehicle Counters:</strong> Real-time count for each vehicle type<br/>
            <strong>Status Panel:</strong> Shows current playback status and frame info<br/>
            <strong>Performance Control:</strong> Frame skip slider for optimization
          </p>
        </div>
        
        <div style={{ background: 'rgba(255, 255, 255, 0.1)', padding: '20px', borderRadius: '15px' }}>
          <h3 style={{ color: '#e91e63', marginTop: '0' }}>⚠ Important Notes</h3>
          <p>
            • Ensure the backend server is running on http://127.0.0.1:5000<br/>
            • Higher frame skip values increase speed but may reduce accuracy<br/>
            • The system works best with clear, well-lit video footage<br/>
            • Refresh the page if you encounter connection issues
          </p>
        </div>
      </div>
    </div>
  );

  // Dashboard (Main Traffic Analysis)
  const Dashboard = () => (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap' }}>
        <h2 style={{ margin: '0', fontSize: '2em' }}>🎯 Traffic Analysis Dashboard</h2>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={playNewVideo}
            disabled={isLoading}
            style={{
              ...buttonStyle,
              background: 'linear-gradient(45deg, #4caf50, #8bc34a)',
              opacity: isLoading ? 0.6 : 1
            }}
          >
            {isLoading ? '⏳' : '🎬 Play New Video'}
          </button>
          <button
            onClick={restartVideo}
            disabled={isLoading}
            style={{
              ...buttonStyle,
              background: 'linear-gradient(45deg, #ff9800, #ffc107)',
              opacity: isLoading ? 0.6 : 1
            }}
          >
            {isLoading ? '⏳' : '🔄 Restart Video'}
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          background: 'rgba(244, 67, 54, 0.2)',
          border: '1px solid rgba(244, 67, 54, 0.5)',
          borderRadius: '10px',
          padding: '15px',
          margin: '20px 0',
          color: '#ffcdd2',
          textAlign: 'center'
        }}>
          ⚠ {error}
        </div>
      )}

      {!videoStarted ? (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div style={{ fontSize: '4em', marginBottom: '20px' }}>🎥</div>
          <p style={{ fontSize: '1.2em', marginBottom: '30px' }}>
            Ready to analyze traffic? Click below to start the video analysis stream.
          </p>
          <button 
            onClick={startAnalysis}
            style={{
              ...buttonStyle,
              fontSize: '18px',
              padding: '15px 30px',
              background: 'linear-gradient(45deg, #4caf50, #8bc34a)'
            }}
          >
            🚀 Start Analysis
          </button>
        </div>
      ) : (
        <div>
          {/* Video and Controls Section */}
          <div style={{ 
            display: 'flex', 
            gap: '30px', 
            marginBottom: '30px',
            flexWrap: 'wrap'
          }}>
            {/* Video Feed */}
            <div style={{ flex: '2', minWidth: '400px' }}>
              <h3 style={{ marginBottom: '20px' }}>📹 Live Feed from Server</h3>
              <img
                src={`${backendUrl}/video_feed`}
                alt="Live Traffic Analysis"
                style={{
                  width: '100%',
                  maxWidth: '800px',
                  height: 'auto',
                  border: '3px solid rgba(255, 255, 255, 0.3)',
                  borderRadius: '15px',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                }}
              />
            </div>

            {/* Controls Panel */}
            <div style={{
              flex: '1',
              minWidth: '300px',
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '25px',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <h3 style={{ marginBottom: '20px' }}>🎮 Controls</h3>
              
              {/* Play/Pause and Reset Buttons */}
              <div style={{ marginBottom: '25px' }}>
                <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', flexWrap: 'wrap' }}>
                  <button
                    onClick={togglePlayPause}
                    disabled={isLoading}
                    style={{
                      background: isPlaying 
                        ? 'linear-gradient(45deg, #ff9800, #ffc107)' 
                        : 'linear-gradient(45deg, #4caf50, #8bc34a)',
                      border: 'none',
                      borderRadius: '25px',
                      padding: '12px 24px',
                      color: 'white',
                      fontWeight: 'bold',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s ease',
                      minWidth: '120px',
                      opacity: isLoading ? 0.6 : 1
                    }}
                  >
                    {isLoading ? '⏳' : (isPlaying ? '⏸ Pause' : '▶ Play')}
                  </button>
                  
                  <button
                    onClick={resetCounters}
                    disabled={isLoading}
                    style={{
                      background: 'linear-gradient(45deg, #f44336, #e91e63)',
                      border: 'none',
                      borderRadius: '25px',
                      padding: '12px 24px',
                      color: 'white',
                      fontWeight: 'bold',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s ease',
                      minWidth: '120px',
                      opacity: isLoading ? 0.6 : 1
                    }}
                  >
                    {isLoading ? '⏳' : '🔄 Reset'}
                  </button>
                </div>
              </div>

              {/* Frame Skip Control */}
              <div style={{ marginBottom: '25px' }}>
                <h4 style={{ marginBottom: '10px' }}>⚡ Performance Control</h4>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                  <span style={{ minWidth: '80px' }}>Frame Skip:</span>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={frameSkip}
                    onChange={(e) => updateFrameSkip(parseInt(e.target.value))}
                    style={{
                      flex: 1,
                      minWidth: '100px',
                      height: '8px',
                      borderRadius: '5px',
                      background: 'rgba(255, 255, 255, 0.3)',
                      outline: 'none'
                    }}
                  />
                  <span style={{ 
                    minWidth: '20px', 
                    fontWeight: 'bold',
                    color: '#ffa726'
                  }}>
                    {frameSkip}
                  </span>
                </div>
                <small style={{ color: '#ccc' }}>
                  Higher values = faster processing, lower accuracy
                </small>
              </div>

              {/* Status Display */}
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '10px',
                padding: '15px'
              }}>
                <h4 style={{ marginBottom: '10px' }}>📊 Status</h4>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span>Playback:</span>
                  <span style={{ 
                    fontWeight: 'bold', 
                    color: isPlaying ? '#4caf50' : '#ff9800' 
                  }}>
                    {isPlaying ? 'Playing' : 'Paused'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span>Frame:</span>
                  <span style={{ fontWeight: 'bold', color: '#ffa726' }}>
                    {currentFrame.toLocaleString()}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Skip Rate:</span>
                  <span style={{ fontWeight: 'bold', color: '#ffa726' }}>
                    1:{frameSkip}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Vehicle Counts Display */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '20px',
            marginBottom: '30px'
          }}>
            {/* Car Counter */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>🚗</div>
              <h3>Cars</h3>
              <p style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#4caf50' }}>
                {counts.car}
              </p>
            </div>

            {/* Motorcycle Counter */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>🏍</div>
              <h3>Motorcycles</h3>
              <p style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2196f3' }}>
                {counts.motorcycle}
              </p>
            </div>

            {/* Bus Counter */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>🚌</div>
              <h3>Buses</h3>
              <p style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#ff9800' }}>
                {counts.bus}
              </p>
            </div>

            {/* Truck Counter */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '1px solid rgba(255, 255, 255, 0.2)'
            }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>🚚</div>
              <h3>Trucks</h3>
              <p style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#e91e63' }}>
                {counts.truck}
              </p>
            </div>

            {/* Total Counter */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '15px',
              padding: '20px',
              textAlign: 'center',
              border: '2px solid rgba(255, 255, 255, 0.4)'
            }}>
              <div style={{ fontSize: '2em', marginBottom: '10px' }}>📊</div>
              <h3>Total Vehicles</h3>
              <p style={{ fontSize: '2em', fontWeight: 'bold', color: '#ffa726' }}>
                {counts.total}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // Main Render Switch
  return (
    <div style={containerStyle}>
      <Header />
      <div style={cardStyle}>
        {currentPage === 'home' && <HomePage />}
        {currentPage === 'login' && <LoginPage />}
        {currentPage === 'dashboard' && isLoggedIn && <Dashboard />}
        {currentPage === 'about' && <AboutPage />}
        {currentPage === 'howto' && <HowToUsePage />}
        {!['home','login','dashboard','about','howto'].includes(currentPage) && (
          <div style={{ textAlign: 'center', padding: '50px' }}>
            <h2>404 - Page Not Found</h2>
          </div>
        )}
      </div>
      <Footer />
    </div>
  );
}