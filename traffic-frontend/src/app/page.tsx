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

export default function Home() {
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

  const backendUrl = 'http://127.0.0.1:5000'; // आपके Flask सर्वर का URL

  const startAnalysis = () => {
    setVideoStarted(true);
    setError(null);
  };

  const togglePlayPause = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${backendUrl}/play_pause`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
        headers: {
          'Content-Type': 'application/json',
        },
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
        headers: {
          'Content-Type': 'application/json',
        },
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
    if (videoStarted) {
      // Fetch counts and status from the backend every 1 second
      const intervalId = setInterval(() => {
        fetch(`${backendUrl}/get_counts`)
          .then(response => {
            if (!response.ok) {
              throw new Error('Network response was not ok');
            }
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

      // Cleanup function to clear the interval when the component unmounts
      return () => clearInterval(intervalId);
    }
  }, [videoStarted, backendUrl]);

  return (
    <div style={{ 
      padding: '20px', 
      fontFamily: 'system-ui, -apple-system, sans-serif',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh',
      color: 'white'
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        background: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '20px',
        padding: '30px',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.2)'
      }}>
        <h1 style={{ 
          textAlign: 'center', 
          marginBottom: '30px',
          fontSize: '2.5em',
          textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)'
        }}>
          🚗 Real-Time Traffic Analysis
        </h1>

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
            ⚠️ {error}
          </div>
        )}

        {!videoStarted ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <p style={{ fontSize: '1.2em', marginBottom: '30px' }}>
              Click the button to start the video analysis stream from the backend.
            </p>
            <button 
              onClick={startAnalysis}
              style={{
                background: 'linear-gradient(45deg, #4caf50, #8bc34a)',
                border: 'none',
                borderRadius: '25px',
                padding: '15px 30px',
                fontSize: '18px',
                color: 'white',
                fontWeight: 'bold',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.3)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.2)';
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
                <h2 style={{ marginBottom: '20px' }}>📹 Live Feed from Server</h2>
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
                      {isLoading ? '⏳' : (isPlaying ? '⏸️ Pause' : '▶️ Play')}
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
            <div style={{ marginTop: '30px' }}>
              <h3 style={{ 
                marginBottom: '20px',
                textAlign: 'center',
                fontSize: '1.8em'
              }}>
                🚙 Vehicle Count Statistics
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                gap: '20px'
              }}>
                {[
                  { label: 'Total', count: counts.total, emoji: '🚗', color: '#ffa726' },
                  { label: 'Cars', count: counts.car, emoji: '🚙', color: '#4caf50' },
                  { label: 'Motorcycles', count: counts.motorcycle, emoji: '🏍️', color: '#2196f3' },
                  { label: 'Buses', count: counts.bus, emoji: '🚌', color: '#ff9800' },
                  { label: 'Trucks', count: counts.truck, emoji: '🚛', color: '#f44336' }
                ].map((item, index) => (
                  <div
                    key={index}
                    style={{
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: '15px',
                      padding: '20px',
                      textAlign: 'center',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      transition: 'transform 0.3s ease'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.transform = 'translateY(-5px)';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                    }}
                  >
                    <div style={{ fontSize: '2em', marginBottom: '10px' }}>
                      {item.emoji}
                    </div>
                    <div style={{
                      fontSize: '2.5em',
                      fontWeight: 'bold',
                      color: item.color,
                      marginBottom: '5px'
                    }}>
                      {item.count.toLocaleString()}
                    </div>
                    <div style={{
                      fontSize: '1em',
                      textTransform: 'capitalize',
                      color: '#ccc'
                    }}>
                      {item.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}