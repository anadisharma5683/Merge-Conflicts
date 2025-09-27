'use client';

import { useState } from 'react';
import { useTrafficData } from '@/hooks/useTrafficData';
import StartScreen from '../start-screen/StartScreen';
import VideoFeed from '../video-feed/VideoFeed';
import Controls from '../controls/Controls';
import VehicleCountsDisplay from '../vehicle-counts/VehicleCountsDisplay';

export default function Dashboard() {
  const [videoStarted, setVideoStarted] = useState(false);
  const {
    counts,
    isPlaying,
    frameSkip,
    currentFrame,
    isLoading,
    error,
    backendUrl,
    togglePlayPause,
    resetCounters,
    updateFrameSkip,
  } = useTrafficData(videoStarted);

  const startAnalysis = () => {
    setVideoStarted(true);
  };

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
          <StartScreen onStartAnalysis={startAnalysis} />
        ) : (
          <div>
            {/* Video and Controls Section */}
            <div style={{ 
              display: 'flex', 
              gap: '30px', 
              marginBottom: '30px',
              flexWrap: 'wrap'
            }}>
              <VideoFeed backendUrl={backendUrl} />
              
              <Controls
                isPlaying={isPlaying}
                isLoading={isLoading}
                frameSkip={frameSkip}
                currentFrame={currentFrame}
                onTogglePlayPause={togglePlayPause}
                onResetCounters={resetCounters}
                onUpdateFrameSkip={updateFrameSkip}
              />
            </div>

            {/* Vehicle Counts Display */}
            <VehicleCountsDisplay counts={counts} />
          </div>
        )}
      </div>
    </div>
  );
}