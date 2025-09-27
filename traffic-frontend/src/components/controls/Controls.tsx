'use client';

import StatusPanel from '../status-panel/StatusPanel';

interface ControlsProps {
  isPlaying: boolean;
  isLoading: boolean;
  frameSkip: number;
  currentFrame: number;
  onTogglePlayPause: () => void;
  onResetCounters: () => void;
  onUpdateFrameSkip: (value: number) => void;
}

export default function Controls({ 
  isPlaying, 
  isLoading, 
  frameSkip, 
  currentFrame,
  onTogglePlayPause, 
  onResetCounters, 
  onUpdateFrameSkip 
}: ControlsProps) {
  return (
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
            onClick={onTogglePlayPause}
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
            onClick={onResetCounters}
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
            onChange={(e) => onUpdateFrameSkip(parseInt(e.target.value))}
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
      <StatusPanel 
        isPlaying={isPlaying}
        currentFrame={currentFrame}
        frameSkip={frameSkip}
      />
    </div>
  );
}