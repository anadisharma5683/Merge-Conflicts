'use client';

interface StatusPanelProps {
  isPlaying: boolean;
  currentFrame: number;
  frameSkip: number;
}

export default function StatusPanel({ isPlaying, currentFrame, frameSkip }: StatusPanelProps) {
  return (
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
  );
}