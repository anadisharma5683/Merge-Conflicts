'use client';

interface VideoFeedProps {
  backendUrl: string;
}

export default function VideoFeed({ backendUrl }: VideoFeedProps) {
  return (
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
  );
}