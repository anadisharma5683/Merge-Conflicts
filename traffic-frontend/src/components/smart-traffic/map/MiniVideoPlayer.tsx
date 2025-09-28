'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize2 } from 'lucide-react';
import { Theme } from '@/types/smart-traffic';

interface MiniVideoPlayerProps {
  videoUrl?: string;
  liveStreamUrl?: string;
  crossPathId: number;
  theme: Theme;
  isEnabled?: boolean;
}

export default function MiniVideoPlayer({ 
  videoUrl, 
  liveStreamUrl, 
  crossPathId, 
  theme, 
  isEnabled = true 
}: MiniVideoPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Priority: liveStreamUrl > videoUrl > placeholder
  const activeVideoUrl = liveStreamUrl || videoUrl;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !activeVideoUrl) return;

    const handleLoadStart = () => setIsLoading(true);
    const handleCanPlay = () => setIsLoading(false);
    const handleError = () => {
      setVideoError(true);
      setIsLoading(false);
      console.error('Video failed to load:', activeVideoUrl);
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('loadstart', handleLoadStart);
    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('error', handleError);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('loadstart', handleLoadStart);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('error', handleError);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [activeVideoUrl]);

  const handlePlayPause = async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      if (isPlaying) {
        video.pause();
      } else {
        await video.play();
      }
    } catch (error) {
      console.error('Error playing video:', error);
      setVideoError(true);
    }
  };

  const handleMuteToggle = () => {
    const video = videoRef.current;
    if (!video) return;
    
    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;
    
    if (!isFullscreen) {
      video.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setIsFullscreen(!isFullscreen);
  };

  if (!isEnabled || !activeVideoUrl) {
    return (
      <div style={{
        width: '100%',
        height: '200px',
        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
        borderRadius: '10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        position: 'relative',
        border: `2px solid ${theme.primary}20`
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '30px', marginBottom: '10px' }}>📹</div>
          <div style={{ fontSize: '16px', fontWeight: '500' }}>Live Traffic Feed</div>
          <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '5px' }}>
            Cross Path {crossPathId}
          </div>
          <div style={{ fontSize: '11px', opacity: 0.5, marginTop: '8px' }}>
            {!isEnabled ? 'Video disabled' : 'Video feed not available'}
          </div>
          {videoUrl && (
            <div style={{ fontSize: '10px', opacity: 0.4, marginTop: '4px' }}>
              Expected: {videoUrl}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (videoError) {
    return (
      <div style={{
        width: '100%',
        height: '200px',
        background: 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
        borderRadius: '10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        position: 'relative',
        border: `2px solid ${theme.primary}20`
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '30px', marginBottom: '10px' }}>⚠️</div>
          <div style={{ fontSize: '16px', fontWeight: '500' }}>Video Load Error</div>
          <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '5px' }}>
            Cross Path {crossPathId}
          </div>
          <div style={{ fontSize: '11px', opacity: 0.8, marginTop: '8px' }}>
            Could not load: {activeVideoUrl}
          </div>
          <button
            onClick={() => {
              setVideoError(false);
              setIsLoading(true);
              if (videoRef.current) {
                videoRef.current.load();
              }
            }}
            style={{
              marginTop: '10px',
              padding: '6px 12px',
              background: 'rgba(255,255,255,0.2)',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer',
              fontSize: '11px'
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
      <div style={{
        width: '100%',
        height: '200px',
        background: '#000',
        borderRadius: '10px',
        position: 'relative',
        overflow: 'hidden',
        border: `2px solid ${theme.primary}20`
      }}>
      {/* Video Element */}
      <video
        ref={videoRef}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover'
        }}
        src={activeVideoUrl}
        muted={isMuted}
        autoPlay
        loop
        poster="/images/video-placeholder.jpg"
        onLoadStart={() => setIsLoading(true)}
        onCanPlay={() => setIsLoading(false)}
        onError={() => setVideoError(true)}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />
      
      {/* Loading Indicator */}
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '10px',
          background: 'rgba(0,0,0,0.5)',
          padding: '20px',
          borderRadius: '8px'
        }}>
          <div style={{
            width: '30px',
            height: '30px',
            border: '3px solid rgba(255,255,255,0.3)',
            borderTop: `3px solid ${theme.primary}`,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <div style={{ fontSize: '12px' }}>Loading video...</div>
        </div>
      )}

      {/* Video Controls Overlay */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
        padding: '10px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        {/* Play/Pause Button */}
        <button
          onClick={handlePlayPause}
          style={{
            background: 'rgba(255,255,255,0.2)',
            border: 'none',
            borderRadius: '50%',
            width: '35px',
            height: '35px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: 'white',
            transition: 'all 0.3s ease'
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.3)';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.2)';
          }}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>

        {/* Volume Button */}
        <button
          onClick={handleMuteToggle}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'white',
            opacity: 0.8,
            transition: 'opacity 0.3s ease'
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.opacity = '1';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.opacity = '0.8';
          }}
        >
          {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
        </button>

        {/* Video Info */}
        <div style={{ flex: 1, color: 'white', fontSize: '12px' }}>
          <div style={{ fontWeight: '500' }}>
            {liveStreamUrl ? '🔴 LIVE' : '📹 RECORDED'}
          </div>
          <div style={{ opacity: 0.7 }}>Cross Path {crossPathId}</div>
        </div>

        {/* Fullscreen Button */}
        <button
          onClick={handleFullscreen}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'white',
            opacity: 0.8,
            transition: 'opacity 0.3s ease'
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.opacity = '1';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.opacity = '0.8';
          }}
        >
          <Maximize2 size={16} />
        </button>
      </div>

      {/* Live Indicator */}
      {liveStreamUrl && (
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          background: '#dc2626',
          color: 'white',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '11px',
          fontWeight: '600',
          display: 'flex',
          alignItems: 'center',
          gap: '5px'
        }}>
          <div style={{
            width: '6px',
            height: '6px',
            background: 'white',
            borderRadius: '50%',
            animation: 'pulse 2s infinite'
          }} />
          LIVE
        </div>
      )}
    </div>
    </>
  );
}