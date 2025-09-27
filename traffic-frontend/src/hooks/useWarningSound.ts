import { useCallback, useRef } from 'react';

export const useWarningSound = () => {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const playWarningSound = useCallback(() => {
    // Create audio context for warning sound
    if (typeof window !== 'undefined') {
      try {
        // Create a synthetic warning sound using Web Audio API
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        
        // Create oscillators for siren-like sound
        const oscillator1 = audioContext.createOscillator();
        const oscillator2 = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        // Configure the sound
        oscillator1.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator1.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.5);
        oscillator1.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 1);
        
        oscillator2.frequency.setValueAtTime(600, audioContext.currentTime);
        oscillator2.frequency.exponentialRampToValueAtTime(300, audioContext.currentTime + 0.5);
        oscillator2.frequency.exponentialRampToValueAtTime(600, audioContext.currentTime + 1);
        
        oscillator1.type = 'square';
        oscillator2.type = 'sine';
        
        // Volume control
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.3, audioContext.currentTime + 0.1);
        gainNode.gain.exponentialRampToValueAtTime(0.1, audioContext.currentTime + 0.9);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
        
        // Connect nodes
        oscillator1.connect(gainNode);
        oscillator2.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Play the sound
        oscillator1.start(audioContext.currentTime);
        oscillator2.start(audioContext.currentTime);
        oscillator1.stop(audioContext.currentTime + 1);
        oscillator2.stop(audioContext.currentTime + 1);
        
      } catch (error) {
        console.warn('Could not play warning sound:', error);
      }
    }
  }, []);

  const playActionSound = useCallback(() => {
    if (typeof window !== 'undefined') {
      try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.frequency.setValueAtTime(1000, audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.1);
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
        
      } catch (error) {
        console.warn('Could not play action sound:', error);
      }
    }
  }, []);

  return {
    playWarningSound,
    playActionSound
  };
};