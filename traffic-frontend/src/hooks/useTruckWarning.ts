import { useState, useEffect, useCallback } from 'react';
import { TruckDetection, TruckWarning, TruckWarningAction } from '@/types/smart-traffic';

export const useTruckWarning = () => {
  const [activeWarnings, setActiveWarnings] = useState<TruckWarning[]>([]);
  const [detectionHistory, setDetectionHistory] = useState<TruckDetection[]>([]);

  // Check if current time is within restricted hours (11 PM to 6 AM)
  const isRestrictedHours = useCallback((): boolean => {
    const now = new Date();
    const currentHour = now.getHours();
    
    // Restricted hours: 23:00 (11 PM) to 06:00 (6 AM)
    return currentHour >= 23 || currentHour < 6;
  }, []);

  // Simulate truck detection (in real app, this would come from backend)
  const simulateTruckDetection = useCallback((): TruckDetection => {
    const sections = ['North Junction', 'South Junction', 'East Junction', 'West Junction'];
    const randomSection = sections[Math.floor(Math.random() * sections.length)];
    
    return {
      id: `truck_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      location: `Section ${Math.floor(Math.random() * 4) + 1}`,
      section: randomSection,
      isRestricted: isRestrictedHours(),
      status: 'active',
    };
  }, [isRestrictedHours]);

  // Create warning actions
  const createWarningActions = useCallback((): TruckWarningAction[] => [
    {
      id: 'inform_officers',
      label: 'Inform Nearby Officers',
      type: 'inform',
      color: 'blue',
    },
    {
      id: 'permit_vehicle',
      label: 'Mark as Permitted',
      type: 'permit',
      color: 'green',
    },
    {
      id: 'dismiss_warning',
      label: 'Dismiss Warning',
      type: 'dismiss',
      color: 'red',
    },
  ], []);

  // Create warning from detection
  const createWarning = useCallback((detection: TruckDetection): TruckWarning => {
    const timeStr = detection.timestamp.toLocaleTimeString();
    
    return {
      id: `warning_${detection.id}`,
      detection,
      message: `Truck detected in ${detection.section} at ${timeStr}. Vehicle found during restricted hours (11 PM - 6 AM). Please take appropriate action.`,
      actions: createWarningActions(),
      createdAt: new Date(),
    };
  }, [createWarningActions]);

  // Handle warning action
  const handleWarningAction = useCallback((warningId: string, actionType: string, notes?: string) => {
    setActiveWarnings(prev => prev.map(warning => {
      if (warning.id === warningId) {
        const status: 'active' | 'acknowledged' | 'permitted' = 
          actionType === 'permit' ? 'permitted' : 'acknowledged';
        
        const updatedDetection: TruckDetection = {
          ...warning.detection,
          status,
          notes: notes || `Action taken: ${actionType}`,
        };

        return {
          ...warning,
          detection: updatedDetection,
          resolvedAt: new Date(),
          resolvedBy: 'Officer', // In real app, get from auth context
        };
      }
      return warning;
    }));

    // Move to history after a delay
    setTimeout(() => {
      setActiveWarnings(prev => prev.filter(w => w.id !== warningId));
    }, 3000);

    console.log(`Truck warning ${warningId} resolved with action: ${actionType}`);
  }, []);

  // Check for truck detections (simulate polling backend)
  useEffect(() => {
    const checkForTrucks = () => {
      // Simulate random truck detection during restricted hours
      if (isRestrictedHours() && Math.random() < 0.1) { // 10% chance every 10 seconds
        const detection = simulateTruckDetection();
        
        // Add to history
        setDetectionHistory(prev => [detection, ...prev.slice(0, 49)]); // Keep last 50
        
        // Create warning if during restricted hours
        if (detection.isRestricted) {
          const warning = createWarning(detection);
          setActiveWarnings(prev => {
            // Avoid duplicate warnings
            if (prev.some(w => w.detection.section === detection.section)) {
              return prev;
            }
            return [warning, ...prev];
          });
        }
      }
    };

    // Check every 10 seconds
    const interval = setInterval(checkForTrucks, 10000);
    
    return () => clearInterval(interval);
  }, [isRestrictedHours, simulateTruckDetection, createWarning]);

  // Simulate a truck detection for testing
  const triggerTestWarning = useCallback(() => {
    const detection = simulateTruckDetection();
    detection.isRestricted = true; // Force restricted for testing
    
    setDetectionHistory(prev => [detection, ...prev.slice(0, 49)]);
    
    const warning = createWarning(detection);
    setActiveWarnings(prev => [warning, ...prev]);
  }, [simulateTruckDetection, createWarning]);

  return {
    activeWarnings,
    detectionHistory,
    isRestrictedHours: isRestrictedHours(),
    handleWarningAction,
    triggerTestWarning, // For testing purposes
  };
};