import React, { useEffect, useState } from 'react';
import { AlertTriangle, Users, CheckCircle, X, Clock, MapPin, Siren, ShieldAlert, Zap } from 'lucide-react';
import { TruckWarning } from '@/types/smart-traffic';
import { useWarningSound } from '@/hooks/useWarningSound';

interface TruckWarningNotificationProps {
  warning: TruckWarning;
  onAction: (warningId: string, actionType: string, notes?: string) => void;
}

const TruckWarningNotification: React.FC<TruckWarningNotificationProps> = ({
  warning,
  onAction,
}) => {
  const { playWarningSound, playActionSound } = useWarningSound();
  const [isVisible, setIsVisible] = useState(false);
  const [pulseCount, setPulseCount] = useState(0);

  // Animation and sound effects on mount
  useEffect(() => {
    setIsVisible(true);
    playWarningSound();
    
    // Pulse animation counter
    const pulseInterval = setInterval(() => {
      setPulseCount(prev => prev + 1);
    }, 1000);
    
    return () => clearInterval(pulseInterval);
  }, [playWarningSound]);

  const handleActionClick = (actionType: string) => {
    playActionSound();
    const notes = actionType === 'inform' ? 'Officers notified' : 
                  actionType === 'permit' ? 'Vehicle permitted' : 'Warning dismissed';
    onAction(warning.id, actionType, notes);
  };

  const getActionIcon = (actionType: string) => {
    const icons = { inform: Users, permit: CheckCircle, dismiss: X };
    const Icon = icons[actionType as keyof typeof icons];
    return Icon ? <Icon className="w-4 h-4" /> : null;
  };

  return (
    <div 
      className={`relative transform transition-all duration-500 ${
        isVisible ? 'translate-y-0 opacity-100 scale-100' : 'translate-y-4 opacity-0 scale-95'
      }`}
    >
      {/* Dramatic Background with Gradient */}
      <div className="relative bg-gradient-to-r from-red-600 via-red-500 to-orange-500 p-1 rounded-xl shadow-2xl mb-4">
        <div className="bg-red-50 rounded-lg">
          {/* Animated Border Effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20 animate-pulse rounded-lg"></div>
          
          {/* Pulsing Alert Overlay */}
          <div className={`absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-red-600 to-orange-500 rounded-t-lg animate-pulse`}></div>
          
          <div className="relative p-6">
            {/* Header Section with Dramatic Icons */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                {/* Animated Warning Icons */}
                <div className="relative">
                  <ShieldAlert className={`h-8 w-8 text-red-600 animate-bounce ${pulseCount % 2 === 0 ? 'scale-110' : 'scale-100'} transition-transform`} />
                  <Zap className="absolute -top-1 -right-1 h-4 w-4 text-yellow-500 animate-ping" />
                </div>
                <Siren className="h-6 w-6 text-red-600 animate-spin" />
              </div>
              
              <div className="flex items-center bg-red-100 px-3 py-1 rounded-full">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse mr-2"></div>
                <Clock className="w-4 h-4 text-red-600 mr-1" />
                <span className="text-sm font-mono text-red-700">
                  {warning.detection.timestamp.toLocaleTimeString()}
                </span>
              </div>
            </div>

            {/* Dramatic Title */}
            <div className="mb-4">
              <h2 className="text-2xl font-bold text-red-800 mb-2 animate-pulse">
                🚨 CRITICAL TRAFFIC VIOLATION DETECTED 🚨
              </h2>
              <div className="bg-gradient-to-r from-red-200 to-orange-200 p-3 rounded-lg border-l-4 border-red-600">
                <h3 className="text-lg font-semibold text-red-800 flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2 animate-bounce" />
                  UNAUTHORIZED TRUCK IN RESTRICTED ZONE
                </h3>
              </div>
            </div>

            {/* Location and Details */}
            <div className="bg-white p-4 rounded-lg border-2 border-red-200 mb-4">
              <div className="flex items-center mb-3">
                <MapPin className="w-5 h-5 text-red-600 mr-2" />
                <span className="font-semibold text-gray-800">
                  📍 {warning.detection.section} - {warning.detection.location}
                </span>
              </div>
              
              <div className="bg-red-50 p-3 rounded-md mb-3">
                <p className="text-red-800 font-medium text-sm leading-relaxed">
                  {warning.message}
                </p>
              </div>
              
              {/* Critical Time Zone Warning */}
              <div className="bg-gradient-to-r from-red-100 to-orange-100 p-4 rounded-lg border border-red-300">
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full animate-ping mr-2"></div>
                  <h4 className="text-red-800 font-bold text-sm">⚠️ RESTRICTED TIME ZONE VIOLATION</h4>
                </div>
                <p className="text-red-700 text-xs">
                  🕚 Heavy vehicles prohibited: 11:00 PM - 6:00 AM
                </p>
                <p className="text-red-600 text-xs mt-1 font-medium">
                  This is a SERIOUS traffic code violation requiring immediate officer response!
                </p>
              </div>
            </div>
            
            {/* Emergency Action Section */}
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-4 rounded-lg border-2 border-gray-300">
              <div className="flex items-center mb-3">
                <Siren className="w-5 h-5 text-blue-600 mr-2 animate-spin" />
                <h4 className="text-lg font-bold text-gray-800">
                  🚔 IMMEDIATE OFFICER ACTION REQUIRED
                </h4>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {warning.actions.map((action, index) => {
                  const delay = index * 100;
                  return (
                    <button
                      key={action.id}
                      onClick={() => handleActionClick(action.type)}
                      className={`group relative overflow-hidden transform transition-all duration-300 hover:scale-105 hover:shadow-xl
                        ${action.color === 'blue' ? 'bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800' :
                          action.color === 'green' ? 'bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800' :
                          'bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800'} 
                        text-white font-semibold py-3 px-4 rounded-lg border-2 border-white/20`}
                      style={{ animationDelay: `${delay}ms` }}
                    >
                      <div className="absolute inset-0 bg-white/10 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300"></div>
                      <div className="relative flex items-center justify-center space-x-2">
                        {getActionIcon(action.type)}
                        <span className="text-sm font-bold">{action.label}</span>
                      </div>
                      
                      {/* Button glow effect */}
                      <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 animate-pulse"></div>
                    </button>
                  );
                })}
              </div>
            </div>
            
            {/* Footer with ID and Timestamp */}
            <div className="mt-4 flex justify-between items-center text-xs text-red-600 bg-red-50 p-2 rounded">
              <span className="font-mono">ID: {warning.detection.id}</span>
              <span className="font-medium">
                Generated: {new Date().toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TruckWarningNotification;