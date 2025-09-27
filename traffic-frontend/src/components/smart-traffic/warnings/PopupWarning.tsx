import React, { useEffect } from 'react';
import { 
  AlertTriangle, 
  AlertOctagon, 
  Bell, 
  CheckCircle, 
  XCircle, 
  Hourglass,
  MapPin,
  Clock,
  Siren
} from 'lucide-react';
import Modal from './Modal';

interface PopupWarningProps {
  warning: any;
  isOpen: boolean;
  onClose: () => void;
  onAction: (id: string, action: string) => void;
}

const PopupWarning: React.FC<PopupWarningProps> = ({ warning, isOpen, onClose, onAction }) => {
  // Auto-play sound when popup opens
  useEffect(() => {
    if (isOpen && warning) {
      // Play warning sound
      try {
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        const oscillator1 = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator1.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator1.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.5);
        oscillator1.type = 'square';
        
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.3, audioContext.currentTime + 0.1);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
        
        oscillator1.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator1.start(audioContext.currentTime);
        oscillator1.stop(audioContext.currentTime + 1);
      } catch (error) {
        console.warn('Could not play warning sound:', error);
      }
    }
  }, [isOpen, warning]);

  if (!warning) return null;

  // Convert warning to display format
  const displayWarning = {
    id: warning.id,
    licensePlate: `TRUCK-${warning.id.toString().slice(-3)}`,
    type: 'Restricted Hours Violation',
    timestamp: warning.detection?.timestamp || new Date(),
    status: 'pending',
    priority: 'critical' as const,
    location: warning.detection?.section || 'Unknown Section'
  };

  const timeAgo = Math.round((Date.now() - new Date(displayWarning.timestamp).getTime()) / 60000);

  const handleAction = (action: string) => {
    onAction(displayWarning.id.toString(), action);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="p-8">
        {/* Header with Animated Alert */}
        <div className="text-center mb-6">
          <div className="relative inline-flex items-center justify-center w-20 h-20 mb-4">
            <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-30"></div>
            <div className="relative bg-gradient-to-br from-red-500 to-red-600 rounded-full p-4">
              <AlertOctagon className="w-12 h-12 text-white animate-pulse" />
            </div>
          </div>
          
          <h1 className="text-3xl font-bold text-red-800 mb-2">
            🚨 CRITICAL TRAFFIC VIOLATION
          </h1>
          <p className="text-lg text-red-600 font-medium">
            Immediate Officer Response Required
          </p>
        </div>

        {/* Warning Details */}
        <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-xl p-6 mb-6 border-l-4 border-red-500">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center">
                <Siren className="w-6 h-6 text-red-600 animate-spin" />
              </div>
              <div>
                <h3 className="text-xl font-bold text-red-800">
                  {displayWarning.licensePlate}
                </h3>
                <p className="text-red-600">{displayWarning.type}</p>
              </div>
            </div>
            
            <div className="text-right">
              <div className="flex items-center text-sm text-red-600 mb-1">
                <Clock className="w-4 h-4 mr-1" />
                <span>{timeAgo} minutes ago</span>
              </div>
              <div className="flex items-center text-sm text-red-600">
                <MapPin className="w-4 h-4 mr-1" />
                <span>{displayWarning.location}</span>
              </div>
            </div>
          </div>

          <div className="bg-red-100 rounded-lg p-4 border border-red-200">
            <h4 className="font-bold text-red-800 mb-2 flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              RESTRICTED TIME ZONE VIOLATION
            </h4>
            <p className="text-red-700 text-sm">
              Heavy vehicles are prohibited from 11:00 PM to 6:00 AM in this area. 
              This violation requires immediate attention and appropriate action.
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-slate-800 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-blue-600" />
            Choose Your Response:
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => handleAction('inform')}
              className="group flex flex-col items-center p-4 bg-blue-500 hover:bg-blue-600 text-white rounded-xl transition-all duration-200 hover:scale-105 hover:shadow-lg"
            >
              <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mb-3 group-hover:bg-white/30 transition-colors">
                <Siren className="w-6 h-6" />
              </div>
              <span className="font-semibold">Inform Officers</span>
              <span className="text-xs opacity-80 mt-1 text-center">
                Notify nearby officers about this violation
              </span>
            </button>

            <button
              onClick={() => handleAction('permit')}
              className="group flex flex-col items-center p-4 bg-green-500 hover:bg-green-600 text-white rounded-xl transition-all duration-200 hover:scale-105 hover:shadow-lg"
            >
              <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mb-3 group-hover:bg-white/30 transition-colors">
                <CheckCircle className="w-6 h-6" />
              </div>
              <span className="font-semibold">Mark Permitted</span>
              <span className="text-xs opacity-80 mt-1 text-center">
                Vehicle has authorization for this route
              </span>
            </button>

            <button
              onClick={() => handleAction('dismiss')}
              className="group flex flex-col items-center p-4 bg-slate-500 hover:bg-slate-600 text-white rounded-xl transition-all duration-200 hover:scale-105 hover:shadow-lg"
            >
              <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mb-3 group-hover:bg-white/30 transition-colors">
                <XCircle className="w-6 h-6" />
              </div>
              <span className="font-semibold">Dismiss Alert</span>
              <span className="text-xs opacity-80 mt-1 text-center">
                No action required at this time
              </span>
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-slate-200">
          <div className="flex justify-between items-center text-sm text-slate-600">
            <span>Violation ID: {displayWarning.id}</span>
            <span>System Time: {new Date().toLocaleString()}</span>
          </div>
        </div>
      </div>
    </Modal>
  );
};

export default PopupWarning;