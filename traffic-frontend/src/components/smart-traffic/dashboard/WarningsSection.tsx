import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, TestTube, Activity, Zap, AlertOctagon, Siren } from 'lucide-react';
import { useTruckWarning, useWarningSound } from '@/hooks';
import { TruckWarningNotification } from '@/components/smart-traffic/warnings';

const WarningsSection: React.FC = () => {
  const { 
    activeWarnings, 
    isRestrictedHours, 
    handleWarningAction, 
    triggerTestWarning 
  } = useTruckWarning();
  
  const { playActionSound } = useWarningSound();
  const [alertLevel, setAlertLevel] = useState<'normal' | 'warning' | 'critical'>('normal');
  
  // Update alert level based on active warnings
  useEffect(() => {
    if (activeWarnings.length === 0) {
      setAlertLevel('normal');
    } else if (activeWarnings.length < 3) {
      setAlertLevel('warning');
    } else {
      setAlertLevel('critical');
    }
  }, [activeWarnings.length]);
  
  const handleTestWarning = () => {
    playActionSound();
    triggerTestWarning();
  };

  return (
    <div className={`relative transition-all duration-500 ${
      alertLevel === 'critical' ? 'bg-gradient-to-br from-red-600 via-red-500 to-orange-500 p-1' :
      alertLevel === 'warning' ? 'bg-gradient-to-br from-orange-500 to-red-400 p-1' :
      'bg-white'
    } rounded-xl shadow-2xl`}>
      
      {/* Alert Level Background */}
      <div className={`${
        alertLevel !== 'normal' ? 'bg-white rounded-lg' : ''
      } transition-all duration-500`}>
        
        {/* Dramatic Header */}
        <div className={`p-6 ${
          alertLevel === 'critical' ? 'bg-gradient-to-r from-red-50 to-orange-50' :
          alertLevel === 'warning' ? 'bg-gradient-to-r from-orange-50 to-yellow-50' :
          'bg-white'
        } rounded-t-lg transition-all duration-500`}>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              {/* Dynamic Status Indicator */}
              <div className="relative">
                {alertLevel === 'critical' ? (
                  <AlertOctagon className="h-8 w-8 text-red-600 animate-bounce" />
                ) : alertLevel === 'warning' ? (
                  <AlertTriangle className="h-8 w-8 text-orange-600 animate-pulse" />
                ) : (
                  <Shield className="h-8 w-8 text-blue-600" />
                )}
                
                {alertLevel !== 'normal' && (
                  <Zap className="absolute -top-1 -right-1 h-4 w-4 text-yellow-500 animate-ping" />
                )}
              </div>
              
              <div>
                <h1 className={`text-2xl font-bold transition-colors duration-300 ${
                  alertLevel === 'critical' ? 'text-red-800' :
                  alertLevel === 'warning' ? 'text-orange-800' :
                  'text-gray-800'
                }`}>
                  {alertLevel === 'critical' ? '🚨 CRITICAL ALERT SYSTEM' :
                   alertLevel === 'warning' ? '⚠️ WARNING MANAGEMENT CENTER' :
                   '🛡️ Traffic Security Dashboard'}
                </h1>
                
                <p className={`text-sm font-medium ${
                  alertLevel === 'critical' ? 'text-red-600' :
                  alertLevel === 'warning' ? 'text-orange-600' :
                  'text-gray-600'
                }`}>
                  {activeWarnings.length} active violation{activeWarnings.length !== 1 ? 's' : ''} detected
                </p>
              </div>
            </div>
            
            {/* Status and Controls */}
            <div className="flex items-center space-x-4">
              {/* Restriction Status */}
              <div className={`flex items-center px-4 py-2 rounded-full text-sm font-bold transition-all duration-300 ${
                isRestrictedHours 
                  ? 'bg-red-100 text-red-800 border-2 border-red-300 animate-pulse' 
                  : 'bg-green-100 text-green-800 border-2 border-green-300'
              }`}>
                <div className={`w-3 h-3 rounded-full mr-2 animate-pulse ${
                  isRestrictedHours ? 'bg-red-500' : 'bg-green-500'
                }`} />
                <Siren className={`w-4 h-4 mr-2 ${
                  isRestrictedHours ? 'animate-spin text-red-600' : 'text-green-600'
                }`} />
                {isRestrictedHours ? '🌙 RESTRICTED ZONE ACTIVE' : '☀️ Normal Operations'}
              </div>
              
              {/* Test Button */}
              <button
                onClick={handleTestWarning}
                className={`group relative overflow-hidden flex items-center px-4 py-2 rounded-lg font-bold text-sm transition-all duration-300 transform hover:scale-105 ${
                  alertLevel === 'critical' ? 'bg-red-600 hover:bg-red-700 text-white' :
                  alertLevel === 'warning' ? 'bg-orange-600 hover:bg-orange-700 text-white' :
                  'bg-yellow-600 hover:bg-yellow-700 text-white'
                } shadow-lg hover:shadow-xl`}
              >
                <div className="absolute inset-0 bg-white/20 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300"></div>
                <TestTube className="w-4 h-4 mr-2 relative z-10" />
                <span className="relative z-10">SIMULATE VIOLATION</span>
              </button>
            </div>
          </div>
          
          {/* Alert Level Indicator Bar */}
          <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
            <div className={`h-2 rounded-full transition-all duration-1000 ${
              alertLevel === 'critical' ? 'w-full bg-gradient-to-r from-red-600 to-red-800' :
              alertLevel === 'warning' ? 'w-2/3 bg-gradient-to-r from-orange-500 to-red-500' :
              'w-1/3 bg-gradient-to-r from-green-500 to-blue-500'
            }`}></div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="p-6 space-y-6">
          {activeWarnings.length === 0 ? (
            <div className={`text-center py-12 rounded-lg transition-all duration-500 ${
              isRestrictedHours ? 'bg-red-50 border-2 border-red-200' : 'bg-green-50 border-2 border-green-200'
            }`}>
              <div className="relative mb-6">
                {isRestrictedHours ? (
                  <AlertTriangle className="h-16 w-16 text-red-400 mx-auto animate-pulse" />
                ) : (
                  <Shield className="h-16 w-16 text-green-400 mx-auto" />
                )}
                <Activity className="absolute top-4 right-4 h-6 w-6 text-blue-500 animate-bounce" />
              </div>
              
              <h3 className={`text-xl font-bold mb-2 ${
                isRestrictedHours ? 'text-red-700' : 'text-green-700'
              }`}>
                {isRestrictedHours ? '🌙 MONITORING RESTRICTED ZONE' : '✅ All Systems Secure'}
              </h3>
              
              <p className={`text-lg font-medium mb-4 ${
                isRestrictedHours ? 'text-red-600' : 'text-green-600'
              }`}>
                {isRestrictedHours 
                  ? 'System actively scanning for unauthorized vehicles...' 
                  : 'All traffic is compliant with current regulations'}
              </p>
              
              <div className={`inline-block px-6 py-3 rounded-full text-sm font-bold ${
                isRestrictedHours ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
              }`}>
                {isRestrictedHours ? '🔍 Active Surveillance Mode' : '🎆 Zero Violations Detected'}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {activeWarnings.map((warning, index) => (
                <div 
                  key={warning.id}
                  className="transform transition-all duration-500"
                  style={{ animationDelay: `${index * 200}ms` }}
                >
                  <TruckWarningNotification
                    warning={warning}
                    onAction={handleWarningAction}
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Enhanced Footer Information */}
        <div className={`p-6 rounded-b-lg transition-all duration-500 ${
          alertLevel === 'critical' ? 'bg-gradient-to-r from-red-100 to-orange-100 border-t-2 border-red-300' :
          alertLevel === 'warning' ? 'bg-gradient-to-r from-orange-100 to-yellow-100 border-t-2 border-orange-300' :
          'bg-blue-50 border-t-2 border-blue-200'
        }`}>
          <div className="flex items-center justify-between">
            <div>
              <h4 className={`text-sm font-bold mb-2 ${
                alertLevel === 'critical' ? 'text-red-800' :
                alertLevel === 'warning' ? 'text-orange-800' :
                'text-blue-800'
              }`}>
                🚛 Heavy Vehicle Restriction Protocol
              </h4>
              <p className="text-xs text-gray-700 leading-relaxed">
                🕛 <strong>Restricted Hours:</strong> 11:00 PM - 6:00 AM | 
                📏 <strong>Violation Code:</strong> HVR-2024 | 
                👮 <strong>Response Required:</strong> Immediate Officer Action
              </p>
            </div>
            
            <div className={`text-right ${
              alertLevel === 'critical' ? 'text-red-600' :
              alertLevel === 'warning' ? 'text-orange-600' :
              'text-blue-600'
            }`}>
              <div className="text-xs font-mono">
                System Status: {alertLevel.toUpperCase()}
              </div>
              <div className="text-xs font-mono">
                Last Update: {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WarningsSection;