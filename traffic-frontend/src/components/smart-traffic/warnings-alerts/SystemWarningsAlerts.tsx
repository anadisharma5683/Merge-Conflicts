'use client'

import React from 'react';
import { Card } from '@/components/ui/card';
import { CheckCircle, AlertTriangle, Info, Clock } from 'lucide-react';

interface SystemAlert {
  id: string;
  type: 'info' | 'warning' | 'success';
  message: string;
  timestamp: string;
  severity?: 'low' | 'medium' | 'high';
}

const SystemWarningsAlerts: React.FC = () => {
  const alerts: SystemAlert[] = [
    {
      id: '1',
      type: 'info',
      message: 'Camera at Vijay Nagar C-4 is offline.',
      timestamp: '2 minutes ago',
      severity: 'medium'
    },
    {
      id: '2',
      type: 'warning',
      message: 'High congestion detected at AB Road, deviating from prediction.',
      timestamp: '5 minutes ago',
      severity: 'high'
    },
    {
      id: '3',
      type: 'success',
      message: 'System update completed successfully.',
      timestamp: '1 hour ago',
      severity: 'low'
    }
  ];

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-orange-600" />;
      case 'info':
        return <Info className="w-5 h-5 text-blue-600" />;
      default:
        return <Info className="w-5 h-5 text-gray-600" />;
    }
  };

  const getAlertBorderColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-l-green-500';
      case 'warning':
        return 'border-l-orange-500';
      case 'info':
        return 'border-l-blue-500';
      default:
        return 'border-l-gray-500';
    }
  };

  const getAlertBgColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-green-50';
      case 'warning':
        return 'bg-orange-50';
      case 'info':
        return 'bg-blue-50';
      default:
        return 'bg-gray-50';
    }
  };

  return (
    <Card className="bg-white rounded-lg shadow-lg p-6 space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-2xl font-bold text-gray-900">System Warnings & Alerts</h2>
        
        {/* No Critical Warnings Status */}
        <div className="flex items-center space-x-2 p-4 bg-green-50 border border-green-200 rounded-lg">
          <CheckCircle className="w-5 h-5 text-green-600" />
          <span className="font-semibold text-green-800">No Critical Warnings</span>
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {alerts.map((alert) => (
          <Card 
            key={alert.id} 
            className={`p-4 border-l-4 ${getAlertBorderColor(alert.type)} ${getAlertBgColor(alert.type)} hover:shadow-md transition-shadow`}
          >
            <div className="flex items-start space-x-3">
              {/* Alert Icon */}
              <div className="flex-shrink-0 mt-0.5">
                {getAlertIcon(alert.type)}
              </div>

              {/* Alert Content */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 mb-1">
                  {alert.message}
                </p>
                <div className="flex items-center space-x-2 text-xs text-gray-500">
                  <Clock className="w-3 h-3" />
                  <span>{alert.timestamp}</span>
                </div>
              </div>

              {/* Severity Badge (if applicable) */}
              {alert.severity && (
                <div className="flex-shrink-0">
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    alert.severity === 'high' ? 'bg-red-100 text-red-800' :
                    alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                  </span>
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>

      {/* Alert Statistics */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-200">
        <div className="text-center">
          <p className="text-2xl font-bold text-green-600">0</p>
          <p className="text-sm text-gray-500">Critical</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-orange-600">1</p>
          <p className="text-sm text-gray-500">Warnings</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-blue-600">2</p>
          <p className="text-sm text-gray-500">Info</p>
        </div>
      </div>
    </Card>
  );
};

export default SystemWarningsAlerts;