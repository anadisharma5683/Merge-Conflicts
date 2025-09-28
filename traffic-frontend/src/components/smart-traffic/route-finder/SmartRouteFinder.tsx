'use client'

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MapPin, Clock, TrendingUp, Search, Lightbulb, Sparkles, ChevronRight } from 'lucide-react';

interface RouteOption {
  name: string;
  description: string;
  distance: number;
  duration: number;
  trafficLevel: number;
}

interface TrafficPrediction {
  hour: number;
  time: string;
  congestionLevel: number;
  confidence: number;
}

const SmartRouteFinder: React.FC = () => {
  const [selectedArea, setSelectedArea] = useState('Vijay Nagar');

  const routeOptions: RouteOption[] = [
    {
      name: 'NH-16',
      description: 'Patia → Bhubaneswar Station',
      distance: 12,
      duration: 25,
      trafficLevel: 65
    },
    {
      name: 'City Road',
      description: 'Patia → Bhubaneswar Station',
      distance: 10,
      duration: 35,
      trafficLevel: 80
    }
  ];

  const predictions: TrafficPrediction[] = [
    { hour: 10, time: '10 PM', congestionLevel: 92, confidence: 95 },
    { hour: 11, time: '11 PM', congestionLevel: 75, confidence: 90 },
    { hour: 12, time: '12 AM', congestionLevel: 50, confidence: 88 }
  ];

  const recommendations = [
    'Increase signal time for North-South corridor.',
    'Divert traffic via Ring Road.'
  ];

  const getTrafficColor = (level: number) => {
    if (level >= 80) return 'text-red-600';
    if (level >= 60) return 'text-orange-600';
    return 'text-green-600';
  };

  const getTrafficBgColor = (level: number) => {
    if (level >= 80) return 'bg-red-50';
    if (level >= 60) return 'bg-orange-50';
    return 'bg-green-50';
  };

  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Smart Route Finder */}
      <Card className="bg-white rounded-lg shadow-lg p-6 space-y-6">
        <div className="flex items-center space-x-2">
          <MapPin className="w-5 h-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Smart Route Finder</h2>
        </div>

        <div className="space-y-4">
          {routeOptions.map((route, index) => (
            <Card key={index} className="p-4 border border-gray-200 hover:border-blue-300 transition-colors">
              <div className="space-y-3">
                <div>
                  <h3 className="font-semibold text-gray-900">{route.name}</h3>
                  <p className="text-sm text-gray-600">{route.description}</p>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-blue-50 rounded-lg p-3 text-center">
                    <div className="flex items-center justify-center space-x-1 text-blue-600 mb-1">
                      <ChevronRight className="w-4 h-4 rotate-90" />
                    </div>
                    <p className="text-lg font-bold text-blue-600">{route.distance} km</p>
                    <p className="text-xs text-gray-500">km</p>
                  </div>

                  <div className="bg-green-50 rounded-lg p-3 text-center">
                    <div className="flex items-center justify-center text-green-600 mb-1">
                      <Clock className="w-4 h-4" />
                    </div>
                    <p className="text-lg font-bold text-green-600">{route.duration} min</p>
                    <p className="text-xs text-gray-500">min</p>
                  </div>

                  <div className={`rounded-lg p-3 text-center ${getTrafficBgColor(route.trafficLevel)}`}>
                    <div className="flex items-center justify-center text-orange-600 mb-1">
                      <TrendingUp className="w-4 h-4" />
                    </div>
                    <p className={`text-lg font-bold ${getTrafficColor(route.trafficLevel)}`}>{route.trafficLevel}%</p>
                    <p className="text-xs text-gray-500">traffic</p>
                  </div>
                </div>
              </div>
            </Card>
          ))}

          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium flex items-center justify-center space-x-2">
            <Search className="w-4 h-4" />
            <span>Find Optimal Route</span>
          </Button>
        </div>
      </Card>

      {/* AI Traffic Predictions */}
      <Card className="bg-white rounded-lg shadow-lg p-6 space-y-6">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-5 h-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">AI Traffic Predictions</h2>
        </div>

        <div className="space-y-4">
          {/* Analysis Button */}
          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium flex items-center justify-center space-x-2">
            <Sparkles className="w-4 h-4" />
            <span>Analyze {selectedArea}</span>
          </Button>

          {/* Forecast Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-blue-600">{selectedArea} Forecast</h3>
              <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full">AI Powered</span>
            </div>

            <div className="space-y-3">
              {predictions.map((prediction, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="bg-blue-100 rounded-lg px-3 py-2 text-center">
                      <p className="text-lg font-bold text-blue-600">{prediction.hour}</p>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{prediction.time}</p>
                      <p className="text-sm text-gray-500">Confidence: {prediction.confidence}%</p>
                    </div>
                  </div>
                  <div className={`text-right ${getTrafficColor(prediction.congestionLevel)}`}>
                    <p className="text-2xl font-bold">{prediction.congestionLevel}%</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Smart Recommendations */}
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Lightbulb className="w-5 h-5 text-yellow-600" />
              <h3 className="font-semibold text-gray-900">Smart Recommendations</h3>
            </div>

            <div className="space-y-2">
              {recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-2 p-3 bg-yellow-50 rounded-lg">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2 flex-shrink-0"></div>
                  <p className="text-sm text-gray-700">{recommendation}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Reasoning Section */}
          <div className="bg-blue-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-2">Reasoning</h4>
            <p className="text-sm text-gray-600">
              Based on historical traffic patterns and current conditions, we predict increased congestion 
              during peak hours. The AI model suggests optimizing signal timing to improve traffic flow.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SmartRouteFinder;