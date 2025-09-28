'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Car, Zap, Clock, CheckCircle2 } from 'lucide-react';
import type { Area } from '@/lib/types';
import { getCongestionColor } from '@/lib/data';
import { cn } from '@/lib/utils';

type AreaMonitorProps = {
  areas: Area[];
  selectedArea: Area | null;
  onAreaSelect: (area: Area) => void;
};

const statusColors = {
  Low: 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-300 border-green-200 dark:border-green-700',
  Medium: 'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300 border-amber-200 dark:border-amber-700',
  High: 'bg-orange-100 text-orange-800 dark:bg-orange-900/50 dark:text-orange-300 border-orange-200 dark:border-orange-700',
  Critical: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-300 border-red-200 dark:border-red-700',
};

const metricIcons = [
  { icon: Car, label: 'Vehicles', color: 'text-purple-600 dark:text-purple-400', bgColor: 'bg-purple-100 dark:bg-purple-900/50' },
  { icon: Zap, label: 'km/h', color: 'text-blue-600 dark:text-blue-400', bgColor: 'bg-blue-100 dark:bg-blue-900/50' },
  { icon: Clock, label: 'min wait', color: 'text-orange-600 dark:text-orange-400', bgColor: 'bg-orange-100 dark:bg-orange-900/50' },
];

export default function AreaMonitor({ areas, selectedArea, onAreaSelect }: AreaMonitorProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="text-primary" />
          Area-wise Congestion Monitor
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4">
        {areas.map((area) => {
          const isSelected = selectedArea?.id === area.id;
          const metrics = [area.vehicleCount, area.averageSpeed, area.waitTime];

          return (
            <div
              key={area.id}
              onClick={() => onAreaSelect(area)}
              className={cn(
                'area-card-hover relative cursor-pointer rounded-xl border-2 p-4 transition-all',
                isSelected ? 'border-primary bg-secondary shadow-lg scale-105' : 'bg-card'
              )}
            >
              <Badge
                className={cn('absolute right-4 top-4 border', statusColors[area.congestionStatus], area.congestionStatus === 'Critical' ? 'animate-pulse' : '')}
              >
                {area.congestionStatus}
              </Badge>
              <h3 className="text-lg font-bold">{area.name}</h3>
              
              <div className="my-4 flex items-baseline justify-between">
                <p className="text-sm text-muted-foreground">Congestion</p>
                <p className="text-4xl font-bold" style={{ color: getCongestionColor(area.congestionLevel) }}>
                  {area.congestionLevel}%
                </p>
              </div>

              <Progress value={area.congestionLevel} className="h-2 [&>div]:bg-red-500" indicatorClassName={getCongestionColor(area.congestionLevel).replace('#', 'bg-[#') + ']'} style={{'--indicator-color': getCongestionColor(area.congestionLevel)} as React.CSSProperties} />

              <div className="mt-4 grid grid-cols-3 gap-2">
                {metricIcons.map((metric, index) => (
                    <div key={index} className={cn("text-center rounded-lg p-2", metric.bgColor)}>
                        <metric.icon className={cn("mx-auto h-6 w-6", metric.color)} />
                        <p className={cn("font-bold text-lg", metric.color)}>{metrics[index]}</p>
                        <p className={cn("text-xs font-medium", metric.color)}>{metric.label}</p>
                    </div>
                ))}
              </div>

              {isSelected && (
                <div className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
