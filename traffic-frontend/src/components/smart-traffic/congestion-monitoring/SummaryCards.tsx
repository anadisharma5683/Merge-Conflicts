'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Map, AlertTriangle, Zap, Car } from 'lucide-react';
import type { Area } from '@/lib/types';

type SummaryCardProps = {
  areas: Area[];
};

export default function SummaryCards({ areas }: SummaryCardProps) {
  const criticalAreas = areas.filter(a => a.congestionStatus === 'Critical').length;
  const avgSpeed = areas.length > 0 ? Math.round(areas.reduce((sum, a) => sum + a.averageSpeed, 0) / areas.length) : 0;
  const totalVehicles = areas.reduce((sum, a) => sum + a.vehicleCount, 0);

  const summaryData = [
    {
      title: 'Total Areas',
      value: areas.length,
      icon: Map,
      color: 'from-blue-500 to-blue-600',
      iconColor: 'bg-blue-600/50',
    },
    {
      title: 'Critical Areas',
      value: criticalAreas,
      icon: AlertTriangle,
      color: 'from-red-500 to-red-600',
      iconColor: 'bg-red-600/50',
    },
    {
      title: 'Average Speed',
      value: `${avgSpeed} km/h`,
      icon: Zap,
      color: 'from-green-500 to-green-600',
      iconColor: 'bg-green-600/50',
    },
    {
      title: 'Total Vehicles',
      value: totalVehicles.toLocaleString(),
      icon: Car,
      color: 'from-amber-500 to-amber-600',
      iconColor: 'bg-amber-600/50',
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {summaryData.map((item, index) => (
        <Card key={index} className={`card-hover bg-gradient-to-br ${item.color} text-white border-0`}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-blue-50/80">{item.title}</CardTitle>
            <div className={`p-2 rounded-full ${item.iconColor}`}>
              <item.icon className="h-5 w-5 text-white" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{item.value}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
