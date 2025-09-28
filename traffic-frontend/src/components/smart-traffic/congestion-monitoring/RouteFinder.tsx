'use client';

import { useState, useTransition } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { MapPin, ArrowRight, Milestone, Clock, TrafficCone, Star, Search, Loader2 } from 'lucide-react';
import { getOptimalRoutes } from '@/app/actions';
import { useToast } from '@/hooks/use-toast';
import type { Route } from '@/lib/types';
import { cn } from '@/lib/utils';
import { initialRoutes } from '@/lib/data';

const metricIcons = [
  { icon: Milestone, label: 'km', color: 'text-blue-600 dark:text-blue-400', bgColor: 'bg-blue-100 dark:bg-blue-900/50' },
  { icon: Clock, label: 'min', color: 'text-green-600 dark:text-green-400', bgColor: 'bg-green-100 dark:bg-green-900/50' },
  { icon: TrafficCone, label: 'traffic', color: 'text-orange-600 dark:text-orange-400', bgColor: 'bg-orange-100 dark:bg-orange-900/50' },
];

export default function RouteFinder() {
  const [routes, setRoutes] = useState<Route[]>(initialRoutes);
  const [isPending, startTransition] = useTransition();
  const { toast } = useToast();

  const handleFindOptimalRoute = () => {
    startTransition(async () => {
      const result = await getOptimalRoutes(routes);
      if (result.success && result.data) {
        setRoutes(result.data);
        toast({
          title: 'Optimal Route Found!',
          description: 'AI has analyzed the traffic and found the best route.',
        });
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: result.error || 'Could not find optimal route.',
        });
      }
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MapPin className="text-primary" />
          Smart Route Finder
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {routes.map((route) => {
          const metrics = [route.totalDistance, route.totalTime, `${route.averageCongestion}%`];
          return (
            <div key={route.id} className={cn("rounded-xl border-2 p-4 transition-all", route.isOptimal ? 'border-primary bg-secondary' : 'bg-card')}>
              <div className="mb-4 flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-bold">{route.name}</h3>
                  <div className="flex items-center text-sm text-muted-foreground">
                    <span>{route.origin}</span>
                    <ArrowRight className="mx-2 h-4 w-4" />
                    <span>{route.destination}</span>
                  </div>
                </div>
                {route.isOptimal && (
                  <div className="flex items-center gap-1 rounded-full bg-primary px-3 py-1 text-xs font-semibold text-primary-foreground">
                    <Star className="h-3 w-3" /> Recommended
                  </div>
                )}
              </div>
              <div className="grid grid-cols-3 gap-2">
                {metricIcons.map((metric, index) => (
                    <div key={index} className={cn("text-center rounded-lg p-2", metric.bgColor)}>
                        <metric.icon className={cn("mx-auto h-6 w-6", metric.color)} />
                        <p className={cn("font-bold text-lg", metric.color)}>{metrics[index]}</p>
                        <p className={cn("text-xs font-medium", metric.color)}>{metric.label}</p>
                    </div>
                ))}
              </div>
            </div>
          );
        })}
        <Button onClick={handleFindOptimalRoute} className="w-full font-bold card-hover" disabled={isPending}>
          {isPending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
          Find Optimal Route
        </Button>
      </CardContent>
    </Card>
  );
}
