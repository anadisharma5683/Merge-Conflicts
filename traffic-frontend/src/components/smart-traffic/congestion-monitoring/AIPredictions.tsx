'use client';

import { useState, useTransition, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Bot, Sparkles, Lightbulb, Loader2 } from 'lucide-react';
import { getCongestionPrediction } from '@/app/actions';
import { useToast } from '@/hooks/use-toast';
import type { Area } from '@/lib/types';
import type { PredictFutureCongestionOutput } from '@/ai/flows/predict-future-congestion';
import { getCongestionColor } from '@/lib/data';
import { Badge } from '@/components/ui/badge';
import { initialPredictions } from '@/lib/data';

type AIPredictionsProps = {
  selectedArea: Area | null;
};

export default function AIPredictions({ selectedArea }: AIPredictionsProps) {
  const [prediction, setPrediction] = useState<PredictFutureCongestionOutput | null>(
    selectedArea ? initialPredictions.find(p => p.areaId === selectedArea.id) || null : null
  );
  const [isPending, startTransition] = useTransition();
  const { toast } = useToast();

  const handleGetPrediction = () => {
    if (!selectedArea) {
        toast({ variant: 'destructive', title: 'No Area Selected', description: 'Please select an area to get a prediction.' });
        return;
    }
    startTransition(async () => {
      const result = await getCongestionPrediction(selectedArea);
      if (result.success && result.data) {
        setPrediction(result.data);
        toast({
          title: `AI Prediction for ${selectedArea.name}`,
          description: 'Future congestion levels have been forecasted.',
        });
      } else {
        toast({
          variant: 'destructive',
          title: 'Error',
          description: result.error || 'Could not get prediction.',
        });
      }
    });
  };

  useEffect(() => {
    if (selectedArea) {
      setPrediction(initialPredictions.find(p => p.areaId === selectedArea.id) || null);
    }
  }, [selectedArea]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="text-primary" />
          AI Traffic Predictions
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button onClick={handleGetPrediction} className="w-full font-bold card-hover" disabled={isPending || !selectedArea}>
          {isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="mr-2 h-4 w-4" />
          )}
          {selectedArea ? `Analyze ${selectedArea.name}` : 'Select an Area to Analyze'}
        </Button>

        {prediction ? (
          <div className="space-y-4 rounded-lg border bg-secondary/50 p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-secondary-foreground">{prediction.areaName} Forecast</h3>
              <Badge variant="outline" className="border-accent text-accent">AI Powered</Badge>
            </div>
            
            <div className="space-y-2">
              {prediction.predictions.map((pred, idx) => (
                <div key={idx} className="flex items-center justify-between rounded-md bg-card p-3 shadow-sm">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary font-bold">
                        {pred.timeSlot.split(' ')[0]}
                    </div>
                    <div>
                        <p className="font-semibold">{pred.timeSlot}</p>
                        <p className="text-xs text-muted-foreground">Confidence: {pred.confidence}%</p>
                    </div>
                  </div>
                  <p className="text-2xl font-bold" style={{ color: getCongestionColor(pred.predictedCongestion) }}>
                    {pred.predictedCongestion}%
                  </p>
                </div>
              ))}
            </div>

            <Card className="bg-card">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Lightbulb className="h-5 w-5 text-amber-500" />
                  Smart Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
                  {prediction.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card className="bg-card">
                <CardHeader className="pb-2">
                    <CardTitle className="text-base">Reasoning</CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-muted-foreground">{prediction.reasoning}</p>
                </CardContent>
            </Card>

          </div>
        ) : (
          <div className="flex h-64 items-center justify-center rounded-lg border-2 border-dashed text-center">
            <p className="text-muted-foreground">
                {selectedArea ? 'Click the button above to get AI predictions.' : 'Select an area from the monitor to begin.'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
