import React from 'react'
import { CheckCircle2, XCircle, Loader2, Gauge } from "lucide-react";
import { Card } from "./card-file-upload";
import { Progress } from "./progress";

export function PredictionResults({ prediction, isAnalyzing }) {
  if (isAnalyzing) {
    return (
      <Card className="flex items-center justify-center bg-card p-12">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="text-muted-foreground">Processing prediction...</p>
        </div>
      </Card>
    );
  }

  if (!prediction) return null;

  const confidencePercent = Math.round(prediction.confidence * 100);
  const isHighConfidence = prediction.confidence >= 0.8;

  return (
    <Card className="bg-card p-6">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Gauge className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">Prediction</h3>
        </div>

        <div className="flex items-center justify-center py-8">
          <div className="text-center">
            {prediction.isExoplanet ? (
              <div className="flex flex-col items-center gap-4">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-accent/10">
                  <CheckCircle2 className="h-12 w-12 text-accent" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-accent">Exoplanet Detected</p>
                  <p className="text-sm text-muted-foreground">Transit signal detected</p>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-muted">
                  <XCircle className="h-12 w-12 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-2xl font-bold text-foreground">No Detected</p>
                  <p className="text-sm text-muted-foreground">Without planet transit signal</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Confidence</span>
            <span className={`text-lg font-bold ${isHighConfidence ? "text-accent" : "text-primary"}`}>
              {confidencePercent}%
            </span>
          </div>
          <Progress value={confidencePercent} className="h-3" />
          <p className="text-xs text-muted-foreground">
            {isHighConfidence ? "High confidence" : "Moderate confidence - Check results"}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4 rounded-lg border border-border bg-secondary/50 p-4">
          <div>
            <p className="text-xs text-muted-foreground">Algorithm</p>
            <p className="font-mono text-sm font-semibold">CNN</p>
          </div>
        </div>
      </div>
    </Card>
  );
}
