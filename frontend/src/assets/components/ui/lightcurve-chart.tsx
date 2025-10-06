import React from 'react'
import { Card } from "./card-file-upload";
import { Loader2, Activity } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface LightCurveChartProps {
  data: { time: number; flux: number }[];
  isAnalyzing: boolean;
}

/**
 * Componente que muestra una curva de luz (Light Curve Chart).
 * @param {LightCurveChartProps} props - Las propiedades del componente.
 * @returns {JSX.Element | null} El gráfico o un estado de carga/nulo.
 */

export function LightCurveChart({ data, isAnalyzing }: LightCurveChartProps) {
  if (isAnalyzing) {
    return (
      <Card className="flex items-center justify-center bg-card p-12">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="text-muted-foreground">Creating ligth curve</p>
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) return null;

  const fluxValues = data.map((d) => d.flux);
  const minFlux = Math.min(...fluxValues);
  const maxFlux = Math.max(...fluxValues);
  const padding = (maxFlux - minFlux) * 0.1 || 0.01;
  const yDomain = [minFlux - padding, maxFlux + padding];

  return (
    <Card className="bg-card p-6 w-auto">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-accent" />
          <h3 className="text-lg font-semibold">Light Curve</h3>
        </div>

        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 10, right: 30, left: 40, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />

              <XAxis
                dataKey="time"
                stroke="#A0AEC0"
                tick={{ fill: "#A0AEC0" }}
                label={{
                  value: "Time (days)",
                  position: "insideBottom",
                  offset: -5,
                  fill: "#A0AEC0",
                }}
              />

              <YAxis
                stroke="#A0AEC0"
                tick={{ fill: "#A0AEC0" }}
                tickFormatter={(value) => value.toFixed(4)}
                label={{
                  value: "Normalized Flow",
                  angle: -90,
                  position: "insideLeft",
                  offset: -35,
                  fill: "#2D3748",
                }}
                domain={yDomain}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E2E8F0",
                  borderRadius: "8px",
                  color: "#2D3748",
                }}
                labelStyle={{ color: "#A0AEC0" }}
                itemStyle={{ color: "#2D3748" }}
              />

              <ReferenceLine
                y={1.0}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="3 3"
              />

              <Line
                type="monotone"
                dataKey="flux"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, fill: "#2563eb", stroke: "#2563eb" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between rounded-lg border border-border bg-secondary/50 p-3 text-xs">
          <div>
            <span className="text-muted-foreground">Data points: </span>
            <span className="font-mono font-semibold">{data.length}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Range: </span>
            <span className="font-mono font-semibold">
              {data[0]?.time.toFixed(1)} -{" "}
              {data[data.length - 1]?.time.toFixed(1)} days
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}
