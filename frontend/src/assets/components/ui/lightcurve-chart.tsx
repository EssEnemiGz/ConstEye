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
          <p className="text-muted-foreground">Generando curva de luz...</p>
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
          <h3 className="text-lg font-semibold">Curva de Luz</h3>
        </div>

        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />

              <XAxis
                dataKey="time"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                label={{
                  value: "Tiempo (días)",
                  position: "insideBottom",
                  offset: -5,
                  fill: "hsl(var(--muted-foreground))",
                }}
              />

              <YAxis
                stroke="hsl(var(--muted-foreground))"
                tick={{ fill: "hsl(var(--muted-foreground))" }}
                label={{
                  value: "Flujo Normalizado",
                  angle: -90,
                  position: "insideLeft",
                  fill: "hsl(var(--muted-foreground))",
                }}
                domain={yDomain}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                  color: "hsl(var(--popover-foreground))",
                }}
                labelStyle={{ color: "hsl(var(--muted-foreground))" }}
              />

              <ReferenceLine
                y={1.0}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="3 3"
              />

              <Line
                type="monotone"
                dataKey="flux"
                stroke="hsl(var(--primary))"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, fill: "hsl(var(--accent))" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-between rounded-lg border border-border bg-secondary/50 p-3 text-xs">
          <div>
            <span className="text-muted-foreground">Puntos de datos: </span>
            <span className="font-mono font-semibold">{data.length}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Rango: </span>
            <span className="font-mono font-semibold">
              {data[0]?.time.toFixed(1)} -{" "}
              {data[data.length - 1]?.time.toFixed(1)} días
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
}
