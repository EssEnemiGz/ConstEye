import React from 'react'
import { useState } from 'react'
import Header from './components/header.tsx'
import FeatureCard from './components/ui/feature-card.tsx'
import { LightCurveChart } from "./components/ui/lightcurve-chart"
import { PredictionResults } from "./components/ui/prediction-results"
import { FileUpload } from "./components/ui/file-upload.tsx"
import {
  Brain, Upload,
  ChartNoAxesCombined,
  TrendingUp
} from 'lucide-react'

function Index() {
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [prediction, setPrediction] = useState<{
    isExoplanet: boolean
    confidence: number
    lightCurveData: { time: number; flux: number }[]
  } | null>(null)

  const API_URL = 'http://localhost:7777/api/predict'

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile)
    setIsAnalyzing(true)
    setPrediction(null)

    const formData = new FormData()
    formData.append('file', uploadedFile)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Error desconocido del servidor' }))
        throw new Error(`Error en el análisis (${response.status}): ${errorData.detail}`)
      }

      const apiResult = await response.json()
      const finalPrediction = {
        isExoplanet: apiResult.isExoplanet,
        confidence: apiResult.confidence,
        lightCurveData: apiResult.lightCurveData,
      }

      setPrediction(finalPrediction)

    } catch (error) {
      console.error("Fallo la predicción:", error)
      setPrediction(null)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const generateMockLightCurve = () => {
    const data = []
    const hasTransit = Math.random() > 0.3

    for (let i = 0; i < 200; i++) {
      const time = i / 10
      let flux = 1.0 + (Math.random() - 0.5) * 0.005

      // Simular tránsito planetario
      if (hasTransit && i > 80 && i < 120) {
        const transitDepth = 0.015 + Math.random() * 0.025
        const transitProgress = (i - 80) / 40
        flux -= transitDepth * Math.sin(transitProgress * Math.PI)
      }

      data.push({ time, flux })
    }

    return data
  }

  return (
    <>
      <Header></Header>
      <main>
        <section className='py-20 px-4 bg-gradient-to-br from-blue-50 to-purple-50 via-white'>
          <div className='flex text-center max-w-4xl mx-auto flex-col items-center'>
            <p className='py-1 px-3 border border-gray-300 font-semibold text-xs rounded-2xl mb-6 max-w-fit'>
              Open source | Free | Self-Hosted
            </p>
            <h1 className='text-6xl font-bold mb-6 text-gray-900'>
              <span className='text-blue-600'>Constellation Eye</span><br />
              Explore the universe by your self
            </h1>
            <p className='max-w-2xl text-xl text-gray-600 mb-6'>
              Upload csv or npz files and access to free AI model to explore the universe just using your laptop
            </p>
            <div className='flex flex-col sm:flex-row gap-4 justify-center mb-12'>
              <FileUpload onFileUpload={handleFileUpload} isAnalyzing={isAnalyzing} currentFile={file} />
            </div>
            {/* Results Section */}
            {(prediction || isAnalyzing) && (
              <div className="space-y-6 w-full">
                <div className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-accent" />
                  <h2 className="text-xl font-semibold">Resultados del Análisis</h2>
                </div>

                <div className="grid gap-6 lg:grid-cols-1">
                  <PredictionResults prediction={prediction} isAnalyzing={isAnalyzing} />
                  <LightCurveChart data={prediction?.lightCurveData || []} isAnalyzing={isAnalyzing} />
                </div>
              </div>
            )}

            {/* Info Cards */}
            {!prediction && !isAnalyzing && (
              <div className="grid gap-4 md:grid-cols-3">
                <FeatureCard
                  icon={<Upload className='h-12 w-12 text-blue-600 mb-4'></Upload>}
                  title={"Supported formats"}
                  description={"NPZ and CSV files with lightcurves data from spacial telescopes"}
                ></FeatureCard>
                <FeatureCard
                  icon={<Brain className='h-12 w-12 text-green-600 mb-4'></Brain>}
                  title={"Advanced Analysis"}
                  description={"Automatic planets transit detection using neural networks"}
                ></FeatureCard>
                <FeatureCard
                  icon={<ChartNoAxesCombined className='h-12 w-12 text-purple-600 mb-4'></ChartNoAxesCombined>}
                  title={"Visualization"}
                  description={"Interactive light curve graphs"}
                ></FeatureCard>
              </div>
            )}
          </div>
        </section>
      </main >
    </>
  )
}

export default Index
