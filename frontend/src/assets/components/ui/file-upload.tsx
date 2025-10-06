import React from "react"
import { useCallback } from "react"
import { Upload, FileText, Loader2 } from "lucide-react"
import { Button } from "./button-file-upload"
import { Card } from "./card-file-upload"

interface FileUploadProps {
  onFileUpload: (file: File) => void
  isAnalyzing: boolean
  currentFile: File | null
}

export function FileUpload({ onFileUpload, isAnalyzing, currentFile }: FileUploadProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file && (file.name.endsWith(".npz") || file.name.endsWith(".csv"))) {
        onFileUpload(file)
      }
    },
    [onFileUpload],
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        onFileUpload(file)
      }
    },
    [onFileUpload],
  )

  return (
    <Card className="border-2 border-dashed border-border bg-card/50 transition-colors hover:border-primary/50">
      <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} className="p-8">
        <div className="flex flex-col items-center justify-center gap-4 text-center">
          {isAnalyzing ? (
            <>
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
              <div>
                <p className="text-lg font-semibold">Analizing data...</p>
                <p className="text-sm text-muted-foreground">Processing ligth curve with AI</p>
              </div>
            </>
          ) : currentFile ? (
            <>
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-accent/10">
                <FileText className="h-8 w-8 text-accent" />
              </div>
              <div>
                <p className="text-lg font-semibold">{currentFile.name}</p>
                <p className="text-sm text-muted-foreground">{(currentFile.size / 1024).toFixed(2)} KB</p>
              </div>
              <Button onClick={() => document.getElementById("file-input")?.click()} variant="outline" className="mt-2">
                Upload other file
              </Button>
            </>
          ) : (
            <>
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                <Upload className="h-8 w-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-semibold">Drop your file here</p>
                <p className="text-sm text-muted-foreground">or click to select</p>
              </div>
              <Button onClick={() => document.getElementById("file-input")?.click()} className="mt-2">
                Select file
              </Button>
              <p className="text-xs text-muted-foreground">Formats: NPZ, CSV (máx. 50MB)</p>
            </>
          )}
        </div>
        <input id="file-input" type="file" accept=".npz,.csv" onChange={handleFileInput} className="hidden" />
      </div>
    </Card>
  )
}

