from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.predictor import ExoPredictor
import os

MODEL_PATH = "utils/models/exo_cnn.pth" 
CURVE_LENGTH = 2000 

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}. ¡Asegúrate de que exista!")

try:
    EXO_PREDICTOR = ExoPredictor(MODEL_PATH, CURVE_LENGTH)
except Exception as e:
    print(f"Error al inicializar ExoPredictor: {e}")
    EXO_PREDICTOR = None


app = FastAPI(
    title="Constellation Eye API",
    description="Servicio de predicción de exoplanetas por IA."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict")
async def predict_light_curve(file: UploadFile = File(...)):
    """
    Recibe un archivo (CSV o NPZ), lo preprocesa y devuelve la predicción del modelo.
    """
    if EXO_PREDICTOR is None:
        raise HTTPException(
            status_code=500,
            detail="Model unavaible"
        )

    filename = file.filename
    if not filename.lower().endswith(('.csv', '.npz')):
        raise HTTPException(
            status_code=400,
            detail="Not valid file type"
        )

    try:
        file_content = await file.read()

        prediction_result = EXO_PREDICTOR.predict_from_file(
            file_content=file_content,
            filename=filename
        )

        is_exoplanet = prediction_result['predicted_class_id'] in [1, 2]

        response_data = {
            "isExoplanet": is_exoplanet,
            "confidence": prediction_result['confidence'],
            "lightCurveData": prediction_result['lightCurveData'],
            "fullReport": prediction_result
        }

        return JSONResponse(content=response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error de procesamiento de datos: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API está funcionando", "predictor_loaded": EXO_PREDICTOR is not None}
