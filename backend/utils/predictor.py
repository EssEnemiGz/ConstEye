import torch
import numpy as np
import io
import pandas as pd
import torch.nn.functional as F
from utils.models.network import ExoCNN

CLASS_LABELS = {
    0: "No Exoplanet",
    1: "Exoplanet",
    2: "Candidate"
}

class ExoPredictor:
    """
    Clase abstractora para manejar la carga de un modelo ExoCNN y realizar predicciones 
    en curvas de luz cargadas desde archivos CSV o NPZ, utilizando el 
    preprocesamiento exacto del LightCurveDataset.
    """
    def __init__(self, model_path: str, curve_length: int = 2000):
        """
        Inicializa el predictor, carga el modelo.

        :param model_path: Ruta al archivo .pth del modelo.
        :param curve_length: Longitud de curva esperada (2000, como en LightCurveDataset).
        """
        self.MODEL_PATH = model_path
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # La longitud máxima de la curva es 2000, según tu Dataset
        self.CURVE_LENGTH = curve_length 
        self.model = None
        self.load_model()
        print(f"ExoPredictor inicializado. Modelo en: {self.DEVICE}")

    def load_model(self):
        self.model = ExoCNN().to(self.DEVICE)
        self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.DEVICE))
        self.model.eval()

    def _preprocess_flux(self, flux_data: np.ndarray) -> torch.Tensor:
        """
        Aplica el preprocesamiento exacto usado en LightCurveDataset:
        1. Normalización Z-score.
        2. Truncamiento o Relleno a self.CURVE_LENGTH (2000).
        3. Formato a tensor (1, 1, L).

        :param flux_data: Array de NumPy del flujo (curva de luz).
        :return: Tensor de PyTorch listo para la inferencia (1, 1, L).
        """

        if np.std(flux_data) != 0:
            flux_processed = (flux_data - np.mean(flux_data)) / np.std(flux_data)
        else:
            flux_processed = flux_data - np.mean(flux_data) # Solo centrar

        current_len = len(flux_processed)
        max_len = self.CURVE_LENGTH

        if current_len > max_len:
            flux_processed = flux_processed[:max_len]
        elif current_len < max_len:
            flux_processed = np.pad(
                flux_processed, 
                (0, max_len - current_len), 
                mode='constant', 
                constant_values=0
            )

        flux_tensor = torch.tensor(flux_processed, dtype=torch.float32)
        flux_tensor = flux_tensor.unsqueeze(0).unsqueeze(0) 

        return flux_tensor.to(self.DEVICE)

    def _load_data_from_file(self, file: io.BytesIO, file_type: str) -> np.ndarray:
        """
        Carga la serie de flujo desde un buffer de archivo (BytesIO) NPZ o CSV.
        """
        if file_type == 'npz':
            npz_file = np.load(file)
            flux_data = npz_file['flux'].copy()

        elif file_type == 'csv':
            df = pd.read_csv(file)
            if 'flux' in df.columns:
                flux_data = df['flux'].values
            elif len(df.columns) > 1:
                flux_data = df.iloc[:, 1].values
            else:
                flux_data = df.iloc[:, 0].values

        else:
            raise ValueError("Tipo de archivo no soportado. Use 'npz' o 'csv'.")

        if flux_data.ndim > 1:
            flux_data = flux_data.squeeze() 

        if flux_data.ndim != 1:
            raise ValueError(f"El archivo {file_type} no contiene una curva de luz 1D válida.")

        return flux_data

    def predict_from_file(self, file_content: bytes, filename: str) -> dict:
        """
        Realiza la predicción del modelo a partir del contenido binario de un archivo.
        """
        if self.model is None:
             raise RuntimeError("El modelo no se cargó correctamente.")

        file_type = filename.split('.')[-1].lower()
        file_buffer = io.BytesIO(file_content)

        raw_flux_data = self._load_data_from_file(file_buffer, file_type)
        input_tensor = self._preprocess_flux(raw_flux_data)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

        predicted_class_idx = np.argmax(probs)
        confidence = probs[predicted_class_idx]

        flux_for_chart = input_tensor.cpu().squeeze().numpy()
        time_for_chart = np.arange(len(flux_for_chart)) 

        light_curve_data = [
            {"time": t.item(), "flux": f.item()} 
            for t, f in zip(time_for_chart, flux_for_chart)
        ]

        return {
            "predicted_class_id": predicted_class_idx.item(),
            "prediction_label": CLASS_LABELS.get(predicted_class_idx.item(), "Desconocido"),
            "confidence": confidence.item(),
            "probabilities": {CLASS_LABELS[i]: p.item() for i, p in enumerate(probs)},
            "lightCurveData": light_curve_data
        }
