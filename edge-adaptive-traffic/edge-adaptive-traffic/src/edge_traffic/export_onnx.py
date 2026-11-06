"""Export the forecaster to ONNX for lightweight edge inference (needs skl2onnx)."""
import joblib
from skl2onnx import to_onnx
import numpy as np

model = joblib.load("models/forecaster.joblib")
onx = to_onnx(model, np.zeros((1, 8), dtype=np.float32))
open("models/forecaster.onnx", "wb").write(onx.SerializeToString())
print("Saved models/forecaster.onnx")
