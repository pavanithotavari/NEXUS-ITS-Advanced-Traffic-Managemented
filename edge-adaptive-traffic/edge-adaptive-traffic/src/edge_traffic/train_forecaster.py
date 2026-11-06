import argparse
from .config import load_config
from .predictor import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rf", "mlp"], default="rf",
                   help="rf = RandomForest, mlp = small neural network")
    a = p.parse_args()
    mae, base = train(load_config(), kind=a.model)
    print(f"[{a.model}] Forecaster MAE: {mae:.2f} veh/min  (persistence baseline: {base:.2f})")
