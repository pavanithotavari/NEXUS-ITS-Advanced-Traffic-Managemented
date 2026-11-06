"""ML traffic-demand forecaster (RandomForest) - predicts vehicles arriving next minute."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from .simulator import Intersection

LAGS = 5
AGG = 12  # 12 steps x 5 s = 1 minute


def build_dataset(cfg, hours=200, seed=1):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for h in range(hours):
        env = Intersection(cfg, seed=int(rng.integers(1e9)))
        env.reset()
        counts, hrs = [], []
        for i in range(cfg["sim"]["episode_steps"]):
            _, _, _, info = env.step(0)
            counts.append(int(info["arrivals"].sum()))
            hrs.append(env.hour)
        counts = np.array(counts)
        per_min = counts[: len(counts) // AGG * AGG].reshape(-1, AGG).sum(1)
        mins_hr = np.array(hrs)[::AGG][: len(per_min)]
        for i in range(LAGS, len(per_min) - 1):
            X.append(features(per_min[i - LAGS:i + 1], mins_hr[i]))
            y.append(per_min[i + 1])
    return np.array(X), np.array(y)


def features(last_counts, hour):
    return list(last_counts) + [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]


def make_model(kind):
    if kind == "mlp":  # small neural network, cheap enough for edge devices
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=600, random_state=0))
    return RandomForestRegressor(n_estimators=80, max_depth=10, n_jobs=-1, random_state=0)


def train(cfg, path="models/forecaster.joblib", kind="rf"):
    X, y = build_dataset(cfg)
    split = int(0.8 * len(X))
    model = make_model(kind)
    model.fit(X[:split], y[:split])
    mae = mean_absolute_error(y[split:], model.predict(X[split:]))
    base = mean_absolute_error(y[split:], X[split:, LAGS])  # persistence baseline
    joblib.dump(model, path)
    return mae, base


class Forecaster:
    def __init__(self, path="models/forecaster.joblib"):
        self.model = joblib.load(path)

    def predict(self, last_counts, hour):
        return float(self.model.predict([features(last_counts[-(LAGS + 1):], hour)])[0])
