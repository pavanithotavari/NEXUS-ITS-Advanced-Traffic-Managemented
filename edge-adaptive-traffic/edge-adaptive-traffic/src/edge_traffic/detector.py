"""Vehicle detection at the edge (YOLO) + per-approach zone counting.

OpenCV/ultralytics are imported lazily so the rest of the project runs without them.
"""
APPROACHES = ["N", "S", "E", "W"]


class YoloDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO  # pip install ultralytics
        d = cfg["detector"]
        self.model = YOLO(d["weights"])
        self.conf = d["conf"]
        self.classes = d["vehicle_classes"]
        self.zones = cfg["zones"]

    def count(self, frame):
        """Return [N, S, E, W] vehicle counts for one BGR frame."""
        h, w = frame.shape[:2]
        res = self.model.predict(frame, conf=self.conf, classes=self.classes, verbose=False)[0]
        counts = [0, 0, 0, 0]
        for box in res.boxes.xyxy.cpu().numpy():
            cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
            for i, a in enumerate(APPROACHES):
                x1, y1, x2, y2 = self.zones[a]
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    counts[i] += 1
                    break
        return counts
