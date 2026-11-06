from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from edge_traffic import api


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, field, direction):
        self.documents.sort(key=lambda item: item[field], reverse=direction < 0)
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeCollection:
    def __init__(self):
        self.documents = []

    def insert_one(self, document):
        stored = dict(document, _id="test-id")
        self.documents.append(stored)
        return SimpleNamespace(inserted_id="test-id")

    def find(self, query):
        return FakeCursor(
            [dict(doc) for doc in self.documents if doc["intersection_id"] == query["intersection_id"]]
        )

    def find_one(self, query, sort):
        matches = self.find(query).sort(*sort[0])
        return next(iter(matches), None)


def test_create_and_query_telemetry(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(api, "get_telemetry_collection", lambda: collection)
    reading = api.TelemetryReading.model_validate({
        "intersection_id": "junction01",
        "t": 4,
        "phase": "EW_GREEN",
        "queues": {"N": 1, "S": 0, "E": 3, "W": 7},
        "action": 0,
    })

    created = api.create_telemetry(reading)
    history = api.list_telemetry(intersection_id="junction01", limit=10)
    latest = api.latest_telemetry("junction01")

    assert created["_id"] == "test-id"
    assert created["recorded_at"].tzinfo == timezone.utc
    assert history[0]["queues"]["W"] == 7
    assert latest["phase"] == "EW_GREEN"


def test_telemetry_rejects_negative_queue_counts():
    with pytest.raises(ValidationError):
        api.TelemetryReading.model_validate({
            "t": 0,
            "phase": "NS_GREEN",
            "queues": {"N": -1, "S": 0, "E": 0, "W": 0},
            "action": 0,
        })


def test_telemetry_timestamp_is_timezone_aware():
    reading = api.TelemetryReading.model_validate({
        "t": 0,
        "phase": "NS_GREEN",
        "queues": {"N": 0, "S": 0, "E": 0, "W": 0},
        "action": 0,
    })
    assert isinstance(reading.recorded_at, datetime)
    assert reading.recorded_at.tzinfo == timezone.utc
