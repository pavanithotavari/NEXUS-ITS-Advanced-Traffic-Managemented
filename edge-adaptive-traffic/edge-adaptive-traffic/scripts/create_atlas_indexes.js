// Run with mongosh after selecting the Atlas cluster.
db = db.getSiblingDB("traffic_management");
db.telemetry.createIndex({ intersection_id: 1, recorded_at: -1 });
