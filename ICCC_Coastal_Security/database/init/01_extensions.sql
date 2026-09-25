-- Executed once by the postgis image on first start.
CREATE EXTENSION IF NOT EXISTS postgis;
-- TimescaleDB (optional, for vessel_track_points at scale) requires the timescale/timescaledb-ha image:
--   CREATE EXTENSION IF NOT EXISTS timescaledb;
--   SELECT create_hypertable('vessel_track_points', 'ts', migrate_data => true);
