-- Runs once, when the Postgres volume is created.
-- The prediction log lives in its own database, separate from MLflow's tables.
CREATE DATABASE monitoring;
