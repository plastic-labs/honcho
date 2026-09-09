-- Runs once, on first initialisation of an empty PGDATA.
--
-- The sandbox uses its own database rather than `postgres` so that reset can DROP
-- and re-CREATE it from a seeded template. You cannot drop the database you are
-- connected to, and `postgres` is the default connection target for maintenance.
CREATE DATABASE honcho_sandbox;

\connect honcho_sandbox
CREATE EXTENSION IF NOT EXISTS vector;
