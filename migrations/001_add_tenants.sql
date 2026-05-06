-- Migration: Add multi-tenant support
-- Adds the tenants table and tenant_id FK to workspaces

-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    configuration JSONB NOT NULL DEFAULT '{}'::jsonb,
    admin_jwt_secret TEXT,
    CONSTRAINT tenant_id_length CHECK (length(id) = 21),
    CONSTRAINT tenant_name_length CHECK (length(name) <= 512),
    CONSTRAINT tenant_id_format CHECK (id ~ '^[A-Za-z0-9_-]+$')
);

CREATE INDEX IF NOT EXISTS ix_tenants_created_at ON tenants (created_at DESC);

-- Add tenant_id to workspaces (nullable for backward compatibility)
ALTER TABLE workspaces
    ADD COLUMN IF NOT EXISTS tenant_id TEXT,
    ADD CONSTRAINT fk_workspaces_tenant_id
        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS ix_workspaces_tenant_id ON workspaces (tenant_id);
