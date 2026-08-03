# --- Stage 1: Lightning Fast Dependencies ---
FROM oven/bun:1 AS builder
WORKDIR /app

# Copy lockfile + config for deterministic, cached installs
COPY bun.lock package.json ./
RUN bun install --frozen-lockfile

# Copy source code
COPY . .

# --- Stage 2: Bun Runtime ---
FROM oven/bun:1
WORKDIR /app

# Ingest all the raw files (including node_modules) built gracefully by Bun
COPY --from=builder /app /app

# Setup environment 
ENV CI=true

# Run as non-root user (bun user is built into the oven/bun image)
USER bun

# Run the native Bun mock server (no wrangler dev)
CMD ["bun", "run", "run-mocked.ts"]
