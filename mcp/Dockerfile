# --- Stage 1: Lightning Fast Dependencies ---
FROM oven/bun:1 AS builder
WORKDIR /app

# Copy configuration and use Bun's superior async network stack to bypass WSL MTU hangs
COPY package.json ./
RUN bun install

# Copy source code
COPY . .

# --- Stage 2: Node Native Runtime ---
FROM node:20
WORKDIR /app

# Ingest all the raw files (including node_modules) built gracefully by Bun
COPY --from=builder /app /app

# Setup environment 
ENV CI=true

# Run the native Bun mock server (no wrangler dev)
CMD ["bun", "run", "run-mocked.ts"]
