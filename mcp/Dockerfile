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

# Revert boot command to native NPM/Node context to bypass Wrangler's anti-Bun blocker
# Crucially, we wrap it in a shell to inject the Docker container's environment variable into Wrangler's isolated sandbox
CMD sh -c "npx wrangler dev src/index.ts --port 8787 --ip 0.0.0.0 --var HONCHO_API_URL:$HONCHO_API_URL"
