FROM oven/bun:1.2.0

WORKDIR /app

# Required env vars (must be set at runtime):
#   HONCHO_USER - Required. The user identifier for Honcho API.
# Optional env vars:
#   HONCHO_API_URL - Defaults to "http://honcho-api:8000"
#   HONCHO_WORKSPACE - Defaults to "default"
#   HONCHO_API_KEY - Defaults to "not-needed"

# Create non-root user for security
RUN adduser --disabled-password --gecos '' mcpuser

# Install dependencies
COPY package.json bun.lock ./
RUN bun install

# Copy source
COPY src ./src
COPY tsconfig.json ./

# Set ownership
RUN chown -R mcpuser:mcpuser /app

USER mcpuser

# Run stdio-based MCP server
CMD ["bun", "run", "src/standalone.ts"]
