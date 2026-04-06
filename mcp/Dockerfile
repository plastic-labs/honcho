FROM oven/bun:1.2.0

WORKDIR /app

# Install dependencies
COPY package.json bun.lock ./
RUN bun install

# Copy source
COPY src ./src
COPY tsconfig.json ./

# Keep container alive and pass stdin to the MCP server
CMD ["sh", "-c", "bun run src/standalone.ts"]
