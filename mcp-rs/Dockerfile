FROM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home mcp
COPY --from=builder /app/target/release/honcho-mcp-rs /usr/local/bin/honcho-mcp-rs

USER mcp
EXPOSE 8787
ENV MCP_BIND_ADDRESS=0.0.0.0:8787

CMD ["honcho-mcp-rs"]
