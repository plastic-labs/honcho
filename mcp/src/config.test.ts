import { expect, test } from "bun:test";
import { parseEnvConfig, sentryOptions } from "./config.ts";

test("HONCHO_TIMEOUT_MS sets the request timeout", () => {
  const base = { HONCHO_API_KEY: "k" };
  expect(parseEnvConfig({ ...base, HONCHO_TIMEOUT_MS: "90000" }).timeoutMs).toBe(90000);
  expect(parseEnvConfig({ ...base, HONCHO_TIMEOUT_MS: "nope" }).timeoutMs).toBeUndefined();
  expect(parseEnvConfig(base).timeoutMs).toBeUndefined();
});

test("Sentry events drop request headers and body", () => {
  const { beforeSend } = sentryOptions({});
  const event = beforeSend({
    type: undefined,
    request: {
      method: "POST",
      url: "https://mcp.honcho.dev/",
      headers: { authorization: "Bearer secret" },
      data: '{"content":"memory"}',
    },
  });
  expect(event.request).toEqual({ method: "POST", url: "https://mcp.honcho.dev/" });
});
