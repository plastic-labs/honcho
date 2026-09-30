import { expect, test } from "bun:test";
import { parseEnvConfig } from "./config.ts";

test("HONCHO_TIMEOUT_MS sets the request timeout", () => {
  const base = { HONCHO_API_KEY: "k" };
  expect(parseEnvConfig({ ...base, HONCHO_TIMEOUT_MS: "90000" }).timeoutMs).toBe(90000);
  expect(parseEnvConfig({ ...base, HONCHO_TIMEOUT_MS: "nope" }).timeoutMs).toBeUndefined();
  expect(parseEnvConfig(base).timeoutMs).toBeUndefined();
});
