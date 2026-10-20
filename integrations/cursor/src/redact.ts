/**
 * Best-effort secret redaction before anything is sent to Honcho.
 * Same rules as claude-honcho's redact.ts; novel formats pass through.
 */

const RULES: Array<{ pattern: RegExp; replacement: string }> = [
  // KEY=value assignments with a secret-bearing key
  {
    pattern:
      /\b(\w*(?:PASSWORD|PASSWD|PWD|SECRET|TOKEN|API_?KEY|ACCESS_KEY|CREDENTIALS?)\w*)\s*=\s*("[^"]*"|'[^']*'|[^\s;|&"']+)/gi,
    replacement: "$1=[redacted]",
  },
  // --password / --token style CLI flags
  {
    pattern: /(--(?:password|passwd|pwd|token|api-?key|secret|auth)[= ])(?:"[^"]*"|'[^']*'|[^\s;|&"']+)/gi,
    replacement: "$1[redacted]",
  },
  // Authorization headers
  {
    pattern: /(authorization:\s*(?:bearer|basic|token)\s+)[^\s"']+/gi,
    replacement: "$1[redacted]",
  },
  // Credentials embedded in URLs (scheme://user:pass@host)
  {
    pattern: /([a-z][a-z0-9+.-]*:\/\/[^/\s:@]+:)[^@\s]+@/gi,
    replacement: "$1[redacted]@",
  },
  // Well-known token shapes: Honcho, AWS, sk-, GitHub, Slack, GitLab, npm
  {
    pattern:
      /\b(?:hch[_-]?[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{16,}|(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{36})\b/g,
    replacement: "[redacted]",
  },
];

export function redactSecrets(text: string): string {
  let out = text;
  for (const rule of RULES) out = out.replace(rule.pattern, rule.replacement);
  return out;
}

/** Collapse fenced code blocks longer than `maxLines` so the deriver sees prose, not source. */
export function collapseCodeBlocks(text: string, maxLines = 12): string {
  return text.replace(/```[^\n]*\n([\s\S]*?)```/g, (block, body: string) => {
    const lines = body.split("\n").length;
    return lines > maxLines ? `[code block omitted, ${lines} lines]` : block;
  });
}
