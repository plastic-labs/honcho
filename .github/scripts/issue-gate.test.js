'use strict';

// Self-check for the gate decision logic. No framework, no install:
//   node .github/scripts/issue-gate.test.js
// Covers checkGate() only — the side-effecting halves (runGate/runSweep) are
// exercised against the real API via `pr-sweeper.yml`'s dry_run dispatch.

const assert = require('node:assert');
const { checkGate, REQUIRED_LABEL, EXEMPT_LABEL } = require('./issue-gate.js');

const pull = (over = {}) => ({
  number: 1, state: 'open', draft: false,
  user: { type: 'User' }, author_association: 'NONE', labels: [],
  ...over,
});

// `linked` is the list of issues GitHub resolves as closing references.
const stub = (linked) => ({
  graphql: async () => ({
    repository: { pullRequest: { closingIssuesReferences: {
      nodes: linked.map((i) => ({
        number: i.number, state: i.state || 'OPEN',
        labels: { nodes: (i.labels || []).map((name) => ({ name })) },
      })),
    } } },
  }),
});

const run = (linked, over) =>
  checkGate({ github: stub(linked), owner: 'o', repo: 'r', pr: pull(over) });

const cases = [
  ['no linked issue fails', () => run([]), (r) => r.passed === false],
  ['linked but unapproved fails', () => run([{ number: 7 }]), (r) => r.passed === false],
  ['linked and approved passes',
    () => run([{ number: 7, labels: [REQUIRED_LABEL] }]),
    (r) => r.passed === true && r.issue === 7],
  ['approved but closed fails',
    () => run([{ number: 7, state: 'CLOSED', labels: [REQUIRED_LABEL] }]),
    (r) => r.passed === false],
  ['picks the approved one out of several',
    () => run([{ number: 7 }, { number: 8, labels: [REQUIRED_LABEL] }]),
    (r) => r.passed === true && r.issue === 8],

  // Exemptions.
  ['maintainer skips', () => run([], { author_association: 'MEMBER' }), (r) => r.passed === true],
  ['collaborator skips', () => run([], { author_association: 'COLLABORATOR' }), (r) => r.passed === true],
  ['bot skips', () => run([], { user: { type: 'Bot' } }), (r) => r.passed === true],
  ['draft skips', () => run([], { draft: true }), (r) => r.passed === true],
  [`${EXEMPT_LABEL} skips`, () => run([], { labels: [{ name: EXEMPT_LABEL }] }), (r) => r.passed === true],

  // Regression guard: GitHub hands CONTRIBUTOR to anyone who has previously
  // committed, i.e. every returning outside contributor. It must stay gated.
  ['CONTRIBUTOR is still gated',
    () => run([], { author_association: 'CONTRIBUTOR' }),
    (r) => r.passed === false],
];

(async () => {
  let failed = 0;
  for (const [name, thunk, ok] of cases) {
    const result = await thunk();
    if (ok(result)) {
      console.log(`  ok   ${name}`);
    } else {
      failed++;
      console.log(`  FAIL ${name} -> ${JSON.stringify(result)}`);
    }
  }
  assert.strictEqual(failed, 0, `${failed} case(s) failed`);
  console.log(`\n${cases.length} passed`);
})();
