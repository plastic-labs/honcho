'use strict';

// Self-check for the gate decision logic. No framework, no install:
//   node .github/scripts/issue-gate.test.js
// Covers checkGate() only — the side-effecting halves (runGate/runSweep) are
// exercised against the real API via `pr-sweeper.yml`'s dry_run dispatch.

const assert = require('node:assert');
const {
  checkGate, findNotices, runSweep, REQUIRED_LABEL, EXEMPT_LABEL, MARKER,
} = require('./issue-gate.js');

const pull = (over = {}) => ({
  number: 1, state: 'open', draft: false,
  user: { type: 'User', login: 'alice' }, labels: [],
  ...over,
});

const notCollaborator = () => {
  const err = new Error('Not Found');
  err.status = 404;
  throw err;
};

// `linked` is the list of issues GitHub resolves as closing references.
const stub = (linked, permission) => ({
  graphql: async () => ({
    repository: { pullRequest: { closingIssuesReferences: {
      nodes: linked.map((i) => ({
        number: i.number, state: i.state || 'OPEN',
        labels: { nodes: (i.labels || []).map((name) => ({ name })) },
      })),
    } } },
  }),
  rest: {
    repos: {
      getCollaboratorPermissionLevel: async () => {
        if (!permission) return notCollaborator();
        return { data: { permission } };
      },
    },
  },
});

const run = (linked, over, permission) =>
  checkGate({ github: stub(linked, permission), owner: 'o', repo: 'r', pr: pull(over) });

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
  ['write permission skips', () => run([], {}, 'write'), (r) => r.passed === true],
  ['maintain permission skips', () => run([], {}, 'maintain'), (r) => r.passed === true],
  ['bot skips', () => run([], { user: { type: 'Bot' } }), (r) => r.passed === true],
  ['draft skips', () => run([], { draft: true }), (r) => r.passed === true],
  [`${EXEMPT_LABEL} skips`, () => run([], { labels: [{ name: EXEMPT_LABEL }] }), (r) => r.passed === true],

  ['triage permission is still gated', () => run([], {}, 'triage'), (r) => r.passed === false],
  ['MEMBER association without write is still gated',
    () => run([], { author_association: 'MEMBER' }),
    (r) => r.passed === false],
  ['CONTRIBUTOR with write skips',
    () => run([], { author_association: 'CONTRIBUTOR' }, 'write'),
    (r) => r.passed === true],
];

// --- findNotices: only the bot's own notices count -------------------------
// A stranger pasting the invisible MARKER into a comment must not suppress the
// notice or become the grace-window clock.
const commentsStub = (comments) => ({
  paginate: async () => comments,
  rest: { issues: { listComments: null } },
});

const noticeCases = [
  ['a user comment carrying MARKER is not a notice',
    [{ id: 1, user: { type: 'User' }, body: `sneaky ${MARKER}`, created_at: 'x' }], 0],
  ['a bot comment carrying MARKER is a notice',
    [{ id: 2, user: { type: 'Bot' }, body: `${MARKER}\nnotice`, created_at: 'x' }], 1],
  ['a bot comment without MARKER is not a notice',
    [{ id: 3, user: { type: 'Bot' }, body: 'unrelated', created_at: 'x' }], 0],
  ['a user MARKER does not mask the real bot notice',
    [{ id: 4, user: { type: 'User' }, body: MARKER, created_at: 'x' },
     { id: 5, user: { type: 'Bot' }, body: MARKER, created_at: 'y' }], 1],
];

// --- runSweep: the stale-draft pass must honour every exemption ------------
const draft = (over) => ({
  number: 9, draft: true, state: 'open', labels: [],
  user: { type: 'User', login: 'alice' },
  updated_at: new Date(Date.now() - 400 * 86400_000).toISOString(),
  ...over,
});

async function sweepClosed(pr, permission) {
  const closed = [];
  const github = {
    paginate: async (route) => (route === 'pulls' ? [pr] : []),
    rest: {
      pulls: {
        list: 'pulls',
        update: async ({ pull_number }) => closed.push(pull_number),
      },
      issues: { listComments: 'comments', createComment: async () => {} },
      repos: {
        getCollaboratorPermissionLevel: async () => {
          if (!permission) return notCollaborator();
          return { data: { permission } };
        },
      },
    },
  };
  await runSweep({
    github, core: { info() {}, warning() {} },
    context: { repo: { owner: 'o', repo: 'r' } }, dryRun: false,
  });
  return closed;
}

const sweepCases = [
  ['stale draft from an outside author closes', draft({}), 1],
  ['stale draft from a bot is left alone', draft({ user: { type: 'Bot' } }), 0],
  ['stale draft from a writer is left alone', draft({}), 0, 'write'],
  [`stale draft with ${EXEMPT_LABEL} is left alone`, draft({ labels: [{ name: EXEMPT_LABEL }] }), 0],
  ['recent draft is left alone', draft({ updated_at: new Date().toISOString() }), 0],
];

(async () => {
  let failed = 0;
  for (const [name, comments, want] of noticeCases) {
    const got = (await findNotices({ github: commentsStub(comments), owner: 'o', repo: 'r', number: 1 })).length;
    if (got === want) console.log(`  ok   ${name}`);
    else { failed++; console.log(`  FAIL ${name} -> ${got} notices, wanted ${want}`); }
  }
  for (const [name, pr, want, permission] of sweepCases) {
    const got = (await sweepClosed(pr, permission)).length;
    if (got === want) console.log(`  ok   ${name}`);
    else { failed++; console.log(`  FAIL ${name} -> closed ${got}, wanted ${want}`); }
  }
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
  console.log(`\n${cases.length + noticeCases.length + sweepCases.length} passed`);
})();
