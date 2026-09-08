'use strict';

// Self-check for the gate decision logic. No framework, no install:
//   node .github/scripts/issue-gate.test.js
// Covers checkGate() only — the side-effecting halves (runGate/runSweep) are
// exercised against the real API via `pr-sweeper.yml`'s dry_run dispatch.

const assert = require('node:assert');
const {
  checkGate, findNotices, runSweep, whoseTurn,
  REQUIRED_LABEL, EXEMPT_LABEL, MARKER,
  STALE_LABEL, STALE_MARKER, NO_AUTOCLOSE_LABEL,
  RESPONSE_STALE_DAYS, STALE_GRACE_HOURS,
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

// --- whoseTurn: only an unanswered hand-back is inside the SLA -------------
// Pure decision table. The stale pass closes on `author` and nothing else, so
// every branch that returns `maintainer` is a pull request we must not touch.
const HOUR = 3600_000;
const t = (hoursAgo) => Date.now() - hoursAgo * HOUR;

const turnCases = [
  ['never handed back is ours',
    { asked: 0, answered: t(900) }, 'maintainer'],
  ['approved is ours even after a hand-back',
    { reviewDecision: 'APPROVED', asked: t(900), answered: 0 }, 'maintainer'],
  ['hand-back with no reply is theirs',
    { reviewDecision: 'CHANGES_REQUESTED', asked: t(900), answered: 0 }, 'author'],
  ['hand-back the author replied to is ours',
    { reviewDecision: 'CHANGES_REQUESTED', asked: t(900), answered: t(100) }, 'maintainer'],
  ['reply older than the hand-back is still theirs',
    { reviewDecision: 'CHANGES_REQUESTED', asked: t(100), answered: t(900) }, 'author'],
  ['a re-review after the author replied is theirs again',
    { reviewDecision: 'CHANGES_REQUESTED', asked: t(50), answered: t(100) }, 'author'],
];

// --- runSweep: the stale pass ---------------------------------------------
// Records every write the sweep makes, so a case can assert on "warned but not
// closed" — the distinction the whole warn-then-close design rests on.
const idleDays = (days) => new Date(Date.now() - days * 86400_000).toISOString();

function sweepHarness({ pr, activity = {}, comments = [], permission }) {
  const log = { closed: [], comments: [], labels: [], unlabels: [], deleted: [] };
  const github = {
    paginate: async (route) => (route === 'pulls' ? [pr] : comments),
    // Serves both queries the sweep issues: the gate's closing-issue lookup and
    // the stale pass's activity lookup.
    graphql: async () => ({
      repository: { pullRequest: {
        closingIssuesReferences: { nodes: [] },
        author: { login: (pr.user || {}).login || 'alice' },
        reviewDecision: activity.reviewDecision || null,
        commits: { nodes: activity.commit ? [{ commit: { committedDate: activity.commit } }] : [] },
        reviews: { nodes: activity.reviews || [] },
        comments: { nodes: activity.comments || [] },
        timelineItems: { nodes: activity.labeled || [] },
      } },
    }),
    rest: {
      pulls: {
        list: 'pulls',
        update: async ({ pull_number }) => log.closed.push(pull_number),
      },
      issues: {
        listComments: 'comments',
        createComment: async ({ body }) => log.comments.push(body),
        addLabels: async ({ labels }) => log.labels.push(...labels),
        removeLabel: async ({ name }) => log.unlabels.push(name),
        deleteComment: async ({ comment_id }) => log.deleted.push(comment_id),
      },
      repos: {
        getCollaboratorPermissionLevel: async () => {
          if (!permission) return notCollaborator();
          return { data: { permission } };
        },
      },
    },
  };
  return runSweep({
    github, core: { info() {}, warning() {} },
    context: { repo: { owner: 'o', repo: 'r' } }, dryRun: false,
  }).then(() => log);
}

const open = (over = {}) => ({
  number: 20, draft: false, state: 'open', labels: [],
  user: { type: 'User', login: 'alice' },
  updated_at: idleDays(RESPONSE_STALE_DAYS + 5),
  ...over,
});

const staleNotice = (hoursAgo) => [{
  id: 99, user: { type: 'Bot' }, body: `${STALE_MARKER}\nwarned`,
  created_at: new Date(t(hoursAgo)).toISOString(),
}];

const review = (over = {}) => ({
  state: 'CHANGES_REQUESTED', submittedAt: new Date(t(400)).toISOString(),
  authorAssociation: 'MEMBER', author: { login: 'maintainer', __typename: 'User' },
  ...over,
});

const handedBack = { reviewDecision: 'CHANGES_REQUESTED', reviews: [review()] };

// A review bot runs as CONTRIBUTOR and can submit CHANGES_REQUESTED. If that
// counted, the sweeper would close pull requests no human ever looked at.
const botHandedBack = { reviewDecision: 'CHANGES_REQUESTED', reviews: [review({
  authorAssociation: 'CONTRIBUTOR', author: { login: 'coderabbitai', __typename: 'Bot' },
})] };

const staleCases = [
  ['idle hand-back is warned, not closed',
    { pr: open(), activity: handedBack },
    (l) => l.closed.length === 0 && l.labels.includes(STALE_LABEL)
      && l.comments.some((c) => c.includes(STALE_MARKER))],

  ['a bot requesting changes does not start the clock',
    { pr: open(), activity: botHandedBack },
    (l) => l.closed.length === 0 && l.labels.length === 0 && l.comments.length === 0],

  ['the author requesting changes on their own pull request does not count',
    { pr: open(), activity: { reviewDecision: 'CHANGES_REQUESTED', reviews: [review({
      authorAssociation: 'NONE', author: { login: 'alice', __typename: 'User' } })] } },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  ['idle pull request nobody reviewed is left completely alone',
    { pr: open(), activity: {} },
    (l) => l.closed.length === 0 && l.labels.length === 0 && l.comments.length === 0],

  ['approved and idle is left alone',
    { pr: open(), activity: { ...handedBack, reviewDecision: 'APPROVED' } },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  ['fresh hand-back is left alone',
    { pr: open({ updated_at: idleDays(1) }), activity: handedBack },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  ['a writer is never warned',
    { pr: open(), activity: handedBack, permission: 'write' },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  [`${NO_AUTOCLOSE_LABEL} pins it`,
    { pr: open({ labels: [{ name: NO_AUTOCLOSE_LABEL }] }), activity: handedBack },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  ['a draft is left to the stale-draft pass',
    { pr: open({ draft: true, updated_at: idleDays(8) }), activity: handedBack },
    (l) => l.closed.length === 0 && l.comments.length === 0],

  ['warned and still silent past the grace window closes',
    { pr: open({ labels: [{ name: STALE_LABEL }], updated_at: idleDays(0) }),
      comments: staleNotice(STALE_GRACE_HOURS + 1), activity: handedBack },
    (l) => l.closed.length === 1],

  ['warned but still inside the grace window does not close',
    { pr: open({ labels: [{ name: STALE_LABEL }], updated_at: idleDays(0) }),
      comments: staleNotice(STALE_GRACE_HOURS - 10), activity: handedBack },
    (l) => l.closed.length === 0],

  ['a reply after the warning stands the close down',
    { pr: open({ labels: [{ name: STALE_LABEL }], updated_at: idleDays(0) }),
      comments: staleNotice(STALE_GRACE_HOURS + 1),
      activity: { ...handedBack, commit: new Date(t(1)).toISOString() } },
    (l) => l.closed.length === 0 && l.unlabels.includes(STALE_LABEL) && l.deleted.includes(99)],

  ['the gate label wins — one clock at a time',
    { pr: open({ labels: [{ name: 'needs-approved-issue' }] }), activity: handedBack },
    (l) => l.labels.includes(STALE_LABEL) === false],
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
  for (const [name, activity, want] of turnCases) {
    const got = whoseTurn({ reviewDecision: null, ...activity }).turn;
    if (got === want) console.log(`  ok   ${name}`);
    else { failed++; console.log(`  FAIL ${name} -> ${got}, wanted ${want}`); }
  }
  for (const [name, setup, ok] of staleCases) {
    const log = await sweepHarness(setup);
    if (ok(log)) console.log(`  ok   ${name}`);
    else { failed++; console.log(`  FAIL ${name} -> ${JSON.stringify(log)}`); }
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
  const total = cases.length + noticeCases.length + sweepCases.length
    + turnCases.length + staleCases.length;
  console.log(`\n${total} passed`);
})();
