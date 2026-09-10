'use strict';

/**
 * Issue gate — shared logic for `.github/workflows/issue-gate.yml` (immediate
 * feedback on pull request events) and `.github/workflows/pr-sweeper.yml`
 * (deferred re-check, close, and stale-draft cleanup).
 *
 * Both workflows `require` this file through actions/github-script, so it must
 * stay dependency-free: neither job runs an install step.
 *
 * See CONTRIBUTING.md for the policy this enforces.
 */

const REQUIRED_LABEL = 'maintainer-approved';
const GATE_LABEL = 'needs-approved-issue';
const EXEMPT_LABEL = 'gate-exempt';
const MARKER = '<!-- issue-gate -->';
const DISCORD = 'http://discord.gg/honcho';

// Hours a labelled pull request has before the sweeper closes it. Measured from
// the notice comment, so the clock starts when the author was actually told —
// not when the pull request was opened.
const GRACE_HOURS = 72;

// Days without activity before a draft from outside the org is closed.
const DRAFT_STALE_DAYS = 30;

// The contributor response SLA from CONTRIBUTING.md, in days. The clock runs
// only while the pull request is waiting on its author — see `whoseTurn`.
const RESPONSE_STALE_DAYS = 7;

// Hours between the stale warning and the close. Same shape as GRACE_HOURS:
// nobody gets closed without having been told first.
const STALE_GRACE_HOURS = 72;

// Days a pull request waiting on *us* may sit idle before it is closed.
// `null` disables that pass, which is the intended default: a pull request no
// maintainer has looked at is our backlog, not a contributor SLA breach.
const UNREVIEWED_STALE_DAYS = null;

// Ceiling on stale closes per sweep. The open queue predates this policy, so a
// mistake here should cost a handful of pull requests and not the whole list.
const MAX_STALE_CLOSES = 10;

const STALE_LABEL = 'stale';
const STALE_MARKER = '<!-- pr-stale -->';

// Manual pin. Overrides the stale pass on a single pull request.
const NO_AUTOCLOSE_LABEL = 'no-autoclose';

// Labels a maintainer applies to hand a pull request back to its author. Doing
// so starts the SLA clock, exactly like a changes-requested review.
const HANDBACK_LABELS = ['needs-changes'];

// Associations that count as a maintainer hand-back. `CONTRIBUTOR` is missing on
// purpose: review bots run as CONTRIBUTOR, and a bot asking for changes must not
// start a clock that closes work no human has looked at.
const HANDBACK_ASSOCIATIONS = ['OWNER', 'MEMBER', 'COLLABORATOR'];

const hasLabel = (pr, name) => (pr.labels || []).some((l) => l.name === name);

const isBot = (account) => Boolean(account) && account.type === 'Bot';

/**
 * Why this pull request is exempt from the gate, or null if it is not.
 *
 * Single source of truth: every caller that acts on a pull request runs this.
 */
const exemptReason = async ({ github, owner, repo, pr }) => {
  if (isBot(pr.user)) return 'author is a bot';
  if (hasLabel(pr, EXEMPT_LABEL)) return `carries the ${EXEMPT_LABEL} label`;

  const username = pr.user && pr.user.login;
  if (!username) return null;

  const permission = await repoPermission({ github, owner, repo, username });
  if (WRITE_PERMISSIONS.includes(permission)) {
    return `author has ${permission} permission`;
  }
  return null;
};

// Repo roles that skip the gate. `read` / `triage` do not.
const WRITE_PERMISSIONS = ['admin', 'maintain', 'write'];

/** Highest repo permission for `username`, or null if they are not a collaborator. */
async function repoPermission({ github, owner, repo, username }) {
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      owner, repo, username,
    });
    return data.permission;
  } catch (err) {
    if (err && err.status === 404) return null;
    throw err;
  }
}

const CLOSING_ISSUES = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 20) {
          nodes {
            number
            state
            labels(first: 50) { nodes { name } }
          }
        }
      }
    }
  }
`;

/**
 * Decide whether a pull request clears the gate.
 *
 * Reads GitHub's own resolved issue links rather than parsing the body, so both
 * `Fixes #123` and the sidebar "Development" link count. A bare `#123` mention
 * deliberately does not — that is a reference, not a claim to close.
 *
 * @returns {Promise<{passed: boolean, skipped?: string, issue?: number, reason?: string}>}
 */
async function checkGate({ github, owner, repo, pr }) {
  if (pr.state !== 'open') return { passed: true, skipped: 'pull request is not open' };
  if (pr.draft) return { passed: true, skipped: 'pull request is a draft' };
  const exempt = await exemptReason({ github, owner, repo, pr });
  if (exempt) return { passed: true, skipped: exempt };

  const data = await github.graphql(CLOSING_ISSUES, { owner, repo, number: pr.number });
  const issues = data.repository.pullRequest.closingIssuesReferences.nodes;

  if (issues.length === 0) {
    return { passed: false, reason: 'This pull request is not linked to an issue.' };
  }

  const approved = issues.find(
    (i) => i.state === 'OPEN' && i.labels.nodes.some((l) => l.name === REQUIRED_LABEL),
  );
  if (approved) return { passed: true, issue: approved.number };

  const detail = issues
    .map((i) => `#${i.number} (${i.state === 'CLOSED' ? 'closed' : 'not approved'})`)
    .join(', ');
  return {
    passed: false,
    reason:
      `The linked ${issues.length === 1 ? 'issue is' : 'issues are'} not open with the ` +
      `\`${REQUIRED_LABEL}\` label: ${detail}.`,
  };
}

function noticeBody({ owner, repo, reason }) {
  return [
    MARKER,
    'Thanks for the contribution. This pull request does not clear our issue gate yet.',
    '',
    `**${reason}**`,
    '',
    `Every pull request to Honcho needs to be linked to an open issue carrying the \`${REQUIRED_LABEL}\` label. We do this so the review queue only holds work we have already agreed should be built — it means nobody spends time on a change we cannot merge.`,
    '',
    'To get this moving:',
    '',
    `1. Find or open an issue describing the change. [Approved issues are here](https://github.com/${owner}/${repo}/issues?q=is%3Aissue+is%3Aopen+label%3A${REQUIRED_LABEL}).`,
    `2. Make the case for it in [Discord](${DISCORD}) — maintainers are most active there, and it is by far the fastest route to a decision.`,
    `3. Once the issue has the label, link it: put \`Fixes #<number>\` in this pull request's description, or use **Development** in the sidebar.`,
    '',
    `**This will close automatically in ${GRACE_HOURS} hours if it is still unlinked.** Nothing is lost if that happens — link the issue, reopen, and it goes into the review queue.`,
    '',
    `See [CONTRIBUTING.md](https://github.com/${owner}/${repo}/blob/main/CONTRIBUTING.md) for the full process. If you think this is wrong, say so here and a maintainer will take a look.`,
  ].join('\n');
}

/**
 * Every gate notice this bot posted on a pull request, oldest first.
 *
 * Authorship is part of the test, not decoration. MARKER is an invisible HTML
 * comment, so anyone who can comment on a public repository can paste it. If
 * user comments counted, a third party could post one on someone else's pull
 * request: `runGate` posts a notice only when none exists, so the author would
 * never be told, and `runSweep` would then measure the grace window from the
 * stranger's timestamp and close them unwarned.
 */
async function findNotices({ github, owner, repo, number, marker = MARKER }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: number, per_page: 100,
  });
  return comments.filter((c) => isBot(c.user) && (c.body || '').includes(marker));
}

/**
 * Drop the gate label and delete the notice.
 *
 * Deleting matters: `runGate` posts a notice only when none exists, and the
 * sweeper measures grace from the notice timestamp. A notice left behind after
 * the gate clears would make a later re-block look weeks old and be closed with
 * no warning.
 */
async function clearGate({ github, owner, repo, pr }) {
  if (hasLabel(pr, GATE_LABEL)) {
    await github.rest.issues
      .removeLabel({ owner, repo, issue_number: pr.number, name: GATE_LABEL })
      .catch(() => {});
  }
  for (const notice of await findNotices({ github, owner, repo, number: pr.number })) {
    await github.rest.issues
      .deleteComment({ owner, repo, comment_id: notice.id })
      .catch(() => {});
  }
}

const ACTIVITY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        author { login }
        reviewDecision
        commits(last: 1) { nodes { commit { committedDate } } }
        reviews(last: 30) {
          nodes { state submittedAt authorAssociation author { login __typename } }
        }
        comments(last: 50) { nodes { createdAt author { login } } }
        timelineItems(last: 50, itemTypes: [LABELED_EVENT]) {
          nodes {
            ... on LabeledEvent { createdAt actor { __typename } label { name } }
          }
        }
      }
    }
  }
`;

/** Most recent of a list of timestamps, in epoch ms. 0 when there are none. */
const latest = (stamps) =>
  stamps.filter(Boolean).map(Date.parse).reduce((a, b) => (b > a ? b : a), 0);

const loginOf = (node) => (node && node.author && node.author.login) || null;

/**
 * A hand-back label applied by a person.
 *
 * Same reasoning as `isHandback`: applying a label needs triage permission, so
 * any human who can do it has standing — but a workflow holding `issues: write`
 * does not, and must not be able to start a clock on work no human reviewed.
 */
const isHandbackLabel = (node) =>
  Boolean(node.label) &&
  HANDBACK_LABELS.includes(node.label.name) &&
  !isBot({ type: node.actor && node.actor.__typename });

/** A changes-requested review that actually hands the pull request back. */
const isHandback = (review, prAuthor) =>
  loginOf(review) !== prAuthor &&
  !isBot({ type: review.author && review.author.__typename }) &&
  HANDBACK_ASSOCIATIONS.includes(review.authorAssociation);

/**
 * When we last handed a pull request back to its author, and when they last
 * did anything about it.
 *
 * `asked` counts only explicit hand-backs by a human maintainer: a
 * changes-requested review, or one of HANDBACK_LABELS. A passing remark in a
 * comment deliberately does not — the clock should start on an action a
 * maintainer took on purpose, and never on a review bot's verdict.
 *
 * `answered` uses the last commit's `committedDate`, which can predate the push
 * that carried it. The skew only ever makes a reply look older than it was, so
 * the error runs toward leaving a pull request open.
 */
async function prActivity({ github, owner, repo, pr }) {
  const data = await github.graphql(ACTIVITY, { owner, repo, number: pr.number });
  const node = data.repository.pullRequest;
  const author = loginOf(node);

  return {
    author,
    reviewDecision: node.reviewDecision,
    asked: latest([
      ...node.reviews.nodes
        .filter((r) => r.state === 'CHANGES_REQUESTED' && isHandback(r, author))
        .map((r) => r.submittedAt),
      ...node.timelineItems.nodes.filter(isHandbackLabel).map((n) => n.createdAt),
    ]),
    answered: latest([
      ...node.commits.nodes.map((n) => n.commit.committedDate),
      ...node.comments.nodes.filter((c) => loginOf(c) === author).map((c) => c.createdAt),
      ...node.reviews.nodes.filter((r) => loginOf(r) === author).map((r) => r.submittedAt),
    ]),
  };
}

/**
 * Whose turn it is. Pure, so the table in issue-gate.test.js can cover it.
 *
 * Only `author` is inside the SLA. Everything else is our queue: closing work
 * nobody reviewed would punish contributors for our own backlog, which is the
 * opposite of what the response window is for.
 *
 * @returns {{turn: 'author'|'maintainer', why: string}}
 */
function whoseTurn({ reviewDecision, asked, answered }) {
  if (reviewDecision === 'APPROVED') return { turn: 'maintainer', why: 'approved, waiting on merge' };
  if (!asked) return { turn: 'maintainer', why: 'never handed back to the author' };
  if (answered > asked) return { turn: 'maintainer', why: 'author already replied to the hand-back' };
  return { turn: 'author', why: 'handed back, no reply since' };
}

/** Undo a stale warning, so a later one is measured from its own timestamp. */
async function clearStale({ github, owner, repo, pr }) {
  if (hasLabel(pr, STALE_LABEL)) {
    await github.rest.issues
      .removeLabel({ owner, repo, issue_number: pr.number, name: STALE_LABEL })
      .catch(() => {});
  }
  const notices = await findNotices({
    github, owner, repo, number: pr.number, marker: STALE_MARKER,
  });
  for (const notice of notices) {
    await github.rest.issues
      .deleteComment({ owner, repo, comment_id: notice.id })
      .catch(() => {});
  }
}

function staleNoticeBody({ owner, repo, days }) {
  return [
    STALE_MARKER,
    `This pull request has been waiting on you for ${Math.round(days)} days.`,
    '',
    `We asked for changes and have not heard back. Honcho gives open source contributions a ${RESPONSE_STALE_DAYS} day response window — see [CONTRIBUTING.md](https://github.com/${owner}/${repo}/blob/main/CONTRIBUTING.md#review) — so this is the ping before it runs out.`,
    '',
    `**This will close automatically in ${STALE_GRACE_HOURS} hours unless something happens here.** A push is enough. So is a comment saying you need more time — we would much rather hold the pull request open than lose the work.`,
    '',
    `Closing is not a judgement on the code. The branch stays on your fork, and reopening puts it straight back in the queue.`,
  ].join('\n');
}

function staleCloseBody() {
  return (
    `Closing this after ${RESPONSE_STALE_DAYS} days with no reply to the requested changes, ` +
    `so the review queue reflects what is actually moving. Nothing here is lost — push to the ` +
    `branch and reopen whenever you pick it back up, or say so in [Discord](${DISCORD}) and a ` +
    `maintainer will reopen it for you.`
  );
}

/**
 * Entry point for `.github/workflows/issue-gate.yml`.
 * Labels and explains. Never closes — that is the sweeper's job.
 */
async function runGate({ github, core, context }) {
  const pr = context.payload.pull_request;
  const { owner, repo } = context.repo;
  const result = await checkGate({ github, owner, repo, pr });

  if (result.passed) {
    core.info(
      result.skipped ? `Skipping gate: ${result.skipped}` : `Gate passed via #${result.issue}`,
    );
    await clearGate({ github, owner, repo, pr });
    return;
  }

  core.warning(`Gate failed: ${result.reason}`);
  await github.rest.issues.addLabels({
    owner, repo, issue_number: pr.number, labels: [GATE_LABEL],
  });

  const notices = await findNotices({ github, owner, repo, number: pr.number });
  if (notices.length > 0) return;

  await github.rest.issues.createComment({
    owner, repo, issue_number: pr.number,
    body: noticeBody({ owner, repo, reason: result.reason }),
  });
}

/** Entry point for `.github/workflows/pr-sweeper.yml`. */
async function runSweep({ github, core, context, dryRun }) {
  const { owner, repo } = context.repo;

  const act = async (what, fn) => {
    core.info(dryRun ? `[dry run] ${what}` : what);
    if (!dryRun) await fn();
  };

  // A pull request can qualify for more than one pass. Whichever closes it
  // first owns it; the rest must not comment on a pull request already closed.
  const closed = new Set();

  const close = (pr, body) => async () => {
    await github.rest.issues.createComment({ owner, repo, issue_number: pr.number, body });
    await github.rest.pulls.update({ owner, repo, pull_number: pr.number, state: 'closed' });
  };

  // Marks the pull request closed for the rest of this run, dry or not, so the
  // dry-run log shows the same set of actions a live run would take.
  const closing = (pr, body) => {
    closed.add(pr.number);
    return close(pr, body);
  };

  const prs = await github.paginate(github.rest.pulls.list, {
    owner, repo, state: 'open', per_page: 100,
  });
  core.info(`${prs.length} open pull requests${dryRun ? ' (dry run)' : ''}`);

  // Re-check everything wearing the gate label. Never close blind: a pull request
  // linked through the sidebar fires no webhook, so the gate workflow cannot have
  // noticed it — this pass is the only thing that will.
  for (const pr of prs.filter((p) => hasLabel(p, GATE_LABEL))) {
    const result = await checkGate({ github, owner, repo, pr });

    if (result.passed) {
      const why = result.skipped || `via #${result.issue}`;
      await act(`#${pr.number}: gate now clear (${why})`, async () => {
        await clearGate({ github, owner, repo, pr });
        await github.rest.issues.createComment({
          owner, repo, issue_number: pr.number,
          body: 'The issue link is in place — this pull request has cleared the gate and is waiting on review.',
        });
      });
      continue;
    }

    const [notice] = await findNotices({ github, owner, repo, number: pr.number });
    if (!notice) {
      core.info(`#${pr.number}: labelled but never notified — leaving it for the gate workflow`);
      continue;
    }

    const hours = (Date.now() - Date.parse(notice.created_at)) / 3_600_000;
    if (hours < GRACE_HOURS) {
      core.info(`#${pr.number}: ${Math.round(GRACE_HOURS - hours)}h of grace left`);
      continue;
    }

    await act(`#${pr.number}: closing — notified ${Math.round(hours)}h ago, still failing`, closing(pr,
      `Closing this: ${GRACE_HOURS} hours have passed and the gate is still not clear. This is not a judgement on the code. Link an approved issue and reopen — it goes straight into the review queue.`,
    ));
  }

  // Stale pull requests waiting on their author. The window in CONTRIBUTING.md
  // is a *response* SLA, so it runs only while the ball is in the contributor's
  // court — `whoseTurn` is what decides that, and the log says which way it
  // went for every candidate. Warn first, close a grace period later.
  //
  // The STALE_LABEL is the cheap test for "already warned": `pulls.list`
  // carries labels, and the warning itself bumps `updated_at`, so idle time
  // stops being a usable clock the moment we post one.
  let budget = MAX_STALE_CLOSES;

  for (const pr of prs) {
    if (closed.has(pr.number)) continue;
    if (pr.draft) continue;                   // the stale-draft pass below owns drafts
    if (hasLabel(pr, GATE_LABEL)) continue;   // the gate pass above owns these
    if (hasLabel(pr, NO_AUTOCLOSE_LABEL)) {
      core.info(`#${pr.number}: pinned by ${NO_AUTOCLOSE_LABEL}`);
      continue;
    }

    if (hasLabel(pr, STALE_LABEL)) {
      // The label is not proof that we put it there, and the situation can have
      // changed since we did. Re-establish both before acting on it, or a
      // maintainer approving a warned pull request still sees it closed 72h
      // later — exactly what CONTRIBUTING.md promises will not happen.
      const exempt = await exemptReason({ github, owner, repo, pr });
      if (exempt) {
        await act(`#${pr.number}: ${exempt} — standing down`, () =>
          clearStale({ github, owner, repo, pr }));
        continue;
      }

      const activity = await prActivity({ github, owner, repo, pr });
      const { turn, why } = whoseTurn(activity);
      if (turn !== 'author') {
        await act(`#${pr.number}: no longer waiting on the author (${why}) — standing down`, () =>
          clearStale({ github, owner, repo, pr }));
        continue;
      }

      const [notice] = await findNotices({
        github, owner, repo, number: pr.number, marker: STALE_MARKER,
      });
      if (!notice) {
        core.warning(`#${pr.number}: ${STALE_LABEL} but no warning comment — re-warning`);
        await act(`#${pr.number}: re-posting the stale warning`, async () => {
          await github.rest.issues.createComment({
            owner, repo, issue_number: pr.number,
            body: staleNoticeBody({ owner, repo, days: RESPONSE_STALE_DAYS }),
          });
        });
        continue;
      }

      // Still their turn, but they answered and were handed back again since
      // the warning. Clear it so the next warning starts its own clock rather
      // than closing them out on a window they already responded to.
      if (activity.answered > Date.parse(notice.created_at)) {
        await act(`#${pr.number}: author replied after the warning — standing down`, () =>
          clearStale({ github, owner, repo, pr }));
        continue;
      }

      const waited = (Date.now() - Date.parse(notice.created_at)) / 3_600_000;
      if (waited < STALE_GRACE_HOURS) {
        core.info(`#${pr.number}: ${Math.round(STALE_GRACE_HOURS - waited)}h left after the warning`);
        continue;
      }
      if (budget <= 0) {
        core.warning(`#${pr.number}: ready to close but the ${MAX_STALE_CLOSES}-close budget is spent`);
        continue;
      }
      budget -= 1;
      await act(`#${pr.number}: closing — warned ${Math.round(waited)}h ago, no reply`,
        closing(pr, staleCloseBody()));
      continue;
    }

    // `updated_at` is an upper bound on every timestamp on the pull request, so
    // anything fresher than the shortest window cannot be stale under any rule.
    const idle = (Date.now() - Date.parse(pr.updated_at)) / 86_400_000;
    if (idle < RESPONSE_STALE_DAYS) continue;

    const exempt = await exemptReason({ github, owner, repo, pr });
    if (exempt) {
      core.info(`#${pr.number}: leaving idle pull request alone — ${exempt}`);
      continue;
    }

    const activity = await prActivity({ github, owner, repo, pr });
    const { turn, why } = whoseTurn(activity);
    const window = turn === 'author' ? RESPONSE_STALE_DAYS : UNREVIEWED_STALE_DAYS;

    if (window === null || idle < window) {
      core.info(`#${pr.number}: ${Math.round(idle)}d idle, waiting on ${turn} — ${why}`);
      continue;
    }

    await act(`#${pr.number}: warning — ${Math.round(idle)}d idle, ${why}`, async () => {
      await github.rest.issues.addLabels({
        owner, repo, issue_number: pr.number, labels: [STALE_LABEL],
      });
      await github.rest.issues.createComment({
        owner, repo, issue_number: pr.number,
        body: staleNoticeBody({ owner, repo, days: idle }),
      });
    });
  }

  // Stale drafts. The gate skips drafts entirely, so they never carry the label;
  // this pass keys off inactivity and applies the shared exemptions itself.
  for (const pr of prs.filter((p) => p.draft)) {
    if (closed.has(pr.number)) continue;
    const exempt = await exemptReason({ github, owner, repo, pr });
    if (exempt) {
      core.info(`#${pr.number}: leaving stale draft alone — ${exempt}`);
      continue;
    }

    const days = (Date.now() - Date.parse(pr.updated_at)) / 86_400_000;
    if (days < DRAFT_STALE_DAYS) continue;

    await act(`#${pr.number}: closing stale draft — ${Math.round(days)}d without activity`, closing(pr,
      `Closing this draft after ${DRAFT_STALE_DAYS} days without activity, to keep the pull request list readable. Reopen whenever you pick it back up — nothing here is lost.`,
    ));
  }
}

module.exports = {
  checkGate, runGate, runSweep, noticeBody, findNotices, exemptReason,
  prActivity, whoseTurn, staleNoticeBody, isHandback, isHandbackLabel,
  REQUIRED_LABEL, GATE_LABEL, EXEMPT_LABEL, MARKER, GRACE_HOURS, DRAFT_STALE_DAYS,
  STALE_LABEL, STALE_MARKER, NO_AUTOCLOSE_LABEL, HANDBACK_LABELS, HANDBACK_ASSOCIATIONS,
  RESPONSE_STALE_DAYS, STALE_GRACE_HOURS, UNREVIEWED_STALE_DAYS, MAX_STALE_CLOSES,
};
