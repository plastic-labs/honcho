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

const hasLabel = (pr, name) => (pr.labels || []).some((l) => l.name === name);

const isBot = (account) => Boolean(account) && account.type === 'Bot';

/**
 * Why this pull request is exempt from the gate, or null if it is not.
 *
 * Single source of truth: every caller that acts on a pull request runs this.
 * The stale-draft sweep previously re-listed these checks and silently lost the
 * bot case.
 */
const exemptReason = (pr) => {
  if (isBot(pr.user)) return 'author is a bot';
  if (WRITE_ACCESS.includes(pr.author_association)) {
    return `author_association is ${pr.author_association}`;
  }
  if (hasLabel(pr, EXEMPT_LABEL)) return `carries the ${EXEMPT_LABEL} label`;
  return null;
};

// Write access to the repository. CONTRIBUTOR is deliberately absent: GitHub uses
// it for "has previously committed to the repository", which describes every
// returning outside contributor, not a maintainer. Do not add it.
const WRITE_ACCESS = ['OWNER', 'MEMBER', 'COLLABORATOR'];

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
  const exempt = exemptReason(pr);
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
async function findNotices({ github, owner, repo, number }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: number, per_page: 100,
  });
  return comments.filter((c) => isBot(c.user) && (c.body || '').includes(MARKER));
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

  const close = (pr, body) => async () => {
    await github.rest.issues.createComment({ owner, repo, issue_number: pr.number, body });
    await github.rest.pulls.update({ owner, repo, pull_number: pr.number, state: 'closed' });
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

    await act(`#${pr.number}: closing — notified ${Math.round(hours)}h ago, still failing`, close(pr,
      `Closing this: ${GRACE_HOURS} hours have passed and the gate is still not clear. This is not a judgement on the code. Link an approved issue and reopen — it goes straight into the review queue.`,
    ));
  }

  // Stale drafts. The gate skips drafts entirely, so they never carry the label;
  // this pass keys off inactivity and applies the shared exemptions itself.
  for (const pr of prs.filter((p) => p.draft)) {
    const exempt = exemptReason(pr);
    if (exempt) {
      core.info(`#${pr.number}: leaving stale draft alone — ${exempt}`);
      continue;
    }

    const days = (Date.now() - Date.parse(pr.updated_at)) / 86_400_000;
    if (days < DRAFT_STALE_DAYS) continue;

    await act(`#${pr.number}: closing stale draft — ${Math.round(days)}d without activity`, close(pr,
      `Closing this draft after ${DRAFT_STALE_DAYS} days without activity, to keep the pull request list readable. Reopen whenever you pick it back up — nothing here is lost.`,
    ));
  }
}

module.exports = {
  checkGate, runGate, runSweep, noticeBody, findNotices, exemptReason,
  REQUIRED_LABEL, GATE_LABEL, EXEMPT_LABEL, MARKER, GRACE_HOURS, DRAFT_STALE_DAYS,
};
