# Paperclip Integration Docs Design

## Goal

Add first-party Honcho documentation for the shipped `@honcho-ai/paperclip-honcho` plugin inside the Honcho repository. The docs should mirror the structure and discoverability of the existing OpenClaw integration page while staying strictly aligned with the Paperclip plugin behavior that has shipped so far.

## Source of Truth

- Shipped Paperclip plugin behavior from `paperclip-honcho/README.md`
- Existing Honcho integration docs structure from `docs/v3/guides/integrations/openclaw.mdx`
- Existing Honcho docs navigation and overview pages

## Scope

The change includes:

- a new integration page at `docs/v3/guides/integrations/paperclip.mdx`
- a navigation entry in `docs/docs.json`
- a Guides overview card in `docs/v3/guides/overview.mdx`

The change does not include:

- edits to the `paperclip-honcho` repository
- claims about planned or host-dependent features that are not shipped
- behavior descriptions that cannot be supported by the plugin README

## Recommended Approach

Create a dedicated Paperclip integration page under the Honcho docs `Integrations` section.

This is preferable to folding Paperclip into a broader overview page because:

- it matches the current docs information architecture
- it keeps the content discoverable next to OpenClaw and other integrations
- it gives enough room to explain the Paperclip operator flow without overloading the guides landing page

## Page Structure

The new `paperclip.mdx` page should follow the same broad shape as the OpenClaw page where applicable:

1. frontmatter with title, icon, description, and sidebar title
2. short introduction explaining what the plugin adds to Paperclip
3. installation instructions for installing the plugin into Paperclip
4. operator setup steps using the plugin settings page
5. explanation of how the sync model works
6. agent-facing tools section
7. configuration notes grounded in the shipped setup flow
8. next steps with links to the Paperclip plugin repository and relevant Honcho docs

## Content Boundaries

The content should explicitly stay within the shipped plugin surface described in the Paperclip README:

- company to workspace mapping
- agents and humans to peers
- issues to sessions
- issue comment sync
- issue document revision sync in bounded sections
- hard ingest cap on normalized content
- operator settings page and issue Memory tab
- operator actions such as config validation, connection testing, initialization, migration scan or import, repair mappings, and prompt-context preview
- agent tools currently exposed by the plugin

The page should explicitly avoid claiming support for:

- automatic prompt-context injection hooks
- run transcript import
- legacy workspace file import
- delegation or run-lineage reconstruction

## Verification

Verification for this docs change should confirm:

- `paperclip.mdx` is wired into the Honcho docs navigation
- the Guides overview includes a Paperclip card
- the page language matches shipped plugin behavior from the README
- the page structure is consistent with the OpenClaw integration guide
