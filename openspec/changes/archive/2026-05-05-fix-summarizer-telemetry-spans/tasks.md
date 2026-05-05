## 1. Source Modification

- [x] 1.1 Remove `@conditional_observe(name="Create Short Summary")` from `create_short_summary` in `src/utils/summarizer.py`.
- [x] 1.2 Inject `track_name="Create Short Summary"` into the `honcho_llm_call` within `create_short_summary`.
- [x] 1.3 Remove `@conditional_observe(name="Create Long Summary")` from `create_long_summary` in `src/utils/summarizer.py`.
- [x] 1.4 Inject `track_name="Create Long Summary"` into the `honcho_llm_call` within `create_long_summary`.

## 2. Verification

- [x] 2.1 Trigger a summary flow within the application (e.g. via agent context limits or explicit test).
- [x] 2.2 Verify in the Langfuse UI that `Create Short Summary` and `Create Long Summary` now appear as root-level `GENERATION` observations containing valid `model` and `tokens` columns, with no empty top-level SPAN wrappers.
