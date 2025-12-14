# Session Template for ARCEUS_DEVELOPMENT_LOG.md

Copy this template when starting new development work on Arceus.
Paste it into `ARCEUS_DEVELOPMENT_LOG.md` under "Development Sessions".

---

### Session X: YYYY-MM-DD - [Brief Descriptive Title]

**Objective**: [One sentence describing what you're trying to achieve]

**User Request**:
> [Direct quote or summary of user's request]

**Context**: [Any relevant background from previous sessions or current state]

---

#### Phase 1: [Phase Name - e.g., "API Enhancement", "Bug Fix", "New Feature"]

**Files Modified**: `file1.py`, `file2.py`
**Files Created**: `new_file.py` (if any)
**Files Deleted**: `old_file.py` (if any)

**Changes**:

1. **file1.py** - [High-level description of changes]
   - **Line XXX-YYY**: [Specific change with reasoning]
   - **New method**: `method_name()` at line XXX
     - Purpose: [What it does]
     - Parameters: [Key parameters]
     - Returns: [Return type and schema]
     - Integration: [How it connects to rest of system]

   - **Modified method**: `existing_method()` at line XXX
     - Changed: [What changed]
     - Reason: [Why it changed]
     - Impact: [What this affects]

   ```python
   # Example code snippet if helpful for understanding
   # Show before/after if it's a modification
   ```

2. **file2.py** - [Description]
   - [Similar structure as above]

**Rationale**:
[Explain WHY these changes were made, not just WHAT. Include:]
- Problem being solved
- Alternative approaches considered
- Why this approach was chosen
- Trade-offs made

**Integration Points**:
[List where this code connects to existing system:]
- Called from: `module.method()` at line XXX
- Calls: `other_module.method()` at line YYY
- Passes data via: `solving_context['key']`
- Stored in: Session metadata as `metadata['type']`

**Result**:
[Concrete outcomes:]
- ✅ What works now that didn't before
- 📊 Performance impact (if measurable)
- 👁️ User-visible changes in TUI
- 💾 Changes to stored data schema

**Testing**:
```bash
# Commands used to test
uv run python -m arceus.main --task-id 007bbfb7

# Expected output
[Describe what should happen]

# Actual output
[What actually happened]
```

**Verification**:
- [ ] Imports work: `uv run python -c "from arceus import module; print('OK')"`
- [ ] Single task runs without errors
- [ ] TUI displays correctly
- [ ] Metrics tracked properly
- [ ] Cost calculation accurate
- [ ] Memory operations counted
- [ ] No regression in existing features

**Known Issues**:
- [List any problems, limitations, or edge cases]
- [Include workarounds if any]

**Future Work**:
- [What should be done next]
- [Technical debt created]
- [Optimization opportunities]

---

#### Phase 2: [If multiple phases in this session]

[Repeat structure from Phase 1]

---

**Session Summary**:

**Total Changes**:
- Files modified: X
- Files created: Y
- Lines added: ~XXX
- Lines removed: ~YYY

**Impact**:
- [High-level impact on system]
- [New capabilities enabled]
- [Performance improvements]

**Dependencies**:
- [New dependencies added]
- [Changed version requirements]

**Breaking Changes**:
- [Any changes that break existing code]
- [Migration path if needed]

**Documentation Updates**:
- [ ] Updated ARCEUS_DEVELOPMENT_LOG.md
- [ ] Updated CLAUDE.md (if patterns changed)
- [ ] Updated QUICK_REFERENCE.md (if needed)
- [ ] Added inline docstrings
- [ ] Updated type hints

**Metrics**:
[If applicable, include before/after metrics:]
- API cost: $X.XX → $Y.YY
- Solve time: XXs → YYs
- Memory usage: X MB → Y MB

---

## Template Usage Notes

### When to Create a New Session
- Start of new development work
- New user request
- After significant time gap (days/weeks)
- Major feature addition or architectural change

### When to Add a New Phase
- Logically distinct part of work
- Different set of files being modified
- Different objective within same session

### Tips for Good Documentation
1. **Be specific**: "Added cost tracking" → "Added `calculate_api_cost()` method that multiplies tokens by MODEL_PRICING rates"
2. **Show connections**: Don't just list changes, explain how they fit together
3. **Include context**: Future you (or future Claude) needs to know WHY
4. **Use code snippets**: Show key patterns, especially if they're new
5. **Track all files**: Even small one-line changes should be noted
6. **Test thoroughly**: Document what you tested and results
7. **Think forward**: What will someone need to know to build on this?

### What NOT to Include
- Extremely verbose code dumps (link to files instead)
- Duplicate information (reference previous sessions if similar)
- Obvious changes (e.g., "added import" unless it's architecturally significant)

### Integration with Development Log
1. Copy this template
2. Fill in all sections
3. Paste at end of "Development Sessions" in ARCEUS_DEVELOPMENT_LOG.md
4. Update "Current State Summary" section
5. Update "File Reference" if new files or major changes
6. Commit with descriptive message

---

**Remember**: This log is for future developers (human or AI). Write it as if you're explaining to a smart colleague who knows Python but not this codebase.
