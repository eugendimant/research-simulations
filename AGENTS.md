# AGENTS.md — Mergeable PRs, no human conflict work

NON-NEGOTIABLE
- I do not resolve conflicts manually in the GitHub UI. If a PR has conflicts, you must fix them by updating the branch yourself and pushing.
- Start from the latest default branch tip before editing anything.
- Before finalizing, update again from the latest default branch, resolve conflicts yourself, and rerun checks until green.

WORKFLOW (must follow)
1) Sync + branch (before edits)
   - git fetch origin
   - git checkout <default-branch>
   - git pull --ff-only
   - git checkout -b <feature-branch>

2) Implement minimal diff (no unrelated edits)
   - Do not touch these unless explicitly required: CHANGELOG.md, any README.md, MEMORY.md
   - Avoid version bumps unless explicitly requested

3) Verify locally (required)
   - Determine the correct commands from README/package config
   - Run the relevant checks (tests, lint, typecheck, build if present)
   - Fix and rerun until green

4) Make PR mergeable (required, before push)
   - git fetch origin
   - git merge origin/<default-branch>
   - If conflicts appear: resolve them yourself, then:
     - git add -A
     - git commit -m "Merge <default-branch> into <feature-branch>"
   - Rerun the checks after conflict resolution

SHORTHAND DEFINITIONS
- **"free-form text issue"**: Refers to the ongoing priority of improving non-LLM
  open-text response quality to be indistinguishable from real human writing.
  Workflow: (1) research published scientific approaches/tools/databases for
  realistic text generation, (2) implement findings on top of existing algorithms
  in response_library.py and persona_library.py, (3) iterate 3-5 times per round,
  (4) audit against human-likeness-checklist.md. Key references: Pennebaker &
  King (1999) LIWC styles, Krosnick (1991) satisficing theory, Denscombe (2008)
  response lengths, Hobbs (1979) discourse coherence, Zipf's law for vocabulary
  naturalness, QWERTY-adjacency typo models.

DONE CRITERIA
- PR shows "able to merge" in GitHub
- Relevant checks pass
