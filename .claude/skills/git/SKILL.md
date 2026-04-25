---
name: git
description: This skill should be used when the user invokes "/git --commit" to stage and commit changes, or "/git --push" to stage, commit, push, tag, and update the changelog. Triggers on "git commit", "commit my changes", "git push", "commit and push", or any git save workflow. Follows the versioning guide at /Users/tehmenghai/SCTP-DSAI/Module 3/versioning-guide.md.
version: 1.1.0
---

# Git Skill

Two-mode git workflow following the project versioning guide
(`/Users/tehmenghai/SCTP-DSAI/Module 3/versioning-guide.md`):

- `--commit` — stage + commit + update CHANGELOG.md
- `--push`   — all of the above + tag + push code + push tag

## Argument Handling

Parse the args passed to this skill:

- Args contain `--push`   → run **Commit** steps then **Push** steps
- Args contain `--commit` → run **Commit** steps only
- No recognised flag      → stop and tell the user:

  > Usage: `/git --commit` to commit only, or `/git --push` to commit, tag, and push.

---

## Commit Steps (both modes)

### Step 1 — Gather context (run in parallel)

```bash
git status
git diff HEAD
git log --oneline -10
```

Read the current CHANGELOG.md to find the latest version number:

```bash
head -30 CHANGELOG.md
```

**Edge cases — stop and report before proceeding:**
- Clean working tree → "Nothing to commit."
- Untracked files only → ask the user whether to include them before staging
- Detached HEAD → warn and ask which branch to use
- No remote configured (for `--push`) → report and suggest `git remote add origin <url>`

Never stage `.env`, `*.key`, `credentials.*`, or any file that looks like it contains secrets — warn the user and skip those files.

### Step 2 — Choose commit type and draft message

Pick the **one** type that best fits the changes, using this table from the versioning guide:

| Type | Use when | Version bump |
|---|---|---|
| `feat` | New feature or capability | MINOR |
| `fix` | Bug fix | PATCH |
| `refactor` | Code restructure, no behaviour change | PATCH |
| `style` | UI/CSS-only changes | PATCH |
| `perf` | Performance improvement | PATCH |
| `docs` | Documentation only | none |
| `chore` | Build, config, CI, dependencies | none |
| `test` | Adding or fixing tests | none |

Add `!` after type for breaking changes (MAJOR bump): `feat!: ...`

Format:
```
<type>(<optional scope>): <short description>

<optional body — explain WHY, not what. Wrap at 80 chars.>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Rules:
- Subject line: imperative mood, lowercase, no trailing period, max 72 chars
- One logical change per commit — don't mix a feature with a refactor

### Step 3 — Determine version bump

Apply this decision flow using the commit type chosen above:

```
Breaking change (! or BREAKING CHANGE)?
  YES → bump MAJOR, reset MINOR and PATCH to 0
  NO  → feat?
          YES → bump MINOR, reset PATCH to 0
          NO  → fix / refactor / style / perf?
                  YES → bump PATCH
                  NO  → no version bump (docs / chore / test)
```

Compute the new version from the current version found in CHANGELOG.md (Step 1).

### Step 4 — Update CHANGELOG.md

If the commit type warrants a version bump (MAJOR, MINOR, or PATCH):

Add a new section at the **top** of CHANGELOG.md (below the file header, above any existing version sections), using today's date (`2026-04-23`):

```markdown
## [X.Y.Z] - 2026-04-23

### Added        ← only include sections that have entries
- <one-line description written for humans, not code>

### Changed
- ...

### Fixed
- ...
```

Map commit type to changelog section:
- `feat` → `### Added`
- `fix` → `### Fixed`
- `refactor` / `style` / `perf` → `### Changed`
- `docs` / `chore` / `test` → no changelog entry needed

If no version bump: skip the CHANGELOG.md edit.

### Step 5 — Stage and commit

Stage the changed files **and** CHANGELOG.md (if updated) by explicit path. Never `git add -A`. Commit with a heredoc:

```bash
git add path/to/file1 path/to/file2 CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat: describe the change

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

If a pre-commit hook fails: fix the reported issue, re-stage, and create a **new** commit. Never use `--no-verify`.

---

## Push Steps (`--push` only)

### Step 6 — Tag the commit

If the commit produced a version bump, create an annotated tag:

```bash
git tag vX.Y.Z
```

If no version bump (docs/chore/test), skip tagging.

### Step 7 — Push code and tag

Check for upstream:

```bash
git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null
```

- Upstream exists → `git push`
- No upstream (new branch) → `git push -u origin <branch-name>`
- Branch is `main` or `master` → warn the user and ask for confirmation before pushing

Then push the tag (if one was created):

```bash
git push origin vX.Y.Z
```

---

## Final Report (both modes)

After completing the workflow, report:
- Commit SHA (short) and message
- New version (e.g. `1.0.0 → 1.1.0`) or "no version bump" if docs/chore/test
- CHANGELOG.md updated: yes/no
- For `--push`: remote and branch pushed to, tag pushed (if any)
- Any warnings (skipped secrets, protected branch, etc.)
