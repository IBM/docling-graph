# GitHub Automation Summary

## Quick Answer to Your Questions

### Should you run pre-commit automatically after commit?

**No, keep it as a local pre-commit hook + CI check.** Here's why:

**Local (pre-commit hook):**
- Fast feedback before commit
- Prevents bad code from entering history
- Developer catches issues immediately
- You already have this configured

**CI (GitHub Actions):**
- Catches issues if someone bypasses local hooks
- Tests on multiple Python versions
- Provides team-wide enforcement
- Required for external contributors

**Best Practice:** Keep both! Local hooks for speed, CI for enforcement.

## What We've Set Up (Simplified)

### 1. CI Pipeline (`.github/workflows/ci.yml`)
**Runs on:** Every push and PR

**What it does:**
- Runs your pre-commit checks (ruff, mypy, pytest)
- Tests on Python 3.10 and 3.12 (Ubuntu only)
- Uploads coverage to Codecov
- Builds distribution packages

**Why:** Simple, fast checks that cover the essentials

### 2. DCO Sign-off Check (`.github/workflows/dco.yml`)
**Runs on:** Every PR

**What it does:**
- Verifies all commits are signed off with `git commit -s`
- Enforces Developer Certificate of Origin

**Why:** Legal requirement for open source contributions

### 3. Security Scanning (`.github/workflows/security.yml`)
**Runs on:** PRs and weekly

**What it does:**
- Dependency vulnerability scanning (high severity only)

**Why:** Basic security without overwhelming you

### 4. Automated Releases (`.github/workflows/release.yml`)
**Runs on:** Git tags like `v0.3.0`

**What it does:**
- Builds package
- Publishes to PyPI automatically
- Creates GitHub release with artifacts

**Why:** One command to release: `git tag v0.3.0 && git push --tags`

### 5. PR Auto-labeling (`.github/workflows/pr-labeler.yml`)
**Runs on:** Every PR

**What it does:**
- Automatically labels PRs based on changed files

**Why:** Better PR organization

### 6. Dependabot (`.github/dependabot.yml`)
**Runs:** Weekly on Mondays

**What it does:**
- Creates PRs to update dependencies
- Groups related dependencies

**Why:** Keep dependencies up-to-date automatically

## Comparison: Local vs CI

| Check | Local (pre-commit) | CI (GitHub Actions) |
|-------|-------------------|---------------------|
| Ruff format | Yes | Yes |
| Ruff lint | Yes | Yes |
| MyPy | Yes | Yes |
| Pytest | Yes | Yes (Python 3.10 & 3.12) |
| UV lock | Yes | Yes |
| DCO sign-off | No | Yes |
| Security scan | No | Yes (Weekly) |

## How to Use This Setup

### Step 1: Test in Your Feature Branch

```bash
# Your branch should already have the files
git add .github/ docs/

# Commit with sign-off
git commit -s -m "feat: add simplified GitHub Actions workflows

- Add CI pipeline with Python 3.10 & 3.12 testing
- Add DCO sign-off verification
- Add basic security scanning
- Add automated release workflow
- Add PR auto-labeling and Dependabot"

# Push to test
git push
```

### Step 2: Review PR

1. Check that workflows run successfully
2. Fix any issues
3. Merge when all checks pass

### Step 3: Configure GitHub Settings (After Merge)

**Branch Protection (Settings → Branches):**
- Require PR reviews
- Require status checks: `Pre-commit checks`, `Test (3.10)`, `Test (3.12)`, `DCO Check`

**Enable Security (Settings → Security):**
- Dependency graph
- Dependabot alerts

**PyPI Publishing (for releases):**
- Go to https://pypi.org/manage/account/publishing/
- Add GitHub as trusted publisher
- Repository: your-username/docling-graph
- Workflow: release.yml
- Environment: pypi

## Your Workflow Going Forward

### Daily Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Run pre-commit locally (fast feedback)
uv run pre-commit run --all-files

# 4. Commit with sign-off
git commit -s -m "feat: add my feature"

# 5. Push and create PR
git push -u origin feature/my-feature
```

### Making a Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git commit -s -m "chore: bump version to 0.3.0"
git push

# 3. Create and push tag
git tag v0.3.0
git push origin v0.3.0

# 4. GitHub Actions automatically:
#    - Builds package
#    - Publishes to PyPI
#    - Creates GitHub release
```

## Key Benefits

### 1. Simple & Fast
- Only 2 Python versions (3.10, 3.12)
- Ubuntu only (most common)
- Pre-commit catches most issues locally

### 2. Essential Security
- Dependency scanning for high severity issues
- Weekly automated checks
- Not overwhelming with alerts

### 3. Easy Releases
- One command: `git tag v0.3.0 && git push --tags`
- Automatic PyPI publishing
- GitHub release created automatically

### 4. Compliance
- DCO enforcement for legal compliance
- Clear PR template guides contributors

## FAQ

### Q: Why only Python 3.10 and 3.12?
**A:** Covers minimum supported (3.10) and latest stable (3.12). Add more versions later if needed.

### Q: Why only Ubuntu?
**A:** Most users are on Linux. Add macOS/Windows testing later if you get platform-specific issues.

### Q: Should I remove local pre-commit hooks?
**A:** No! Keep them. They're faster and catch issues before commit.

### Q: How do I sign commits?
**A:** Use `git commit -s` or configure globally: `git config --global format.signoff true`

### Q: What if I forget to sign a commit?
**A:** Amend it: `git commit --amend -s --no-edit && git push --force-with-lease`

## Next Steps

1. Test workflows in your PR
2. Merge to main after checks pass
3. Configure branch protection rules
4. Set up PyPI trusted publishing
5. Create repository labels (optional)

## Resources

- Full setup guide: [SETUP.md](./SETUP.md)
- Docling reference: https://github.com/docling-project/docling
- GitHub Actions: https://docs.github.com/en/actions
- DCO: https://developercertificate.org/