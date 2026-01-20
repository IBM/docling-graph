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
- Tests on multiple platforms/Python versions
- Provides team-wide enforcement
- Required for external contributors

**Best Practice:** Keep both! Local hooks for speed, CI for enforcement.

## What We've Set Up

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)
**Runs on:** Every push and PR

**What it does:**
- Runs your pre-commit checks (ruff, mypy, pytest)
- Tests on Python 3.10, 3.11, 3.12, 3.13
- Tests on Linux, macOS, Windows
- Uploads coverage to Codecov
- Builds distribution packages

**Why:** Ensures code quality across all platforms and Python versions

### 2. DCO Sign-off Check (`.github/workflows/dco.yml`)
**Runs on:** Every PR

**What it does:**
- Verifies all commits are signed off with `git commit -s`
- Enforces Developer Certificate of Origin

**Why:** Legal requirement for open source contributions (you mentioned this!)

### 3. Security Scanning (`.github/workflows/security.yml`)
**Runs on:** Push, PR, and weekly

**What it does:**
- Dependency vulnerability scanning
- CodeQL security analysis
- Secret scanning (prevents API keys in code)
- Safety checks for Python packages

**Why:** Proactive security for open source projects

### 4. Automated Releases (`.github/workflows/release.yml`)
**Runs on:** Git tags like `v0.3.0`

**What it does:**
- Builds package
- Publishes to PyPI automatically
- Creates GitHub release
- Signs artifacts with Sigstore

**Why:** One command to release: `git tag v0.3.0 && git push --tags`

### 5. PR Auto-labeling (`.github/workflows/pr-labeler.yml`)
**Runs on:** Every PR

**What it does:**
- Automatically labels PRs based on changed files
- Labels: documentation, tests, dependencies, core, cli, etc.

**Why:** Better PR organization and filtering

### 6. Dependabot (`.github/dependabot.yml`)
**Runs:** Weekly on Mondays

**What it does:**
- Creates PRs to update dependencies
- Groups related dependencies
- Updates GitHub Actions versions

**Why:** Keep dependencies secure and up-to-date automatically

## Comparison: Local vs CI

| Check | Local (pre-commit) | CI (GitHub Actions) |
|-------|-------------------|---------------------|
| Ruff format | ✅ | ✅ |
| Ruff lint | ✅ | ✅ |
| MyPy | ✅ | ✅ |
| Pytest | ✅ | ✅ Multi-platform |
| UV lock | ✅ | ✅ |
| DCO sign-off | ❌ | ✅ |
| Security scan | ❌ | ✅ |
| Multi-platform | ❌ | ✅ |
| Multi-version | ❌ | ✅ |

## How to Use This Setup

### Step 1: Test in a Feature Branch

```bash
# Create feature branch
git checkout -b feature/github-actions-setup

# Add all the new files
git add .github/ docs/GITHUB_ACTIONS_SETUP.md docs/AUTOMATION_SUMMARY.md

# Commit with sign-off
git commit -s -m "feat: add GitHub Actions CI/CD workflows

- Add comprehensive CI pipeline with multi-platform testing
- Add DCO sign-off verification
- Add security scanning (CodeQL, dependency review, secrets)
- Add automated release workflow with PyPI publishing
- Add PR auto-labeling and Dependabot
- Add documentation for setup and usage"

# Push to GitHub
git push -u origin feature/github-actions-setup
```

### Step 2: Create Pull Request

1. Go to GitHub and create a PR from your feature branch
2. Watch the workflows run automatically
3. Fix any issues that come up
4. Once all checks pass, merge to main

### Step 3: Configure GitHub Settings

After merging, configure these in GitHub:

**Branch Protection (Settings → Branches):**
- Require PR reviews
- Require status checks: `Pre-commit checks`, `Test`, `Type checking`, `Linting`, `DCO Check`
- Require signed commits (optional)

**Enable Security Features (Settings → Security):**
- Dependency graph
- Dependabot alerts
- Secret scanning
- CodeQL analysis

**PyPI Publishing (for releases):**
- Go to https://pypi.org/manage/account/publishing/
- Add GitHub as trusted publisher
- No API tokens needed!

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

# 5. Push
git push -u origin feature/my-feature

# 6. Create PR - CI runs automatically
# 7. Merge after all checks pass
```

### Making a Release

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git commit -s -m "chore: bump version to 0.3.0"

# 3. Create and push tag
git tag v0.3.0
git push origin v0.3.0

# 4. GitHub Actions automatically:
#    - Builds package
#    - Publishes to PyPI
#    - Creates GitHub release
```

## Key Benefits

### 1. **Quality Assurance**
- Multi-platform testing catches platform-specific bugs
- Multi-version testing ensures compatibility
- Pre-commit + CI = double safety net

### 2. **Security**
- Automatic vulnerability scanning
- Secret detection prevents leaks
- Dependency updates via Dependabot

### 3. **Automation**
- One command releases (`git tag`)
- Auto-labeling organizes PRs
- Dependabot handles updates

### 4. **Compliance**
- DCO enforcement for legal compliance
- Signed commits option
- Audit trail in CI logs

### 5. **Contributor Friendly**
- Clear PR template
- Automatic checks guide contributors
- Fast feedback on issues

## FAQ

### Q: Should I remove local pre-commit hooks?
**A:** No! Keep them. They're faster and catch issues before commit.

### Q: What if CI fails but local pre-commit passed?
**A:** Usually platform-specific issues. Check the CI logs to see which platform/Python version failed.

### Q: Do I need to run pre-commit before every commit?
**A:** Pre-commit hooks run automatically. But you can skip with `git commit --no-verify` if needed (not recommended).

### Q: How do I sign commits?
**A:** Use `git commit -s` or configure globally: `git config --global format.signoff true`

### Q: What if I forget to sign a commit?
**A:** Amend it: `git commit --amend -s --no-edit && git push --force-with-lease`

### Q: Can contributors bypass these checks?
**A:** No, if you enable branch protection. All PRs must pass checks before merging.

## Next Steps

1. Test workflows in feature branch (create PR)
2. Configure branch protection rules
3. Enable security features
4. Set up PyPI trusted publishing
5. Create repository labels
6. Add Codecov token (optional)

## Resources

- Full setup guide: [`docs/GITHUB_ACTIONS_SETUP.md`](./GITHUB_ACTIONS_SETUP.md)
- GitHub Actions: https://docs.github.com/en/actions
- Pre-commit: https://pre-commit.com/
- DCO: https://developercertificate.org/