# GitHub Actions Setup Guide

This document explains the GitHub Actions workflows configured for this project and how to set them up.

## Overview

The project uses several GitHub Actions workflows to automate CI/CD, security scanning, and release management:

1. **CI Workflow** - Runs tests, linting, and type checking
2. **DCO Check** - Ensures all commits are signed off
3. **Security Scanning** - Scans for vulnerabilities
4. **Release Workflow** - Automates package publishing
5. **PR Labeler** - Automatically labels pull requests
6. **Dependabot** - Keeps dependencies up to date

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:** Push to main/develop, Pull Requests, Manual dispatch

**Jobs:**
- **Pre-commit checks**: Runs all pre-commit hooks (ruff, mypy, pytest)
- **Tests**: Runs test suite on Python 3.10 and 3.12 (Ubuntu only)
- **Type checking**: Runs mypy for static type analysis
- **Linting**: Runs ruff formatter and linter
- **Build**: Creates distribution packages

**Key Features:**
- Matrix testing across Python 3.10 and 3.12
- Code coverage reporting to Codecov
- Caching with uv for faster builds
- Artifact upload for distribution packages

### 2. DCO Check (`.github/workflows/dco.yml`)

**Triggers:** Pull Requests

**Purpose:** Ensures all commits include a "Signed-off-by" line as required by the Developer Certificate of Origin (DCO).

**How to sign commits:**
```bash
git commit -s -m "Your commit message"
```

Or configure git to always sign off:
```bash
git config --global format.signoff true
```

### 3. Security Scanning (`.github/workflows/security.yml`)

**Triggers:** Pull Requests, Weekly schedule (Mondays), Manual dispatch

**Jobs:**
- **Dependency Review**: Reviews dependencies in PRs for known vulnerabilities (high severity)

### 4. Release Workflow (`.github/workflows/release.yml`)

**Triggers:** Git tags matching `v*.*.*`

**Jobs:**
- **Build and Publish**: Creates distribution packages and publishes to PyPI using trusted publishing (OIDC)
- **GitHub Release**: Creates GitHub release with artifacts

**Setup Required:**
1. Configure PyPI trusted publishing:
   - Go to PyPI project settings
   - Add GitHub as a trusted publisher
   - Specify repository and workflow file

2. No API tokens needed - uses OIDC authentication

**To create a release:**
```bash
git tag v0.3.0
git push origin v0.3.0
```

### 5. PR Labeler (`.github/workflows/pr-labeler.yml`)

**Triggers:** Pull Request events

**Purpose:** Automatically labels PRs based on changed files using `.github/labeler.yml` configuration.

**Labels:**
- `documentation` - Changes to docs or markdown files
- `tests` - Changes to test files
- `dependencies` - Changes to dependency files
- `ci/cd` - Changes to GitHub Actions or pre-commit
- `core`, `cli`, `llm-clients`, etc. - Component-specific labels

### 6. Dependabot (`.github/dependabot.yml`)

**Schedule:** Weekly on Mondays

**Purpose:** Automatically creates PRs to update dependencies.

**Monitors:**
- GitHub Actions versions
- Python package dependencies
- Groups related dependencies together

## Setup Instructions

### Initial Setup

1. **Create a feature branch to test workflows:**
   ```bash
   git checkout -b feature/github-actions-setup
   git add .github/
   git commit -s -m "feat: add GitHub Actions workflows"
   git push -u origin feature/github-actions-setup
   ```

2. **Create a Pull Request** to test the workflows before merging to main

3. **Verify workflows run successfully** in the PR

### Required GitHub Settings

#### 1. Branch Protection Rules (Settings → Branches)

For `main` branch:
- Require pull request reviews before merging
- Require status checks to pass before merging
  - Select: `Pre-commit checks`, `Test (3.10)`, `Test (3.12)`, `Type checking`, `Linting`, `DCO Check`
- Require branches to be up to date before merging
- Require signed commits (optional but recommended)
- Include administrators

#### 2. Enable GitHub Features

**Settings → Code security and analysis:**
- Dependency graph
- Dependabot alerts
- Dependabot security updates

#### 3. Secrets and Variables

**For Codecov (optional):**
- Add `CODECOV_TOKEN` secret if you want coverage reporting
- Get token from https://codecov.io/

**For PyPI Publishing:**
- No secrets needed! Uses OIDC trusted publishing
- Configure at: https://pypi.org/manage/account/publishing/

#### 4. Repository Labels

Create these labels in your repository (Settings → Labels):
- `documentation`
- `tests`
- `dependencies`
- `ci/cd`
- `core`
- `cli`
- `llm-clients`
- `exporters`
- `extractors`

## Local Development vs CI

### What runs locally (pre-commit):
```bash
uv run pre-commit run --all-files
```
- Ruff formatter
- Ruff linter
- MyPy type checking
- Pytest with coverage
- UV lock file check

### What runs in CI (additional):
- Multi-version testing (Python 3.10, 3.12)
- Security scanning
- Dependency review
- DCO verification

## Best Practices

### 1. Always Sign Your Commits
```bash
git commit -s -m "feat: add new feature"
```

### 2. Run Pre-commit Before Pushing
```bash
uv run pre-commit run --all-files
```

### 3. Keep Dependencies Updated
- Review and merge Dependabot PRs regularly
- Test dependency updates locally first

### 4. Use Conventional Commits
```bash
feat: add new feature
fix: resolve bug
docs: update documentation
chore: update dependencies
test: add test coverage
```

### 5. Test in Feature Branches
- Always create a feature branch
- Open a PR to test workflows
- Merge only after all checks pass

## Troubleshooting

### Pre-commit Checks Fail in CI

**Solution:** Run locally first:
```bash
uv run pre-commit run --all-files
```

### DCO Check Fails

**Solution:** Amend commits to add sign-off:
```bash
git commit --amend -s --no-edit
git push --force-with-lease
```

Or for multiple commits:
```bash
git rebase --signoff HEAD~3  # Last 3 commits
git push --force-with-lease
```

### Tests Fail on Specific Python Version

**Solution:** Test locally with that Python version:
```bash
# Using pyenv or similar
pyenv install 3.10
pyenv local 3.10
uv run pytest
```

### Release Workflow Fails

**Common issues:**
1. PyPI trusted publishing not configured
2. Tag format incorrect (must be `v*.*.*`)
3. Missing permissions in workflow

## Monitoring

### View Workflow Runs
- Go to **Actions** tab in GitHub
- Click on specific workflow to see runs
- Click on a run to see job details and logs

### Enable Notifications
- Settings → Notifications
- Configure for workflow failures
- Set up Slack/Discord webhooks if needed

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [DCO Information](https://developercertificate.org/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)