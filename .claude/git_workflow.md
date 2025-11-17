# Git Workflow Guidelines

## Core Principle

**NEVER push to remote repository without explicit user permission.**

## Workflow Steps

### 1. Making Changes
- Edit files as needed
- Test changes thoroughly
- Run unit tests if applicable

### 2. Staging Changes
```bash
git add <file_path>
# OR
git add .  # Stage all changes
```

### 3. Creating Commits
- Create clear, descriptive commit messages
- Include context about what changed and why
- Follow project commit message format:

```bash
git commit -m "$(cat <<'EOF'
Brief summary of change (50 chars or less)

More detailed explanation if needed:
- What changed
- Why it changed
- Impact of the change

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 4. Pushing to Remote

**CRITICAL**: Before running `git push`, you MUST:
1. Ask the user: "Ready to push these changes to remote?"
2. Wait for explicit confirmation
3. Only push if user confirms

**Example workflow:**
```
Assistant: I've committed the timezone fixes. Ready to push these changes to remote?
User: Yes
Assistant: [runs git push]
```

**Never assume permission to push, even if:**
- The changes seem minor
- You just pushed earlier in the session
- The user asked you to "finish the task"

## Git Commands Reference

### Safe Commands (No Permission Needed)
```bash
git status              # Check repo status
git diff               # View changes
git log                # View commit history
git add <file>         # Stage changes
git commit -m "..."    # Commit changes
git branch             # List branches
```

### Commands Requiring Permission
```bash
git push               # ❌ ALWAYS ASK FIRST
git push --force       # ❌ ALWAYS ASK FIRST (even more dangerous)
git push origin <branch>  # ❌ ALWAYS ASK FIRST
```

## Commit Message Best Practices

### Format
1. **First line**: Brief summary (50 characters max)
2. **Blank line**
3. **Body**: Detailed explanation (wrap at 72 characters)
4. **Blank line**
5. **Footer**: Generated attribution

### Examples

**Good commit message:**
```
Fix timezone bug in OMR strategy execution timing

Updated run_live_paper_trading.py to convert UTC to EST before
comparing schedule times. This ensures exit at 9:31 AM EST and
entry at 3:50 PM EST trigger correctly.

- Added pytz timezone conversion to should_run_now()
- Fixed execution_times schedule format handling
- Fixed specific_time schedule format handling

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Bad commit message:**
```
fixed stuff
```

### Multi-file Commits

When committing multiple related changes:
```bash
git add file1.py file2.py file3.py
git commit -m "Descriptive message about the overall change"
```

When committing unrelated changes, create separate commits:
```bash
git add file1.py
git commit -m "Fix bug in file1"

git add file2.py
git commit -m "Add new feature in file2"
```

## Deployment Workflow

When deploying to production (e.g., EC2):

1. Commit changes locally
2. **Ask user**: "Ready to push and deploy to EC2?"
3. If confirmed:
   ```bash
   git push origin main
   ssh -i ~/.ssh/key.pem user@host "cd repo && git pull && restart_service"
   ```

## Emergency Situations

Even in urgent bug fixes or production issues:
- Commit immediately
- **Ask before pushing**: "Critical fix ready. Push to production now?"
- Wait for confirmation

## Summary Checklist

Before `git push`:
- [ ] Changes tested
- [ ] Commit message is clear and descriptive
- [ ] Attribution footer added
- [ ] **User permission obtained** ✅

## Anti-Patterns to Avoid

❌ **Never do this:**
```bash
# Committing and pushing without asking
git add . && git commit -m "updates" && git push
```

✅ **Always do this:**
```bash
# Commit, then ask
git add . && git commit -m "Clear description of changes"
# [Ask user for permission]
# [Wait for confirmation]
# git push origin main
```

## Related Guidelines

- See [`.claude/code_standards.md`](.claude/code_standards.md) for code quality standards
- See [`.claude/testing.md`](.claude/testing.md) for testing before commits
- See [`.claude/documentation.md`](.claude/documentation.md) for doc updates
