---
name: session-handoff
description: Use when the user asks to snapshot, save, or hand off the current Claude Code session so a fresh session can resume with full context. Triggers include "session handoff", "save this session", "write up where we are", "context dump", "hand this off", "snapshot the session", "running low on context", or any request to record progress so another session can pick it up.
argument-hint: [optional output directory or filename]
---

# Session Handoff

Capture everything that happened in the current session into a single dense Markdown file so a new Claude Code session (typically a long 1M-context run that is filling up) can be handed the file and resume with no loss of context.

It captures, from the conversation already in context: the original task and goals, every subtask and its progress, commands run and their key outputs, a discussion summary, PR and Linear/Jira status, decisions and tradeoffs, open questions and blockers, and concrete next steps.

Synthesize from the conversation already in context. Do not re-derive facts you do not have; record what actually happened. Do not invent progress, outputs, or decisions.

## Arguments

- `$ARGUMENTS` - Optional: an output directory, or a full file path/name. If omitted, infer the directory (see below) and generate the filename.

## Step 1: Determine the output location

**Directory** -- infer from session context, in this priority order:
1. An explicit directory or path in `$ARGUMENTS` (use it verbatim, even if it does not match the session's subject repo).
2. The repo or working directory the session has centered on (most-edited files, the repo of the active branch/PR, a project working dir like `~/Klaviyo/Repos/<repo>`, `Repos/msk/`, `Repos/hipaa/`, etc.).
3. If the session maps to a tracked project (e.g. a memory `project_*` doc names a living-findings dir), use that dir.

If the directory is genuinely ambiguous (work spanned several repos with no clear primary, or nothing was edited), **ask the user** for the output directory instead of guessing. State your best guess as the first option.

**Filename** -- run `date` first to get the real date, then use:
`SESSION-HANDOFF-<short-topic-slug>-<YYYY-MM-DD>.md`
(e.g. `SESSION-HANDOFF-msk-migration-2026-06-22.md`). If `$ARGUMENTS` gave a filename, use it.

If a file of that name already exists, ask before overwriting; offer to append a `-2` suffix or merge instead.

**Non-interactive fallback.** If you cannot prompt the user (e.g. running as a background agent): for an ambiguous directory, use your best-guess directory and state that assumption at the top of the file; for an existing filename, append a `-2` suffix rather than overwriting.

## Step 2: Scan the whole session

Walk the entire conversation from the first message. Pull out, in order of appearance:
- The original task / ask, verbatim intent.
- Every concrete subtask and its current state.
- CLI commands run and the relevant parts of their output (trim noise, keep signal: IDs, counts, error messages, paths, exit results).
- Decisions made and the reasoning/tradeoffs behind them.
- PRs touched (number, repo, URL, draft/merged state, what they change).
- Linear / Jira items touched (ID, status, what changed).
- Open questions, blockers, and what was explicitly deferred.
- Auth/role, env vars, working dirs, and any setup needed to resume.

## Step 3: Write the file

Use the template below. Front-load the "Resume Here" block so the next session gets oriented in seconds, then follow with full detail. The file may be long -- completeness beats brevity for the document as a whole -- but the writing inside stays succinct and dense.

```markdown
# Session Handoff: <topic>

**Date:** <YYYY-MM-DD> · **Working dir(s):** <paths> · **Model/session:** <if relevant>

## Resume Here (read this first)
- **Goal:** <one or two sentences: what we are ultimately trying to do>
- **Status:** <where things stand right now>
- **Next steps:** <ordered, concrete actions the next session should take>
- **Blockers / open questions:** <what is unresolved and who/what it depends on>
- **To resume, you need:** <auth role(s), env vars, branches, dirs, e.g. `s2a-login --role ...`, branch `shuyangw/...`>

## Original Task
<verbatim ask and any clarifications, framed succinctly>

## Subtasks & Progress
- [x] <done item> -- <one-line outcome>
- [ ] <pending item> -- <what's left, where it stands>

## Key Decisions & Tradeoffs
- **<decision>:** <chosen path>. Why: <reason>. Tradeoff/risk: <what we gave up or accepted>.

## Discussion Summary
<dense summary of important back-and-forth, findings, dead ends, and conclusions. Skip pleasantries.>

## Commands & Outputs
Relevant commands run and the signal from their output.
\`\`\`
$ <command>
<trimmed relevant output: IDs, counts, errors, results>
\`\`\`

## PRs
- <repo>#<num> (<draft|open|merged>) <url> -- <what it changes / review state>

## Linear / Jira
- <ID> (<status>) -- <what changed / what it tracks>

## Files Touched
- `<absolute path>` -- <what changed and why>

## Key Takeaways & Gotchas
- <non-obvious facts, gotchas, things that bit us, things to not repeat>

## References
- <links: Confluence, Slack threads, dashboards, docs>
```

Omit any section that has no content rather than leaving an empty heading. Add sections if the session warrants (e.g. "Test Plan", "Infra State").

## Writing rules
- **Succinct and dense.** Telegraphic phrasing, fragments over full sentences where it reads fine. No filler, no recap of these instructions.
- **Verbatim where it matters.** Exact commands, exact error strings, exact IDs and paths. Trim long outputs to the load-bearing lines.
- **Real paths and links.** Absolute local paths; PR/Linear by ID with URL. If the conversation only gave a partial or glob path (e.g. `src/payments/*.ts`), record it verbatim and note "exact names not captured". Never fabricate concrete filenames to satisfy this rule.
- **No fabrication.** If something is unknown or was never confirmed, say so explicitly ("not verified", "unknown") rather than guessing.
- **No em dashes** (per the user's writing-style rule). Use commas, parentheses, or separate sentences.
- The point of the doc is to let a fresh session continue without re-asking. When in doubt about whether a detail helps resumption, include it.

## After writing
Before reporting done, scan the finished file for em dashes (the `—` character) and replace each with a comma, parentheses, `--`, or a separate sentence. Agents tend to introduce em dashes in prose even when told not to, so this final pass is a check, not a reminder.

Then print the absolute path of the file written and a one-line summary of what it covers. Do not commit it unless the user asks.
