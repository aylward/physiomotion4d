#!/usr/bin/env python
"""
claude_github_reviews.py — Screen GitHub PR review comments with Claude.

Workflow:
  1. Fetch all inline review comments and PR-level review bodies via gh CLI.
  2. Build a structured prompt that includes file paths, line numbers, diff
     hunks, and suggestion blocks for every comment.
  3. Invoke Claude Code non-interactively; Claude reads CLAUDE.md and AGENTS.md,
     reads the referenced source files for context, and for each comment decides
     APPLY / REVISE / REJECT with explicit reasoning against project conventions.
  4. Accepted edits are applied as pending working-tree changes (not committed).
  5. A Markdown summary tagged by PR number is written to the repo root.

Usage:
  py utils/claude_github_reviews.py --pr 42
  py utils/claude_github_reviews.py --pr 42 --repo owner/repo
  py utils/claude_github_reviews.py --pr 42 --dry-run
  py utils/claude_github_reviews.py --pr 42 --since-last-push --dry-run

  With --since-last-push, only inline comments and PR-level reviews created after
  the latest reflog time for refs/remotes/<remote>/<PR_head_branch> are included.
  That time is when this clone last saw the remote ref move (push or fetch), not
  necessarily the exact server push timestamp.

Requirements:
  - gh CLI (GitHub CLI) — not a Python package; install separately:
      Windows: winget install GitHub.cli
      Then authenticate: gh auth login
  - Claude Code CLI — https://claude.ai/code
      Windows: winget install Anthropic.ClaudeCode
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Git / repo helpers
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        )
        return Path(out.stdout.strip())
    except subprocess.CalledProcessError:
        print("[ERROR] Not inside a Git repository.")
        sys.exit(1)


def get_repo_slug(repo_root: Path) -> str:
    """Derive owner/repo from the git remote origin URL."""
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
        )
        remote = out.stdout.strip()
    except subprocess.CalledProcessError:
        print("[ERROR] Could not read git remote origin.")
        sys.exit(1)
    m = re.search(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?$", remote)
    if not m:
        print(f"[ERROR] Cannot parse owner/repo from remote: {remote}")
        sys.exit(1)
    return m.group(1)


def parse_github_datetime(iso_str: str) -> datetime:
    """Parse GitHub API timestamps (may end with Z)."""
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _parse_reflog_timestamp(line: str) -> datetime:
    """Parse the @{...} timestamp from one `git reflog show -1` line."""
    m = re.search(r"@\{([^}]+)\}", line)
    if not m:
        raise ValueError(f"no reflog timestamp in line: {line!r}")
    return datetime.fromisoformat(m.group(1).strip())


def get_remote_reflog_cutoff(repo_root: Path, remote: str, head_ref: str) -> datetime:
    """
    Latest reflog time for refs/remotes/<remote>/<head_ref> (when the ref last
    moved in this clone).
    """
    ref = f"refs/remotes/{remote}/{head_ref}"
    cmd = ["git", "reflog", "show", "-1", "--date=iso-strict", ref]
    try:
        out = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
            encoding="utf-8",
        )
    except FileNotFoundError:
        print('[ERROR] "git" not found in PATH.')
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        print(f"[ERROR] Could not read reflog for {ref}.")
        if err:
            print(f"        {err}")
        print(f"        Hint: git fetch {remote} {head_ref}")
        sys.exit(1)

    line = out.stdout.strip().splitlines()
    if not line or not line[0].strip():
        print(f"[ERROR] Empty reflog for {ref}.")
        print(f"        Hint: git fetch {remote} {head_ref}")
        sys.exit(1)

    try:
        return _parse_reflog_timestamp(line[0])
    except ValueError as exc:
        print(f"[ERROR] Could not parse reflog timestamp: {exc}")
        sys.exit(1)


def filter_since_cutoff(
    inline_comments: list[dict],
    reviews: list[dict],
    cutoff: datetime,
) -> tuple[list[dict], list[dict]]:
    """
    Keep inline comments with created_at > cutoff and reviews with
    submitted_at > cutoff (submitted_at required).
    """
    filtered_inline: list[dict] = []
    for c in inline_comments:
        created = c.get("created_at")
        if not created:
            continue
        if parse_github_datetime(created) > cutoff:
            filtered_inline.append(c)

    filtered_reviews: list[dict] = []
    for r in reviews:
        submitted = r.get("submitted_at")
        if not submitted:
            continue
        if parse_github_datetime(submitted) > cutoff:
            filtered_reviews.append(r)

    return filtered_inline, filtered_reviews


# ---------------------------------------------------------------------------
# GitHub API helpers (via gh CLI)
# ---------------------------------------------------------------------------


def _gh_api(endpoint: str, *, paginate: bool = False) -> list | dict:
    """Call `gh api` and return parsed JSON. Merges paginated arrays."""
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd.append(endpoint)
    try:
        out = subprocess.run(
            cmd, check=True, text=True, capture_output=True, encoding="utf-8"
        )
    except FileNotFoundError:
        print('[ERROR] "gh" CLI not found. Install from https://cli.github.com')
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() if exc.stderr else ""
        print(f"[ERROR] gh api {endpoint} failed.")
        if msg:
            print(f"        {msg}")
        sys.exit(exc.returncode)

    raw = out.stdout.strip()
    if not raw:
        return []

    # gh --paginate emits concatenated JSON arrays; normalize to one list.
    if paginate:
        try:
            merged: list = []
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(raw):
                obj, pos = decoder.raw_decode(raw, pos)
                if isinstance(obj, list):
                    merged.extend(obj)
                else:
                    merged.append(obj)
                while pos < len(raw) and raw[pos] in " \t\n\r":
                    pos += 1
            return merged
        except json.JSONDecodeError:
            pass  # Fall through to standard parse

    return json.loads(raw)


def fetch_pr_data(pr_number: int, repo: str) -> dict:
    result = _gh_api(f"repos/{repo}/pulls/{pr_number}")
    assert isinstance(result, dict)
    return result


def fetch_inline_comments(pr_number: int, repo: str) -> list[dict]:
    result = _gh_api(f"repos/{repo}/pulls/{pr_number}/comments", paginate=True)
    return result if isinstance(result, list) else []


def fetch_reviews(pr_number: int, repo: str) -> list[dict]:
    result = _gh_api(f"repos/{repo}/pulls/{pr_number}/reviews", paginate=True)
    return result if isinstance(result, list) else []


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _extract_suggestion(body: str) -> tuple[str, str]:
    """
    Split a GitHub review comment body into (suggestion_code, note_text).

    GitHub / CodeRabbit suggestion blocks use the fenced form:
        ```suggestion
        <replacement lines>
        ```
    Returns ('', body) when no suggestion block is present.
    """
    pattern = re.compile(r"```suggestion\r?\n(.*?)```", re.DOTALL)
    m = pattern.search(body)
    if not m:
        return "", body
    suggestion = m.group(1).rstrip("\n")
    note = pattern.sub("", body).strip()
    return suggestion, note


def _format_review_bodies(reviews: list[dict]) -> str:
    relevant = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    if not relevant:
        return ""
    parts = ["## PR-level review comments\n"]
    for i, r in enumerate(relevant, 1):
        reviewer = r["user"]["login"]
        state = r.get("state", "")
        parts.append(f"### Review {i} — {reviewer} ({state})")
        parts.append(r["body"].strip())
        parts.append("")
    return "\n".join(parts)


def _format_inline_comments(comments: list[dict]) -> str:
    if not comments:
        return ""
    parts = ["## Inline review comments\n"]
    for i, c in enumerate(comments, 1):
        reviewer = c["user"]["login"]
        path = c["path"]
        line = c.get("line") or c.get("original_line", "?")
        diff_hunk = c.get("diff_hunk", "")
        suggestion, note = _extract_suggestion(c["body"])

        parts.append(f"### Inline comment {i}")
        parts.append(f"- **Reviewer:** `{reviewer}`")
        parts.append(f"- **File:** `{path}`")
        parts.append(f"- **Line:** {line}")
        if diff_hunk:
            parts.append(f"\nDiff context:\n```diff\n{diff_hunk}\n```")
        if suggestion:
            parts.append("\nSuggested replacement (replaces the highlighted lines):")
            parts.append(f"```\n{suggestion}\n```")
        if note:
            parts.append(f"\nReviewer note: {note}")
        parts.append("")
    return "\n".join(parts)


def build_prompt(
    pr_number: int,
    pr_data: dict,
    reviews: list[dict],
    inline_comments: list[dict],
    summary_filename: str,
) -> str:
    title = pr_data.get("title", f"PR #{pr_number}")
    branch = pr_data.get("head", {}).get("ref", "unknown")
    base = pr_data.get("base", {}).get("ref", "unknown")

    review_bodies = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    total = len(inline_comments) + len(review_bodies)

    comments_block = "\n".join(
        filter(
            None,
            [
                _format_review_bodies(reviews),
                _format_inline_comments(inline_comments),
            ],
        )
    )

    return textwrap.dedent(f"""\
        You are screening GitHub PR #{pr_number}: "{title}"
        Branch: `{branch}` -> `{base}`
        Total comments to assess: {total}

        ## Step 1 — Read project standards

        Before assessing any comment, read these files:
        - `CLAUDE.md` — coding standards, architecture, working process
        - `AGENTS.md` — role expectations for this codebase
        - If any comment touches `vtk_to_usd/`:
          `src/physiomotion4d/vtk_to_usd/CLAUDE.md`

        ## Step 2 — Assess each comment

        For every comment below, in order:

        1. Read the referenced source file (`path`, near `line`) to understand
           the full context — do not rely solely on the diff hunk.
        2. Decide:
           - **APPLY** — suggestion is correct and consistent with CLAUDE.md.
             Apply it as-is using the Edit tool.
           - **REVISE** — directionally right but conflicts with repo conventions.
             Apply your corrected version using the Edit tool.
           - **REJECT** — wrong, unnecessary, or conflicts with explicit project rules.
             Do not edit the file. State the specific rule or reason.
        3. For APPLY / REVISE: use the Edit tool to make the change.
           Do NOT run git add, git commit, or any git staging commands.
           Leave all edits as pending working-tree modifications only.

        Rejection triggers (from CLAUDE.md — treat these as hard rules):
        - Introduces `X | None` instead of `Optional[X]` (ruff UP007 is suppressed)
        - Adds backward-compat shims, re-exports, or removed-symbol stubs
        - Adds error handling for internal states that cannot happen
        - In classes that inherit from `PhysioMotion4DBase`, uses `print()` instead
          of `self.log_info()` / `self.log_debug()`
        - New runtime workflow / segmentation / registration class does not inherit
          from `PhysioMotion4DBase`; helper/data/container classes need not
        - Adds features or abstractions beyond what was requested
        - Calls `vtk_to_usd` internals from outside `convert_vtk_to_usd.py`
        - Applies coordinate conversion (RAS->Y-up) more than once
        - Exceeds 88-character line length

        ## Step 3 — Write summary

        After processing all {total} comment(s), write `{summary_filename}`
        to the repository root using this exact structure:

        ```markdown
        # PR #{pr_number} Review Summary

        **PR:** {title}
        **Branch:** `{branch}` -> `{base}`
        **Reviewed:** <today's date>
        **Comments processed:** {total}

        ## Results

        | # | File | Line | Reviewer | Decision | Reasoning |
        |---|------|------|----------|----------|-----------|
        | 1 | `path/to/file.py` | 42 | reviewer | APPLIED | one sentence |

        ## Applied changes

        For each APPLIED or REVISED item: file name, what changed, and why it
        was accepted or how it was adjusted.

        ## Rejected suggestions

        For each REJECTED item: what was suggested, and the specific CLAUDE.md
        rule or reasoning that led to rejection.

        ## Observations

        Patterns across the review (recurring style disagreements, areas where
        the reviewer's model of the codebase differs from reality, etc.).
        ```

        ---

        {comments_block}
    """)


# ---------------------------------------------------------------------------
# Claude invocation
# ---------------------------------------------------------------------------


def invoke_claude(prompt: str, repo_root: Path) -> None:
    """
    Invoke Claude Code non-interactively via stdin.

    Uses stdin rather than a CLI argument to avoid Windows CreateProcess
    command-line length limits (~32 KB). Claude's output streams to the
    terminal so the developer can follow along.
    """
    print("[*] Invoking Claude Code to screen review comments...")
    print("    Claude will read source files, assess each suggestion, apply")
    print("    accepted edits, and write the summary. This may take a minute.\n")

    try:
        subprocess.run(
            ["claude", "--print", "--allowedTools", "Read,Edit,Glob,Grep"],
            input=prompt,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
            check=True,
        )
    except FileNotFoundError:
        print('[ERROR] "claude" CLI not found.')
        print("        Install Claude Code: https://claude.ai/code")
        _save_prompt_fallback(prompt, repo_root)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f'[ERROR] "claude" exited with status {exc.returncode}.')
        _save_prompt_fallback(prompt, repo_root)
        sys.exit(exc.returncode)


def _save_prompt_fallback(prompt: str, repo_root: Path) -> None:
    fallback = repo_root / ".claude_review_prompt.txt"
    fallback.write_text(prompt, encoding="utf-8")
    print("[*] Prompt saved to .claude_review_prompt.txt")
    print("    Run manually with:")
    print(
        '      claude --print --allowedTools "Read,Edit,Glob,Grep" '
        "< .claude_review_prompt.txt"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="claude_github_reviews.py",
        description=(
            "Screen GitHub PR review comments with Claude Code and apply "
            "accepted suggestions as pending working-tree changes."
        ),
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        metavar="NUMBER",
        help="Pull request number",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        default=None,
        help="GitHub repo slug (default: inferred from git remote origin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt that would be sent to Claude and exit without changes",
    )
    parser.add_argument(
        "--since-last-push",
        action="store_true",
        dest="since_last_push",
        help=(
            "Only include inline comments and reviews after the latest reflog time "
            "for refs/remotes/<remote>/<PR_head_branch> (this clone)"
        ),
    )
    parser.add_argument(
        "--remote",
        metavar="NAME",
        default="origin",
        help="Git remote name for reflog (default: origin; used with --since-last-push)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    repo: str = args.repo or get_repo_slug(repo_root)
    pr_number: int = args.pr
    summary_filename = f"pr_{pr_number}_review_summary.md"

    print(f"[*] Repository : {repo}")
    print(f"[*] PR         : #{pr_number}")
    print(f"[*] Summary    : {summary_filename}")
    print()

    print("[*] Fetching PR metadata...")
    pr_data = fetch_pr_data(pr_number, repo)
    print(f'    "{pr_data.get("title", "")}"')

    print("[*] Fetching inline review comments...")
    inline_comments = fetch_inline_comments(pr_number, repo)
    print(f"    {len(inline_comments)} inline comment(s)")

    print("[*] Fetching PR-level reviews...")
    reviews = fetch_reviews(pr_number, repo)
    review_bodies = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    print(f"    {len(review_bodies)} review(s) with body text")

    if args.since_last_push:
        head_ref = pr_data.get("head", {}).get("ref")
        if not head_ref:
            print("[ERROR] PR has no head branch ref; cannot use --since-last-push.")
            sys.exit(1)
        remote_ref = f"refs/remotes/{args.remote}/{head_ref}"
        cutoff = get_remote_reflog_cutoff(repo_root, args.remote, head_ref)
        print()
        print("[*] --since-last-push")
        print(f"    Remote ref : {remote_ref}")
        print(f"    Cutoff     : {cutoff.isoformat()}")
        inline_comments, reviews = filter_since_cutoff(inline_comments, reviews, cutoff)
        review_bodies = [
            r
            for r in reviews
            if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
        ]
        print(
            f"    After filter: {len(inline_comments)} inline comment(s), "
            f"{len(review_bodies)} review(s) with body text"
        )

    total = len(inline_comments) + len(review_bodies)
    if total == 0:
        print("\n[*] No review comments found. Nothing to do.")
        sys.exit(0)

    print(f"\n[*] Building prompt for {total} comment(s)...")
    prompt = build_prompt(
        pr_number=pr_number,
        pr_data=pr_data,
        reviews=reviews,
        inline_comments=inline_comments,
        summary_filename=summary_filename,
    )

    if args.dry_run:
        separator = "=" * 60
        if args.since_last_push:
            print()
            print(
                "[dry-run] --since-last-push: using cutoff and counts above "
                "(full prompt follows)."
            )
        print(f"\n{separator}")
        print("PROMPT (dry run — not sent to Claude)")
        print(separator)
        print(prompt)
        print(separator)
        print("\n[dry-run] No files changed.")
        print(f"[dry-run] Summary would be written to: {summary_filename}")
        sys.exit(0)

    invoke_claude(prompt, repo_root)

    print()
    summary_path = repo_root / summary_filename
    if summary_path.exists():
        print(f"[✓] Summary written : {summary_filename}")
    else:
        print("[!] Summary file not found — check Claude output above.")
    print("[*] Inspect changes : git diff")
    print("[*] Stage selectively: git add -p")


if __name__ == "__main__":
    main()
