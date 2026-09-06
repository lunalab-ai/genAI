#!/usr/bin/env python3
"""Validate a Markdown/asset folder before packaging it for Notion import."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

MARKDOWN_LINK_RE = re.compile(r"(?P<image>!)?\[(?P<label>[^\]]*)\]\((?P<target>[^)]+)\)")
ALLOWED_SUFFIXES = {
    ".md", ".markdown", ".txt", ".html", ".htm", ".csv", ".tsv",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"
}
IGNORED_HIDDEN = {".gitkeep"}


@dataclass
class Finding:
    level: str
    path: Path
    message: str


def validate(root: Path, *, strict: bool = False) -> list[Finding]:
    findings: list[Finding] = []
    root = root.resolve()
    if not root.is_dir():
        return [Finding("ERROR", root, "Not a directory")]

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        hidden_parts = [part for part in rel.parts if part.startswith(".") and part not in IGNORED_HIDDEN]
        if hidden_parts:
            findings.append(Finding("ERROR", rel, f"Hidden path can break ZIP import: {hidden_parts[0]}"))
        if path.is_file():
            if path.suffix.lower() and path.suffix.lower() not in ALLOWED_SUFFIXES:
                level = "ERROR" if strict else "WARN"
                findings.append(Finding(level, rel, f"Nonstandard file extension: {path.suffix}"))
            if path.stat().st_size > 50 * 1024 * 1024:
                findings.append(Finding("ERROR", rel, "File is larger than 50 MB; split or externalize it"))

    for md_path in sorted(root.rglob("*.md")):
        rel_md = md_path.relative_to(root)
        text = md_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            is_image = bool(match.group("image"))
            label = match.group("label").strip()
            raw_target = match.group("target").strip().strip("<>")
            if is_image and not label:
                findings.append(Finding("ERROR", rel_md, f"Image at character {match.start()} has empty alt text"))

            parsed = urlparse(raw_target)
            if parsed.scheme in {"http", "https", "mailto"}:
                continue
            if parsed.scheme == "file":
                findings.append(Finding("ERROR", rel_md, f"file:// link is not portable: {raw_target}"))
                continue
            if raw_target.startswith("#"):
                level = "ERROR" if strict else "WARN"
                findings.append(Finding(level, rel_md, f"Anchor-only link may not import cleanly: {raw_target}"))
                continue

            target_without_fragment = unquote(raw_target.split("#", 1)[0].split("?", 1)[0])
            if not target_without_fragment:
                continue
            target_path = Path(target_without_fragment)
            if target_path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", target_without_fragment):
                findings.append(Finding("ERROR", rel_md, f"Absolute local path is not portable: {raw_target}"))
                continue
            resolved = (md_path.parent / target_path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                findings.append(Finding("ERROR", rel_md, f"Link escapes bundle root: {raw_target}"))
                continue
            if not resolved.exists():
                findings.append(Finding("ERROR", rel_md, f"Missing local target: {raw_target}"))
            if "#" in raw_target:
                level = "ERROR" if strict else "WARN"
                findings.append(Finding(level, rel_md, f"Fragment/anchor may not survive Notion import: {raw_target}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    findings = validate(args.root, strict=args.strict)
    for finding in findings:
        print(f"{finding.level}: {finding.path}: {finding.message}")
    errors = [item for item in findings if item.level == "ERROR"]
    if errors:
        print(f"Validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"Validation passed: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
