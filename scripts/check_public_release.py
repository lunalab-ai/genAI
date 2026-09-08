#!/usr/bin/env python3
"""Scan a staged public release for answer keys, private paths, and obvious secrets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_PATH_PARTS = {
    "instructor", "professor", "source_materials", "textbook_scans",
    "publisher_code", "publisher_data", "briefs", "solutions", "answers"
}
FORBIDDEN_NOTEBOOK_TAGS = {"solution", "answer", "instructor-only"}
FORBIDDEN_MARKERS = (
    "BEGIN " + "SOLUTION",
    "INSTRUCTOR " + "ONLY",
    "교수자 " + "전용",
    "DO NOT " + "PUBLISH",
)
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "generic secret assignment": re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".csv", ".r", ".qmd", ".rmd", ".ipynb", ".html"}


@dataclass
class Finding:
    path: Path
    message: str


def scan(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path in ([root] if root.is_file() else sorted(root.rglob("*"))):
        if not path.is_file():
            continue
        rel = Path(path.name) if root.is_file() else path.relative_to(root)
        lower_parts = {part.lower() for part in rel.parts}
        bad_parts = lower_parts.intersection(FORBIDDEN_PATH_PARTS)
        if bad_parts:
            findings.append(Finding(rel, f"Forbidden public path component: {sorted(bad_parts)[0]}"))
        if path.name in {".env", "auth.json"} or path.suffix.lower() in {".pem", ".key"}:
            findings.append(Finding(rel, "Secret-bearing file type/name"))
        if path.stat().st_size > 50 * 1024 * 1024:
            findings.append(Finding(rel, "File exceeds 50 MB"))

        if path.suffix.lower() == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                findings.append(Finding(rel, f"Invalid notebook JSON: {exc}"))
                continue
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    findings.append(Finding(rel, f'Cell {index} contains saved execution output/count'))
                tags = set(cell.get("metadata", {}).get("tags", []))
                private = tags.intersection(FORBIDDEN_NOTEBOOK_TAGS)
                if private:
                    findings.append(Finding(rel, f"Cell {index} contains private tag(s): {sorted(private)}"))

        if path.suffix.lower() == '.pdf':
            try:
                from pypdf import PdfReader

                reader = PdfReader(path)
                text = '\n'.join(page.extract_text() or '' for page in reader.pages)
                if not text.strip():
                    findings.append(Finding(rel, 'PDF has no extractable text'))
            except Exception as exc:
                findings.append(Finding(rel, f'Cannot inspect PDF: {exc}'))
                continue
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    findings.append(Finding(rel, f'Forbidden marker: {marker}'))
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append(Finding(rel, f'Possible {label}'))

        if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 5 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    findings.append(Finding(rel, f"Forbidden marker: {marker}"))
            if re.search(r'https://github\.com/[^/\s]+/[^/\s]*-authoring(?:/|\b)', text):
                findings.append(Finding(rel, 'Private authoring repository URL in public material'))
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append(Finding(rel, f"Possible {label}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    findings = scan(args.root)
    for finding in findings:
        print(f"ERROR: {finding.path}: {finding.message}")
    if findings:
        print(f"Public release scan failed with {len(findings)} finding(s).", file=sys.stderr)
        return 1
    print(f"Public release scan passed: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
