"""Course-independent build, inspect, handoff and allowlist staging commands."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import mimetypes
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml
from markdown_it import MarkdownIt

from check_public_release import scan
from validate_notion_bundle import validate

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = ("course", "notebooks/student", "labs/student", "src", "data/sample")
PUBLIC_FILES = (
    "README.md",
    "scripts/course.py",
    "scripts/check_public_release.py",
    "scripts/validate_notion_bundle.py",
    "requirements.txt",
    ".github/workflows/materials.yml",
)
CSS = """@page {size:A4;margin:16mm} body{font-family:'Noto Sans CJK KR','Malgun Gothic',sans-serif;font-size:10pt;line-height:1.65;color:#172333}
h1{font-size:21pt} h2{font-size:15pt} h1,h2,h3{break-after:avoid} img{max-width:100%;max-height:220mm;display:block;margin:12px auto}
table{width:100%;border-collapse:collapse;font-size:9pt} th,td{border:1px solid #ccd5df;padding:6px} th{background:#edf2f7}
thead{display:table-header-group} tr,img{break-inside:avoid} pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:10px}
blockquote{border-left:3px solid #a4bbd4;margin-left:0;padding-left:12px} a{color:#185c99} p,li{orphans:3;widows:3}"""


def safe(root: Path, relative: str) -> Path:
    p = Path(relative)
    if (
        not relative
        or p.is_absolute()
        or ".." in p.parts
        or any(x.lower() == ".git" for x in p.parts)
    ):
        raise ValueError(f"Unsafe path: {relative}")
    resolved = (root / p).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def load(root=ROOT):
    config = yaml.safe_load((root / "course/course.yml").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", config["repository"]):
        raise ValueError("Invalid repository")
    ids = [s["id"] for s in config["sessions"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate session IDs")
    for s in config["sessions"]:
        if not re.fullmatch(r"w\d{2}[a-z]?", s["id"]):
            raise ValueError("Invalid session ID")
        if s.get("date"):
            dt.date.fromisoformat(str(s["date"]))
    return config


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root):
    h = hashlib.sha256()
    for folder in ("course", "notebooks/source", "labs", "src", "data/sample", "scripts"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and "handouts" not in p.parts:
                h.update(p.relative_to(root).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def render(text, source, root, repository):
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(text)
    for t in tokens:
        for c in t.children or []:
            key = "src" if c.type == "image" else "href" if c.type == "link_open" else None
            if not key:
                continue
            target = c.attrGet(key) or ""
            if c.type == "image":
                if urlparse(target).scheme:
                    raise ValueError("PDF requires local images")
                p = (source.parent / unquote(target)).resolve()
                p.relative_to((root / "course").resolve())
                mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                c.attrSet(key, f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode())
            elif target and not urlparse(target).scheme and not target.startswith("#"):
                p = (source.parent / unquote(target)).resolve()
                rel = p.relative_to(root.resolve()).as_posix()
                c.attrSet(key, f"https://github.com/{repository}/blob/main/{rel}")
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8"><style>'
        + CSS
        + "</style><body>"
        + md.renderer.render(tokens, md.options, {})
        + "</body></html>"
    )


def quiz(root, session):
    data = yaml.safe_load(safe(root, session["quiz"]).read_text(encoding="utf-8"))
    source_path = safe(root, data["source"])
    source = source_path.read_text(encoding="utf-8")
    if source_path.suffix == ".ipynb":
        source = "\n".join("".join(c["source"]) for c in json.loads(source)["cells"])
    if not 5 <= len(data["items"]) <= 10:
        raise ValueError("Expected 5–10 checkpoint questions")
    parts = ["# " + session["title"] + " · 퀴즈 해설", data["provenance"]]
    for i, item in enumerate(data["items"], 1):
        if item["question"] not in source or not item["answer"].strip():
            raise ValueError(f"Quiz/source mismatch: {i}")
        parts.extend([f"## {i}. {item['question']}", item["answer"]])
    p = root / "course/handouts" / f"{session['id']}-quiz.md"
    write(p, "\n\n".join(parts) + "\n")
    return p


def hub(root, config):
    overview = safe(root, config["overview"]).read_text(encoding="utf-8")
    parts = ["# " + config["title"], config["description"]]
    if config.get("hero"):
        safe(root, config["hero"]).read_bytes()
        parts.append(f"![{config['title']}]({config['hero']})")
    parts.extend(
        [
            "## 수업 자료",
            "",
            "| 차시 | 날짜 | 주제 | 강의노트 | 퀴즈 해설 | 실습 | 실행 |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for s in config["sessions"]:
        notes = []
        for p in s["pages"]:
            notes.append(f"[MD]({p}) · [PDF](course/handouts/{Path(p).stem}.pdf)")
        if s.get("notion_url"):
            notes.append(f"[Notion]({s['notion_url']})")
        lab = " · ".join(f"[{Path(p).suffix[1:]}]({p})" for p in s.get("labs", [])) or "—"
        run = f"[Colab]({s['colab_url']})" if s.get("colab_url") else s.get("runtime_label", "—")
        q = (
            f"[MD](course/handouts/{s['id']}-quiz.md) · [PDF](course/handouts/{s['id']}-quiz.pdf)"
            if s.get("quiz")
            else "—"
        )
        parts.append(
            "| "
            + " | ".join(
                [
                    s["id"],
                    str(s.get("date") or "日程未定").replace("日程未定", "일정 미확정"),
                    s["title"].replace("|", "/"),
                    " · ".join(notes) or "—",
                    q,
                    lab,
                    run,
                ]
            )
            + " |"
        )
    if config.get("supplements"):
        parts.extend(["", "## 보충 실습 자료", "", "| 자료 | 실습 | 퀴즈 해설 |", "|---|---|---|"])
        for s in config["supplements"]:
            labs = " · ".join(f"[{Path(p).suffix[1:]}]({p})" for p in s.get("labs", []))
            parts.append(
                f"| {s['title']} | {labs} | [MD](course/handouts/{s['id']}-quiz.md) · [PDF](course/handouts/{s['id']}-quiz.pdf) |"
            )
    parts.extend(
        [
            "",
            overview,
            "## 교재와 참고자료",
            "",
            "[전체 참고자료와 차시별 출처](course/references.md)",
            "[기초 실습 안내](course/basics.md)",
        ]
    )
    write(
        root / "README.md",
        "\n\n".join(parts[: parts.index("## 수업 자료")])
        + "\n\n"
        + "\n".join(parts[parts.index("## 수업 자료") :])
        + "\n",
    )
    refs = ["# 전체 참고자료", "[강의 홈](../README.md)"]
    for s in config["sessions"]:
        refs.append("## " + s["title"])
        for p in s["pages"]:
            text = safe(root, p).read_text(encoding="utf-8")
            refs.append(f"[강의 원문](../{p})")
            section = re.search(r"^## 참고[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
            if section:
                refs.append(section[1].strip())
            refs.extend(sorted(set(re.findall(r"https?://[^\s<>`)\]]+", text))))
    write(root / "course/references.md", "\n\n".join(refs) + "\n")


def build(root=ROOT):
    (root / ".build/build.json").unlink(missing_ok=True)
    c = load(root)
    if (root / "notebooks/source").exists():
        from build_notebooks import build_all

        build_all(
            root / "notebooks/source", root / "notebooks/student", root / "instructor/notebooks"
        )
    errors = [x for x in validate(root / "course/notion", strict=True) if x.level == "ERROR"]
    if errors:
        raise ValueError("\n".join(f"{x.path}: {x.message}" for x in errors))
    documents = sorted((root / "course/notion").rglob("*.md"))
    names = [p.stem for p in documents]
    if len(set(names)) != len(names):
        raise ValueError("PDF filename collision")
    for s in c["sessions"] + c.get("supplements", []):
        for p in s.get("pages", []) + s.get("labs", []):
            if not safe(root, p).is_file():
                raise ValueError(f"Missing student file: {p}")
        if s.get("quiz"):
            documents.append(quiz(root, s))
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page()
        tab.route("https://**/*", lambda route: route.abort())
        tab.route("http://**/*", lambda route: route.abort())
        for p in documents:
            rendered = Path(tmp) / "page.html"
            write(rendered, render(p.read_text(encoding="utf-8"), p, root, c["repository"]))
            tab.goto(rendered.as_uri())
            tab.evaluate("document.fonts.ready")
            if not tab.evaluate(
                "Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)"
            ):
                raise ValueError(f"Broken PDF image: {p}")
            dest = root / "course/handouts" / f"{p.stem}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            tab.pdf(path=str(dest), prefer_css_page_size=True, print_background=True)
        browser.close()
    hub(root, c)
    errors = scan(root / "course")
    for folder in ("notebooks/student", "labs/student", "data/sample"):
        if (root / folder).exists():
            errors.extend(scan(root / folder))
    errors.extend(scan(root / "README.md"))
    if errors:
        raise ValueError("\n".join(f"{e.path}: {e.message}" for e in errors))
    rev = revision(root)
    dest = root / "dist/notion" / f"{c['id']}-{rev[:12]}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root / "course/notion").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root / "course/notion"))
    write(
        root / ".build/build.json",
        json.dumps(
            {
                "revision": rev,
                "zip": str(dest.relative_to(root)),
                "zip_sha256": digest(dest),
                "checks": "Notion strict, PDF images, quiz alignment, public scan; runtime separate",
            },
            indent=2,
        ),
    )
    print("Build passed. Notion ZIP:", dest)


def stage(root=ROOT):
    # The whole public set is reviewed together; no automatic commit/push.
    build(root)
    candidate = root / ".build" / ("release-" + revision(root)[:12])
    if candidate.exists():
        candidate = Path(tempfile.mkdtemp(prefix="release-", dir=root / ".build"))
    candidate.mkdir(parents=True, exist_ok=True)
    for folder in PUBLIC_ROOTS:
        if (root / folder).exists():
            shutil.copytree(
                root / folder,
                candidate / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )
    for rel in PUBLIC_FILES:
        if (root / rel).is_file():
            dst = candidate / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, dst)
    errors = scan(candidate)
    if errors:
        raise ValueError("\n".join(f"{e.path}: {e.message}" for e in errors))
    files = {
        p.relative_to(candidate).as_posix(): digest(p)
        for p in sorted(candidate.rglob("*"))
        if p.is_file()
    }
    write(
        root / ".build/stage.json",
        json.dumps({"path": str(candidate), "revision": revision(root), "files": files}, indent=2),
    )
    print("Scanned candidate:", candidate)
    print("\n".join(files))


def state_path(root, sid):
    if not re.fullmatch(r"w\d{2}[a-z]?", sid):
        raise ValueError("Invalid session ID")
    return root / "briefs" / sid / "workflow.json"


def status(root=ROOT, sid=None):
    c = load(root)
    rev = revision(root)
    result = []
    selected = [s for s in c["sessions"] if not sid or s["id"] == sid]
    if sid and not selected:
        raise ValueError("Register session in course.yml first")
    for s in selected:
        p = state_path(root, s["id"])
        state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        result.append(
            {
                "session": s["id"],
                "title": s["title"],
                "date": s.get("date"),
                "notes": state.get("notes", []),
                "checks": state.get("checks", {}),
                "current_revision": rev,
                "notion_url": s.get("notion_url"),
                "notion_confirmed_current": state.get("notion_revision") == rev,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "stage", "status", "note", "notion", "inputs"])
    parser.add_argument("session", nargs="?")
    parser.add_argument("--note")
    parser.add_argument("--url")
    parser.add_argument("--zip-sha256")
    a = parser.parse_args()
    if a.command == "build":
        build()
    elif a.command == "stage":
        stage()
    elif a.command == "status":
        status(sid=a.session)
    elif a.command == "inputs":
        data = yaml.safe_load(
            state_path(ROOT, a.session).with_name("source-map.yml").read_text(encoding="utf-8")
        )
        failures = []
        for entry in data.get("sources", []):
            p = safe(ROOT, entry["path"])
            if not p.is_file():
                failures.append(entry["path"])
            else:
                print(entry["path"], digest(p))
        failures.extend(data.get("pending", []))
        if failures:
            raise ValueError("Missing sources: " + ", ".join(failures))
    else:
        if not a.note:
            raise ValueError("Record an actual observation or user instruction with --note")
        c = load()
        s = next(s for s in c["sessions"] if s["id"] == a.session)
        p = state_path(ROOT, a.session)
        state = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if a.command == "notion":
            u = urlparse(a.url or "")
            if (
                u.scheme != "https"
                or u.username
                or u.password
                or not any(
                    (u.hostname or "") == d or (u.hostname or "").endswith("." + d)
                    for d in ["notion.site", "notion.com"]
                )
            ):
                raise ValueError("Expected Notion HTTPS URL")
            record = json.loads((ROOT / ".build/build.json").read_text(encoding="utf-8"))
            if (
                record["revision"] != revision(ROOT)
                or record["zip_sha256"] != a.zip_sha256
                or digest(safe(ROOT, record["zip"])) != a.zip_sha256
            ):
                raise ValueError("Build or uploaded ZIP changed")
            s["notion_url"] = a.url
            write(
                ROOT / "course/course.yml", yaml.safe_dump(c, allow_unicode=True, sort_keys=False)
            )
            state["notion_revision"] = revision(ROOT)
            state["zip_sha256"] = a.zip_sha256
            hub(ROOT, c)
        state.setdefault("notes", []).append(
            {
                "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "text": a.note,
                "revision": revision(ROOT),
            }
        )
        write(p, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
