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
    "requirements-lab.txt",
    "scripts/execute_notebook.py",
    ".github/workflows/materials.yml",
    "scripts/vendor/tex-svg-full.js",
    "scripts/vendor/LICENSE.mathjax",
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
        if not re.fullmatch(r"w\d{1,2}[a-z]?", s["id"]):
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
    from html import escape
    config = load(root)
    formulas = {}
    if config.get("math"):
        def capture(match):
            value = match.group(0)
            key = "MATHPLACEHOLDER" + str(len(formulas)) + "END"
            display = value.startswith("$$") or value.startswith(r"\[")
            formula = value[2:-2] if display or value.startswith(r"\(") else value[1:-1]
            formulas[key] = (r"\[" if display else r"\(") + escape(formula) + (r"\]" if display else r"\)")
            return key
        chunks = re.split(r"(```[\s\S]*?```|`[^`\n]*`)", text)
        pattern = r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)|(?<!\\)\$[^$\n]+?(?<!\\)\$"
        text = "".join(re.sub(pattern, capture, chunk) if i % 2 == 0 else chunk for i, chunk in enumerate(chunks))
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
    body = md.renderer.render(tokens, md.options, {})
    for key, formula in formulas.items(): body = body.replace(key, formula)
    math_script = ""
    if formulas:
        asset = root / "scripts/vendor/tex-svg-full.js"
        math_script = '<script>MathJax={startup:{typeset:true},svg:{fontCache:"local"}};</script><script>' + asset.read_text(encoding="utf-8") + '</script>'
    return (
        '<!doctype html><html lang="' + escape(config.get("language", "ko")) + '"><meta charset="utf-8"><style>'
        + CSS
        + "</style><body>"
        + body + math_script
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
    parts = ["# " + session["title"] + (" · Quiz explanations" if load(root).get("language") == "en" else " · 퀴즈 해설"), data["provenance"]]
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
        if s.get("browser_url"):
            run = f"[브라우저 실습]({s['browser_url']}) · " + run
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
    if config.get("supplements") and config.get("show_supplements", True):
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
    if config.get("language") == "en":
        translations = {"## 수업 자료": "## Class materials", "| 차시 | 날짜 | 주제 | 강의노트 | 퀴즈 해설 | 실습 | 실행 |": "| Session | Date | Topic | Lecture notes | Quiz explanations | Practice | Runtime |", "## 보충 실습 자료": "## Supplementary practice", "| 자료 | 실습 | 퀴즈 해설 |": "| Material | Practice | Quiz explanations |", "## 교재와 참고자료": "## Textbooks and references", "전체 참고자료와 차시별 출처": "References and session sources", "기초 실습 안내": "Prerequisites and practice preparation", "일정 미확정": "Date to be confirmed"}
        text = (root / "README.md").read_text(encoding="utf-8")
        for original, translated in translations.items(): text = text.replace(original, translated)
        write(root / "README.md", text)
    refs = ["# References", "[Course home](../README.md)"] if config.get("language") == "en" else ["# 전체 참고자료", "[강의 홈](../README.md)"]
    for s in config["sessions"]:
        refs.append("## " + s["title"])
        for p in s["pages"]:
            text = safe(root, p).read_text(encoding="utf-8")
            refs.append(f"[Lecture source](../{p})" if config.get("language") == "en" else f"[강의 원문](../{p})")
            section = re.search(r"^## 참고[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
            if section:
                refs.append(section[1].strip())
            refs.extend(sorted(set(re.findall(r"https?://[^\s<>`)\]]+", text))))
    write(root / "course/references.md", "\n\n".join(refs) + "\n")


def validate_course_contract(root, c):
    policy = c.get("lesson_policy", {})
    for session in c.get("sessions", []):
        if session["id"] in policy.get("legacy_sessions", []):
            continue
        if policy.get("automatic_notebooks"):
            if not any(p.endswith(".ipynb") for p in session.get("labs", [])) or not session.get("colab_url"):
                raise ValueError("Notebook and Colab URL required: " + session["id"])
        if policy.get("ask_assignment"):
            assignment = session.get("assignment", {})
            if type(assignment.get("requested")) is not bool:
                raise ValueError("Record professor assignment decision: " + session["id"])
            if assignment["requested"]:
                rubric = assignment.get("rubric", [])
                if not rubric or any(not isinstance(x.get("points"), (int, float)) or x["points"] <= 0 for x in rubric) or sum(x["points"] for x in rubric) != 10:
                    raise ValueError("Assignment rubric must total 10: " + session["id"])
                rel = assignment.get("page", "")
                if not rel.startswith("course/notion/") or not safe(root, rel).is_file():
                    raise ValueError("Missing student assignment: " + session["id"])
    if c.get("required_instructor_scripts"):
        for session in c["sessions"]:
            scripts = session.get("instructor_scripts", {})
            for language in c.get("instructor_languages", ["en", "ko"]):
                rel = scripts.get(language, "")
                if not rel.startswith("instructor/"): raise ValueError("Missing private speaker script: " + session["id"] + "/" + language)
                path = safe(root, rel)
                if not path.is_file() or not path.read_text(encoding="utf-8").strip(): raise ValueError("Empty/missing speaker script: " + rel)
            if len(set(scripts.values())) != len(scripts): raise ValueError("Use distinct files for bilingual scripts")
    if c.get("language") == "en":
        for folder in ["course/notion", "course/handouts", "notebooks/student", "labs/student"]:
            for p in (root / folder).rglob("*"):
                if p.is_file() and p.suffix in {".md", ".qmd", ".py", ".R", ".ipynb"} and re.search(r"[가-힣]", p.read_text(encoding="utf-8")):
                    raise ValueError("English student material contains Korean text: " + str(p.relative_to(root)))


def build(root=ROOT):
    (root / ".build/build.json").unlink(missing_ok=True)
    c = load(root)
    validate_course_contract(root, c)
    if c.get("browser", {}).get("enabled"):
        from browser_labs import render as render_browser, check as check_browser
        render_browser(root)
        check_browser(root)
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
            if c.get("math"):
                tab.evaluate("async () => {if(window.MathJax && MathJax.startup) await MathJax.startup.promise}")
                if tab.locator('[data-mjx-error], mjx-merror').count():
                    raise ValueError(f"Math rendering error: {p}")
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
    notices(root)


def notices(root=ROOT, sid=None):
    """Save copyable student announcements privately and print their full text."""
    from urllib.parse import quote
    c = load(root)
    sessions = [s for s in c["sessions"] if sid is None or s["id"] == sid]
    if sid and not sessions:
        raise ValueError("Unknown session: " + sid)
    for s in sessions:
        if not s.get("pages"):
            continue
        base = "https://github.com/" + c["repository"]
        links = []
        if s.get("notion_url"):
            links.append("Notion 강의노트\n" + s["notion_url"])
        else:
            links.append("Notion 강의노트: 링크 등록 후 안내합니다. 우선 아래 Markdown/PDF를 이용하세요.")
        def add(label, relative):
            # Never announce a missing file or an instructor-only path.
            allowed = ("course/notion/", "course/handouts/", "notebooks/student/", "labs/student/")
            if not relative.startswith(allowed):
                raise ValueError("Not a student announcement path: " + relative)
            if safe(root, relative).is_file():
                links.append(label + "\n" + base + "/blob/main/" + quote(relative, safe="/"))
        for page in s["pages"]:
            add("Markdown 강의노트", page)
            add("PDF 강의노트", "course/handouts/" + Path(page).stem + ".pdf")
        if s.get("quiz"):
            add("퀴즈 답안·해설", "course/handouts/" + s["id"] + "-quiz.md")
            add("퀴즈 답안·해설 PDF", "course/handouts/" + s["id"] + "-quiz.pdf")
        for lab in s.get("labs", []):
            add("실습 파일 (" + Path(lab).suffix.lstrip(".") + ")", lab)
        if s.get("colab_url"):
            links.append("Colab 실습\n" + s["colab_url"])
        if s.get("browser_url"):
            links.append("설치 없는 온라인 R 실습\n" + s["browser_url"])
        guidance = s.get("announcement_guidance", [])
        if s.get("colab_url"):
            guidance = ["Colab에서 Drive에 사본을 저장한 뒤 설명을 읽고 한 셀씩 실행하세요."] + guidance
        elif s.get("labs") and c.get("runtime") == "r" and not s.get("browser_url"):
            guidance = ["R과 RStudio를 준비하고 .R 실습 파일을 열어 한 표현식씩 실행하세요."] + guidance
        lines = [f"[{c['title']}] {s.get('date') or '일정 추후 안내'} · {s['id']} {s['title']} 자료 안내",
                 "강의자료와 실습 안내입니다. 아래 링크를 이용해 주세요.",
                 "\n\n".join(links),
                 ("수강생 여러분의 요청을 반영해 강의노트를 PDF로도 제공하고, 강의 끝 퀴즈의 답안·해설을 별도로 제공합니다. 퀴즈를 먼저 풀어 본 뒤 해설로 확인해 주세요."
                  if any(x.startswith("PDF 강의노트") for x in links) and any(x.startswith("퀴즈 답안·해설") for x in links)
                  else "위에 연결된 자료를 이용해 학습하고, 퀴즈 해설이 제공되면 먼저 풀어 본 뒤 확인해 주세요."),
                 "\n".join("- " + item for item in guidance),
                 "전체 강의자료\n" + base]
        body = "\n\n".join(line for line in lines if line) + "\n"
        if c.get("language") == "en":
            translations = {
                "자료 안내": "Materials",
                "일정 추후 안내": "Date to be announced",
                "위에 연결된 자료를 이용해 학습하고, 퀴즈 해설이 제공되면 먼저 풀어 본 뒤 확인해 주세요.": "Study the linked materials. Attempt checkpoint questions before reading any available explanations.",
                "강의자료와 실습 안내입니다. 아래 링크를 이용해 주세요.": "Please use the following lecture and practice materials.",
                "Notion 강의노트: 링크 등록 후 안내합니다. 우선 아래 Markdown/PDF를 이용하세요.": "Notion: the link will follow. Use the Markdown/PDF below in the meantime.",
                "Notion 강의노트": "Notion lecture notes",
                "Markdown 강의노트": "Markdown lecture notes",
                "PDF 강의노트": "PDF lecture notes",
                "퀴즈 답안·해설": "Quiz answers and explanations",
                "전체 강의자료": "All course materials",
                "Colab 실습": "Colab practice",
                "실습 파일": "Practice file",
                "Colab에서 Drive에 사본을 저장한 뒤 설명을 읽고 한 셀씩 실행하세요.": "Save a copy in Drive, read the explanations, and run the Colab cells in order.",
                "수강생 여러분의 요청을 반영해 강의노트를 PDF로도 제공하고, 강의 끝 퀴즈의 답안·해설을 별도로 제공합니다. 퀴즈를 먼저 풀어 본 뒤 해설로 확인해 주세요.": "Lecture notes are also available as PDFs, with separate checkpoint quiz answers and explanations as requested. Attempt the questions before consulting the explanations.",
            }
            for old, new in sorted(translations.items(), key=lambda item: -len(item[0])):
                body = body.replace(old, new)
        destination = root / "instructor/announcements" / (s["id"] + "-smartclass.txt")
        write(destination, body)
        assignment = s.get("assignment", {})
        if assignment.get("requested"):
            page = safe(root, assignment["page"])
            assignment_body = page.read_text(encoding="utf-8")
            if assignment.get("notice_summary"):
                title = assignment_body.splitlines()[0].lstrip("# ")
                assignment_body = (f"[{c['title']}] {title}\n\n"
                    + "Total: 10 points\nDeadline: " + assignment.get("deadline", "As displayed in SmartClass") + "\n\n"
                    + assignment["notice_summary"] + "\n\nRubric (10 points)\n"
                    + "\n".join(f"- {item['criterion']}: {item['points']} points" for item in assignment["rubric"])
                    + "\n\n" + assignment.get("submission_instructions", "Follow the full specification and submit through SmartClass.") + "\n")
            assignment_body += "\n\nAssignment page\n" + base + "/blob/main/" + assignment["page"] + "\n"
            pdf = "course/handouts/" + page.stem + ".pdf"
            if safe(root, pdf).is_file():
                assignment_body += "\nAssignment PDF\n" + base + "/blob/main/" + pdf + "\n"
            if assignment.get("starter"):
                assignment_body += "\nProject starter in Colab\nhttps://colab.research.google.com/github/" + c["repository"] + "/blob/main/" + assignment["starter"] + "\n"
            if s.get("colab_url"):
                assignment_body += "\nStarting Colab notebook\n" + s["colab_url"] + "\n"
            write(root / "instructor/announcements/assignments" / (s["id"] + "-assignment-smartclass.md"), assignment_body)
        print("\n=== SmartClass 공지 초안: 배포 링크 확인 후 복사 ===")
        print(body)
        print("저장:", destination)
        escaped = str(destination.resolve()).replace("'", "''")
        print(f"Get-Content -LiteralPath '{escaped}' -Raw -Encoding utf8 | Set-Clipboard")


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
            source = root / rel
            if rel == ".github/workflows/materials.yml" and (root / "scripts/public-materials.yml").exists():
                source = root / "scripts/public-materials.yml"
            shutil.copy2(source, dst)
    if yaml.safe_load((root / "course/course.yml").read_text(encoding="utf-8")).get("browser", {}).get("enabled"):
        shutil.copytree(root / "browser", candidate / "browser", ignore=shutil.ignore_patterns("_site", ".quarto"))
        for rel in ["scripts/browser_labs.py", "scripts/browser-pages.yml", "scripts/check_browser.py"]:
            source = root / rel
            if source.exists():
                dest = candidate / (".github/workflows/pages.yml" if rel.endswith("browser-pages.yml") else rel)
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
    public_config_path = candidate / "course/course.yml"
    if public_config_path.exists():
        public_config = yaml.safe_load(public_config_path.read_text(encoding="utf-8"))
        public_config.pop("required_instructor_scripts", None)
        for session in public_config.get("sessions", []): session.pop("instructor_scripts", None)
        write(public_config_path, yaml.safe_dump(public_config, allow_unicode=True, sort_keys=False))
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
    if not re.fullmatch(r"w\d{1,2}[a-z]?", sid):
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
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "stage", "status", "note", "notion", "inputs", "notice"])
    parser.add_argument("session", nargs="?")
    parser.add_argument("--note")
    parser.add_argument("--url")
    parser.add_argument("--zip-sha256")
    a = parser.parse_args()
    if a.command == "build":
        build()
    elif a.command == "stage":
        stage()
    elif a.command == "notice":
        notices(sid=a.session)
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
            notices(sid=a.session)
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
