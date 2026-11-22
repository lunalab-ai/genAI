"""Capture two review panels from trusted app state, with no submitted HTML."""
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from .event_design import ASSETS, UPDATE_EVENT, summarize


def capture_screens(manifest, events):
    from playwright.sync_api import Error, sync_playwright
    from .design_app import LABELS, STATES, table_html

    styles = (ASSETS / "design.html").read_text(encoding="utf-8-sig").split("<style>", 1)[1].split("</style>", 1)[0]
    state = manifest["app_state"][0]
    raw = manifest["app"][0]
    label = LABELS[manifest["choice"]]
    mode = "지금 모델 실행" if manifest["mode"] == "live" else "실측 기록 재생"
    facts = " · ".join(f"{k} {v}" for k, v in state["confirmed_facts"].items())
    ids = f"설정 {state['config_id']} · 문서 {state['document_version'][:12]}"

    def p(content):
        return "<p>" + escape(str(content)) + "</p>"

    creation = p(f"{events[UPDATE_EVENT]['name']} · {label} · {mode} · {state['status']}") + p(facts) + p(ids)
    creation += "<h3>모델 원문</h3><pre>" + escape(raw["raw"]) + "</pre>"
    stats = [{**row, "label": LABELS[row["condition"]]} for row in summarize(manifest["validation"], events)]
    fields = [{**row, "status": STATES[row["status"]]} for row in state["rows"]]
    verification = p(f"기본값 {label} · {mode} · 행사당 핵심 3필드") + p(ids)
    verification += table_html(stats, [
        ("label", "조건"), ("retrieved", "찾은 근거"), ("core_correct", "핵심 일치"),
        ("unsupported", "자료 밖 값"), ("valid_sources", "유효 출처"),
        ("input_tokens", "토큰"), ("adapter_params", "학습 스칼라"),
    ])
    verification += "<h3>생성값과 운영 자료 대조</h3>" + table_html(fields, [
        ("field", "항목"), ("generated", "생성값"), ("expected", "원문"), ("status", "판정"),
        ("source", "생성 출처"), ("expected_source", "운영 문서"),
    ])
    pages = {
        "creation.png": ("콘텐츠 만들기", creation),
        "verification.png": ("생성 결과 검증", verification),
    }
    with sync_playwright() as pw, TemporaryDirectory(prefix="w13a-capture-") as temp:
        try:
            browser = pw.chromium.launch(channel="msedge", headless=True)
        except Error:
            browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1170, "height": 740})
            captured = {}
            for name, (title, body) in pages.items():
                path = Path(temp) / (name + ".html")
                path.write_text(
                    '<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>' + styles +
                    '</style></head><body><header><h1>캠퍼스 팝업 콘텐츠 제작 스튜디오</h1></header>'
                    '<main><section><h2>' + escape(title) + "</h2>" + body + "</section></main></body></html>",
                    encoding="utf-8",
                )
                page.goto(path.as_uri())
                captured[name] = page.screenshot(full_page=True)
            return captured
        finally:
            browser.close()


def try_capture(manifest, events):
    try:
        return capture_screens(manifest, events), None
    except (ImportError, OSError, RuntimeError) as exc:
        return {}, str(exc).splitlines()[0]
    except Exception as exc:
        from playwright.sync_api import Error
        if not isinstance(exc, Error):
            raise
        return {}, str(exc).splitlines()[0]
