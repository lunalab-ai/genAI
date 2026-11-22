"""Shared W13A app controller for the local web page and the Colab widget."""
from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import RLock
import webbrowser

from .event_design import (
    ASSETS, AXES, FIELDS, UPDATE_EVENT, UPDATE_FEE, DesignStudio, choose_default, configs, load_json,
    preview, rank_documents, response_mask, summarize, to_zip,
    updated_event,
)

LABELS = {"A": "① Qwen2 기본", "B": "② 검색 추가", "C": "③ 검색 + LoRA", "D": "④ 개인 설정"}
STATES = {"match": "값 일치", "mismatch": "값 불일치", "missing": "출력 누락",
          "unknown_ok": "미정 처리", "unsupported": "자료 밖 값"}


def table_html(rows, cols):
    tr = lambda cells: "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in cells) + "</tr>"
    head = "<tr>" + "".join(f"<th>{html.escape(title)}</th>" for _, title in cols) + "</tr>"
    return "<table><thead>" + head + "</thead><tbody>" + "".join(
        tr(row.get(key, "") for key, _ in cols) for row in rows
    ) + "</tbody></table>"


def preview_view(run, event):
    panel = preview(run, event)
    facts = " · ".join(f"{label} {value}" for label, value in panel["confirmed_facts"].items())
    rows = [
        {**row, "field": FIELDS[row["field"]], "status": STATES[row["status"]],
         "source_ok": "유효" if row["source_ok"] else "확인", "evidence_seen": "있음" if row["evidence_seen"] else "없음"}
        for row in panel["rows"]
    ]
    return {
        **panel, "name": event["name"], "facts_text": facts,
        "condition": run["condition"], "label": LABELS[run["condition"]],
        "mode": "지금 모델 실행" if run["mode"] == "live" else "실측 기록 재생",
        "adapter_sha256": run["adapter_sha256"], "raw": run["raw"], "ranked": run["ranked"],
        "prompt": run["prompt"], "input_tokens": run["input_tokens"], "field_view": rows,
    }


class CourseApp:
    def __init__(self, studio=None, manifest=None):
        self.studio = studio or DesignStudio()
        self.manifest = manifest
        self.applied = False
        self.updated = False

    def evaluate(self, axis, hypothesis, *, mode="reference", selection=rank_documents,
                 chooser=choose_default, mask=response_mask):
        self.manifest = self.studio.make_run(axis, hypothesis, mask, mode=mode,
                                             selection=selection, chooser=chooser)
        self.applied = self.updated = False
        return self.summary()

    def summary(self):
        if self.manifest is None:
            raise ValueError("공통 실험부터 실행하세요.")
        table = summarize(self.manifest["validation"], self.studio.events)
        return {
            "choice": self.manifest["choice"], "rows": [{**row, "label": LABELS[row["condition"]]} for row in table],
            "official": summarize(self.manifest["official"], self.studio.events),
            "runs": [
                {"id": run["run_id"], "case": self.studio.events[run["event"]]["name"],
                 "condition": LABELS[run["condition"]], "raw": run["raw"], "prompt": run["prompt"],
                 "facts": preview(run, self.studio.events[run["event"]])["confirmed_facts"]}
                for run in self.manifest["official"] + self.manifest["validation"]
            ],
        }

    def film_channels(self):
        film = self.studio.events["film"]
        expected = preview(self.studio.run(film, "B"), film)["confirmed_facts"]
        when, where, fee, signup = (expected[FIELDS[key]] for key in ("when", "where", "fee", "signup"))
        facts = f"{when} · {where} · 참가비 {fee} · 신청 {signup}"
        captions = {
            "board": "독립영화 상영회 일정을 안내합니다.",
            "promo": "캠퍼스에서 독립영화 한 편 함께 봐요!",
        }
        path = ASSETS / "channels.json"
        records = load_json(path) if path.is_file() else {}
        return [{"label": label, "intro": captions[key], "facts": facts,
                 "raw": records.get(key, {}).get("raw"), "mode": "실측 기록 재생"}
                for key, label in (("board", "학교 게시판"), ("promo", "학생 홍보"))]

    def apply(self):
        if self.manifest is None:
            raise ValueError("검증 실험부터 실행하세요.")
        self.applied = True
        original = self.studio.events[UPDATE_EVENT]
        return preview_view(self.manifest["app"][0], original)

    def replace_document(self):
        if not self.applied:
            raise ValueError("기본값을 먼저 적용하세요.")
        self.updated = True
        revised = updated_event(self.studio.events)
        return preview_view(self.manifest["app"][1], revised)

    def zip_bytes(self, observations, *, capture=True):
        if not self.updated:
            raise ValueError("앱에서 원본과 갱신 자료를 모두 확인하세요.")
        if len(observations) != 2:
            raise ValueError("값·출처 오류 두 건을 기록하세요.")
        self.manifest["observations"] = observations
        from .design_capture import try_capture

        screenshots, self.capture_error = try_capture(self.manifest, self.studio.events) if capture else ({}, None)
        return to_zip(self.manifest, self.studio.events, screenshots)


def show_widget(app):
    from IPython.display import display
    import ipywidgets as widgets

    stage = widgets.HTML("<b>검증표에서 선택한 설정 → 원본 → 갱신 자료</b>")
    text = widgets.HTML()
    detail = widgets.HTML()
    apply = widgets.Button(description="앱 기본값 적용", button_style="success")
    change = widgets.Button(description="참가비 2,000원 공지 반영", disabled=True)

    def show(panel):
        text.value = (
            f"<h3>{html.escape(panel['name'])} · {html.escape(panel['status'])}</h3>"
            f"<p>{html.escape(panel['label'])} · {html.escape(panel['mode'])}</p>"
            f"<p>{html.escape(panel['facts_text'])}</p>"
            f"<p>설정 ID <code>{panel['config_id']}</code> · 문서 버전 <code>{panel['document_version'][:12]}</code></p>"
        )
        detail.value = table_html(panel["field_view"], [
            ("field", "필드"), ("generated", "생성값"), ("expected", "원문"),
            ("status", "판정"), ("source_ok", "출처"),
        ]) + "<details><summary>생성 모델이 받은 프롬프트와 원문</summary><pre>" + html.escape(
            panel["prompt"] + "\n\n생성 원문\n" + panel["raw"]
        ) + "</pre></details>"

    def apply_cb(_):
        show(app.apply())
        change.disabled = False

    def update_cb(_):
        show(app.replace_document())

    apply.on_click(apply_cb)
    change.on_click(update_cb)
    display(widgets.VBox([stage, widgets.HBox([apply, change]), text, detail]))
    return apply, change, text


def serve(port=8767, *, open_browser=True):
    app = CourseApp()
    lock = RLock()
    page = (ASSETS / "design.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, value, mime, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(value)))
            self.send_header("X-Content-Type-Options", "nosniff")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(value)

        def reply(self, body, status=200):
            self.send(json.dumps(body, ensure_ascii=False).encode("utf-8"),
                      "application/json; charset=utf-8", status=status)

        def do_GET(self):
            if self.path == "/":
                self.send(page, "text/html; charset=utf-8")
            elif self.path == "/api/start":
                event = app.studio.events["film"]
                initial = app.studio.run(event, "A")
                self.reply({**preview_view(initial, event), "channels": app.film_channels()})
            elif self.path == "/photos.png":
                self.send((ASSETS.parent / "w13a/studio-photos.png").read_bytes(), "image/png")
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20000:
                    raise ValueError("요청 형식을 확인하세요.")
                args = json.loads(self.rfile.read(size))
                with lock:
                    if self.path == "/api/experiment":
                        if args["axis"] not in AXES or not args["hypothesis"].strip():
                            raise ValueError("비교 변수와 가설을 먼저 기록하세요.")
                        self.reply(app.evaluate(args["axis"], args["hypothesis"]))
                    elif self.path == "/api/apply":
                        self.reply(app.apply())
                    elif self.path == "/api/update":
                        self.reply(app.replace_document())
                    elif self.path == "/api/first-live":
                        event = app.studio.events["film"]
                        step = app.studio.run(event, "A", mode="live")
                        self.reply(preview_view(step, event))
                    elif self.path == "/api/live":
                        if not app.applied:
                            raise ValueError("기본값을 먼저 적용하세요.")
                        axis = app.manifest["axis"]
                        choice = app.manifest["choice"]
                        selected = configs(axis)[choice]["rank"]
                        if selected and (selected, 24) not in app.studio.runtime.adapter_paths:
                            app.studio.runtime.register_adapter(
                                selected, 24, ASSETS / "adapters" / f"rank{selected}-steps24",
                                app.studio.reference["training"][f"{selected}:24"],
                            )
                        event = updated_event(app.studio.events) if app.updated else app.studio.events[UPDATE_EVENT]
                        run = app.studio.run(event, choice, axis, phase="app", steps=24, mode="live")
                        self.reply(preview_view(run, event))
                    elif self.path == "/api/zip":
                        self.send(app.zip_bytes(args["observations"]), "application/zip",
                                  filename="w13a-design-submission.zip")
                    else:
                        self.send_error(404)
            except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError) as exc:
                self.reply({"error": str(exc)}, status=400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = f"http://127.0.0.1:{server.server_port}/"
    print(address, flush=True)
    if open_browser:
        webbrowser.open(address)
    with server:
        server.serve_forever()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--port", type=int, default=8767)
    cli.add_argument("--no-open", action="store_true")
    args = cli.parse_args()
    serve(args.port, open_browser=not args.no_open)
