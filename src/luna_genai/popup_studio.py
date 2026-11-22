"""W13A studio. Generation, structured verification, and evidence export."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import zipfile
from pathlib import Path

ASSETS = Path(__file__).with_name("assets") / "w13a"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SETTINGS = {"do_sample": False, "max_new_tokens": 180}
FIELDS = {"when": "일시", "where": "장소", "fee": "참가비", "signup": "신청", "capacity": "정원"}
UNKNOWN = "확인 필요"
MODES = ("without", "with")
CHANNELS = {"board": "학교 게시판용 공지", "promo": "학생 홍보용 문구"}
DETAIL_COLUMNS = ("event", "condition", "field", "generated", "expected", "source", "expected_source", "status", "source_ok")
SUMMARY_COLUMNS = ("condition", "core_correct", "core_total", "missing", "unknown_handled", "unsupported", "supported")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def load_catalog():
    return json.loads((ASSETS / "catalog.json").read_text(encoding="utf-8"))


def expected_facts(event):
    """Map each structured field to the authoritative value and document ID."""
    return {
        field: (value, document["id"] if value is not None else "")
        for document in event["docs"]
        for field, value in document["facts"].items()
    }


def format_sources(documents):
    """Take document dictionaries and return the literal text for the prompt."""
    sections = []
    for document in documents:
        lines = ["[" + document["id"] + "] " + document["title"]]
        lines.extend(
            FIELDS[field] + ": " + value
            for field in FIELDS
            if (value := document.get("facts", {}).get(field)) is not None
        )
        if document.get("note"):
            lines.append(document["note"])
        if document.get("text"):
            lines.append(document["text"])
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_prompt(event, channel, context):
    """Use identical instructions for both conditions; vary only context."""
    request = {
        "board": "다음 행사의 학교 게시판 안내를 정확하고 정중하게 작성하세요.",
        "promo": "다음 행사의 학생 홍보 문구를 한국어로 짧고 친근하게 작성하세요.",
    }[channel]
    schema = "\n".join(label + "=" for label in FIELDS.values()) + "\n문장="
    return [
        {"role": "system", "content": "너는 캠퍼스 행사 안내문 작성자다. 자료에 근거해 정확히 한국어로 답한다. 자료에 없는 정보 또는 미정인 항목은 확인 필요라고 쓴다."},
        {"role": "user", "content": (
            request + "\n행사: " + event["name"] + "\n<자료>\n" + context + "\n</자료>\n"
            "여섯 줄로 출력하세요. 앞 다섯 줄은 항목=값|출처ID로 쓰고, 확인 필요 뒤에는 |없음으로 표시합니다. "
            "마지막 줄에는 안내 문장 하나를 적으세요.\n" + schema
        )},
    ]


class CpuGenerator:
    """One pinned model. Load only when the browser or notebook asks for live generation."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(min(2, torch.get_num_threads()))
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.float32
        ).eval()
        self.load_seconds = round(time.perf_counter() - started, 3)

    def __call__(self, messages):
        started = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **SETTINGS)
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        return {
            "raw": self.tokenizer.decode(new_ids, skip_special_tokens=True),
            "seconds": round(time.perf_counter() - started, 3),
            "tokens": int(new_ids.shape[0]),
        }


def read_fields(raw):
    """Read labelled fields from the model draft without repairing its claims."""
    found = {}
    labels = {value: key for key, value in FIELDS.items()}
    labels.update({"신청 방법": "signup", "인원 제한": "capacity", "문장": "draft", "안내문": "draft"})
    for line in raw.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        match = re.match(r"([^:=|：]{1,20})\s*[:=|：]\s*(.*)", line)
        if not match:
            continue
        field = labels.get(match[1].strip(" *"))
        if field is None or field in found:
            continue
        value, _, source = match[2].partition("|")
        source = source.strip(" *[]")
        found[field] = (value.strip(" *"), "" if source in {"없음", "-", "—"} else source)
    return found


def normalized(value, field):
    value = unicodedata.normalize("NFKC", value).strip()
    if field == "fee":
        if value == "무료":
            return "0"
        match = re.fullmatch(r"([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)\s*원?", value)
        if match:
            return match[1].replace(",", "").lstrip("0") or "0"
    return re.sub(r"\s+", "", value)


def verify_run(run, event):
    """Compare typed fields and supporting IDs to the trusted case and actual context."""
    found = read_fields(run["raw"])
    expected = expected_facts(event)
    rows = []
    for field in FIELDS:
        value, cited = found.get(field, ("", ""))
        expected_value, expected_id = expected[field]
        if not value:
            status = "missing"
        elif expected_value is None:
            status = "unknown_ok" if value == UNKNOWN else "unsupported"
        elif normalized(value, field) == normalized(expected_value, field):
            status = "match"
        else:
            status = "mismatch"
        cited_in_context = any("[" + ident + "]" in run["context"] for ident in (expected_id,) if ident)
        source_ok = (status == "match" and cited_in_context and cited == expected_id) or (status == "unknown_ok" and not cited)
        rows.append({
            "event": event["id"], "condition": run["condition"], "field": field,
            "generated": value, "expected": expected_value if expected_value is not None else UNKNOWN,
            "source": cited, "expected_source": expected_id, "status": status, "source_ok": source_ok,
        })
    return rows


class Studio:
    def __init__(self):
        self.catalog = load_catalog()
        self.events = {event["id"]: event for event in self.catalog["events"]}
        self.model = None

    def generate(self, event_id, use_sources, channel, execution, connector=None):
        """Run one prompt via the selected CPU or recorded-result route."""
        event = self.events[event_id]
        condition = "with" if use_sources else "without"
        # The prepared demo takes its context from a verified data snapshot.
        prepared = self.catalog.get("demo_contexts", {}).get(event_id, "")
        context = (prepared if use_sources else "") if connector is None else connector(event["docs"], use_sources)
        target = format_sources(event["docs"]) if use_sources else ""
        if context != target:
            raise ValueError("04 근거 연결 셀을 완료하고 다시 실행하세요. 근거 스위치와 문서 목록을 확인하세요.")
        messages = build_prompt(event, channel, context)
        prompt_hash = digest(messages)
        if execution == "live":
            if self.model is None:
                self.model = CpuGenerator()
            generated = self.model(messages)
        elif execution == "replay":
            replay = json.loads((ASSETS / "replay.json").read_text(encoding="utf-8"))
            generated = replay["outputs"].get(prompt_hash)
            if generated is None:
                raise ValueError("이 조합에는 사전 결과가 없습니다. 공식 요청을 선택하거나 실제 생성을 사용하세요.")
        else:
            raise ValueError("알 수 없는 실행 경로입니다.")
        return {
            "event": event_id, "condition": condition, "channel": channel, "context": context,
            "messages": messages, "prompt_sha256": prompt_hash, "model": MODEL_ID, "revision": REVISION,
            "settings": SETTINGS, "execution": execution, "connection": "student" if connector is not None else "demo",
            **generated,
        }


def detail_and_summary(runs, events):
    rows = [row for run in runs for row in verify_run(run, events[run["event"]])]
    summary = []
    for mode in MODES:
        group = [row for row in rows if row["condition"] == mode]
        core = [row for row in group if row["field"] in {"when", "where", "fee"}]
        summary.append({
            "condition": mode, "core_correct": sum(row["status"] == "match" for row in core),
            "core_total": len(core), "missing": sum(row["status"] == "missing" for row in group),
            "unknown_handled": sum(row["status"] == "unknown_ok" for row in group),
            "unsupported": sum(row["status"] == "unsupported" for row in group),
            "supported": sum(row["source_ok"] for row in group),
        })
    return rows, summary


def best_modes(summary):
    """Prefer safe missing-information behavior, then accurate facts and traceable IDs."""
    keys = {
        row["condition"]: (-row["unsupported"], row["unknown_handled"], row["core_correct"], row["supported"])
        for row in summary
    }
    best = max(keys.values())
    return [mode for mode in MODES if keys[mode] == best]


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def run_snapshot(studio, runs, default):
    rows, summary = detail_and_summary(runs, studio.events)
    screens = {"details": rows, "summary": summary, "active_default": default}
    return {
        "version": studio.catalog["version"], "model": MODEL_ID, "revision": REVISION,
        "catalog_sha256": digest(studio.catalog), "settings": SETTINGS, "default": default,
        "best": best_modes(summary), "execution": runs[0]["execution"] if runs else None,
        "screen_sha256": digest(screens), "screens": screens,
    }


def export_zip(studio, runs, default, screenshots=None):
    expected_pairs = {(event, mode) for event in studio.events for mode in MODES}
    observed_pairs = {(run["event"], run["condition"]) for run in runs}
    if observed_pairs != expected_pairs or len(runs) != len(expected_pairs):
        raise ValueError("07 절에서 두 행사 × 두 설정을 모두 실행하세요.")
    if default not in MODES:
        raise ValueError("제출 확인에서 앱 기본값을 적용하세요.")
    manifest = run_snapshot(studio, runs, default)
    if len({run["execution"] for run in runs}) != 1:
        raise ValueError("공식 비교는 같은 실행 경로의 네 결과를 사용하세요.")
    screenshots = screenshots or {}
    manifest["capture"] = "captured" if set(screenshots) == {"creation.png", "verification.png"} else "failed_or_unavailable"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("generations.json", json_bytes(runs))
        archive.writestr("validation.csv", csv_bytes(DETAIL_COLUMNS, manifest["screens"]["details"]))
        archive.writestr("summary.csv", csv_bytes(SUMMARY_COLUMNS, manifest["screens"]["summary"]))
        archive.writestr("run.json", json_bytes(manifest))
        for name, content in screenshots.items():
            archive.writestr("screenshots/" + name, content)
    return stream.getvalue()
