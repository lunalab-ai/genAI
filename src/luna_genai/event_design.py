"""W13A controlled model design, evidence checking and ZIP contract."""
from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .model import MODEL_ID, MODEL_REVISION
from .popup_studio import FIELDS, UNKNOWN, digest, expected_facts, format_sources, verify_run

ASSETS = Path(__file__).parent / "assets" / "w13a_design"
EMBED_ID = "intfloat/multilingual-e5-small"
EMBED_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
VERSION = "w13a-design-v2"
SETTINGS = {"do_sample": False, "max_new_tokens": 180}
CORE = ("when", "where", "fee")
SPLITS = {"train": 24, "validation": 6, "official": 2}
AXES = ("top_k", "lora_rank")
CHOICES = ("B", "C", "D")
UPDATE_EVENT = "val-book-talk"
UPDATE_FEE = "2,000원"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_events():
    events = load_json(ASSETS / "events.json")["events"]
    counts = {split: sum(e["split"] == split for e in events) for split in SPLITS}
    if len({e["id"] for e in events}) != len(events) or counts != SPLITS or len(events) != sum(SPLITS.values()):
        raise ValueError("행사 ID 또는 데이터 분할을 확인하세요.")
    for event in events:
        facts = expected_facts(event)
        if set(facts) != set(FIELDS) or any(facts[field][0] is None for field in CORE):
            raise ValueError("핵심 사실과 미정 필드가 필요합니다.")
        if len(event["docs"]) != 3 or len({doc["id"] for doc in event["docs"]}) != 3:
            raise ValueError("행사별 운영 메모·FAQ·비교용 참고 문서가 필요합니다.")
    return events


def configs(axis):
    if axis not in AXES:
        raise ValueError("top_k 또는 lora_rank 중 하나를 선택하세요.")
    return {
        "A": {"k": 0, "rank": 0},
        "B": {"k": 2, "rank": 0},
        "C": {"k": 2, "rank": 4},
        "D": {"k": 1 if axis == "top_k" else 2, "rank": 4 if axis == "top_k" else 2},
    }


def training_spec(rank, steps):
    if rank not in (2, 4) or steps not in (4, 24):
        raise ValueError("rank는 2/4, 업데이트는 4/24만 지원합니다.")
    return {
        "rank": rank, "alpha": 2 * rank, "target_modules": ["q_proj", "v_proj"],
        "dropout": 0, "learning_rate": 0.0005, "batch_size": 1, "seed": 42,
        "steps": steps, "dtype": "float32", "max_length": 416, "weight_decay": 0,
    }


def prompt(event, docs):
    context = format_sources(docs)
    instructions = (
        "지시: 아래 자료로 한국어 행사 안내를 작성한다. 일시, 장소, 참가비, 신청, 정원을 각 한 줄씩 적고 출처 ID를 붙인다. "
        "근거가 없으면 확인 필요와 없음으로 표시한다.\n"
        "형식: 항목=값|문서ID. 마지막에는 문장=한국어 안내 한 문장.\n"
    )
    return f"{instructions}행사: {event['name']}\n자료:\n{context}\n답변:\n"


def channel_prompt(event, docs, channel):
    tone = {
        "board": "학교 게시판 공지용으로 정중한 문장으로 쓴다.",
        "promo": "학생 대상 홍보용으로 짧고 친근한 문장으로 쓴다.",
    }[channel]
    return prompt(event, docs).replace("형식:", tone + "\n형식:", 1)


def target(event):
    facts = expected_facts(event)
    rows = [
        f"{label}={facts[field][0] or UNKNOWN}|{facts[field][1] if facts[field][0] else '없음'}"
        for field, label in FIELDS.items()
    ]
    when, where, fee = (facts[field][0] for field in CORE)
    rows.append(f"문장={event['name']}은 {when}에 {where}에서 열립니다. 참가비는 {fee}입니다.")
    return "\n".join(rows) + "\n"


def training_documents(event):
    return [doc for doc in event["docs"] if doc["facts"]]


def response_mask(prefix, response):
    """Ignore the prompt in the next-token loss; keep all response tokens."""
    return [-100] * len(prefix) + list(response)


def mask_witness(tokens, mask=response_mask):
    prefix, response = tokens["prefix"], tokens["response"]
    labels = mask(prefix, response)
    if labels != response_mask(prefix, response):
        raise ValueError("프롬프트 위치와 정답 응답 위치의 마스크가 일치하지 않습니다.")
    return {
        "event": tokens["event"], "input_sha256": digest(prefix + response),
        "label_sha256": digest(labels), "prompt_tokens": len(prefix),
        "response_tokens": len(response),
    }


def rank_documents(scores, k):
    """Order cosine scores descending and use document ID to resolve ties."""
    return [row["id"] for row in sorted(scores, key=lambda item: (-item["score"], item["id"]))[:k]]


def updated_event(events):
    event = deepcopy(events[UPDATE_EVENT])
    doc = next(source for source in event["docs"] if "fee" in source["facts"])
    doc["facts"]["fee"] = UPDATE_FEE
    event["variant"] = "updated"
    return event


def document_version(event):
    return digest(event["docs"])


def evidence_for(event, ids):
    by_id = {doc["id"]: doc for doc in event["docs"]}
    return [by_id[identifier] for identifier in ids]


def reference_key(event, k, rank, steps):
    return f"{event['id']}:{event.get('variant', 'original')}:{k}:{rank}:{steps if rank else 0}"


def field_rows(run, event):
    rows = verify_run({**run, "condition": run["condition"]}, event)
    for row in rows:
        row["run_id"] = run["run_id"]
        row["phase"] = run["phase"]
        row["variant"] = event.get("variant", "original")
        row["evidence_seen"] = row["expected_source"] in run["used_docs"]
    return rows


def preview(run, event):
    rows = field_rows(run, event)
    safe = all(row["source_ok"] for row in rows)
    facts = expected_facts(event)
    return {
        "event": event["id"], "variant": event.get("variant", "original"),
        "document_version": document_version(event), "config_id": run["config_id"],
        "run_id": run["run_id"], "prompt_sha256": run["prompt_sha256"],
        "used_docs": run["used_docs"], "status": "필드 검사 통과" if safe else "검토 필요",
        "confirmed_facts": {FIELDS[key]: facts[key][0] or UNKNOWN for key in FIELDS},
        "rows": rows,
    }


def summarize(records, events):
    tables = []
    for condition in ("A", "B", "C", "D"):
        batch = [run for run in records if run["condition"] == condition]
        if not batch:
            continue
        rows = [row for run in batch for row in field_rows(run, events[run["event"]])]
        core = [row for row in rows if row["field"] in CORE]
        supports = [
            (set(run["used_docs"]), {key for val, key in expected_facts(events[run["event"]]).values() if val is not None})
            for run in batch
        ]
        tables.append({
            "condition": condition, "case_count": len(batch),
            "retrieved": sum(len(actual & required) for actual, required in supports),
            "retrieval_total": sum(len(required) for _, required in supports),
            "core_correct": sum(row["status"] == "match" for row in core), "core_total": len(core),
            "unsupported": sum(row["status"] == "unsupported" for row in rows),
            "valid_sources": sum(row["status"] == "match" and row["source_ok"] for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "input_tokens": sum(run["input_tokens"] for run in batch),
            "adapter_params": 67584 * batch[0]["rank"],
        })
    return tables


def choose_default(rows):
    return min(
        (row for row in rows if row["condition"] in CHOICES),
        key=lambda row: (
            row["unsupported"], -row["core_correct"], -row["valid_sources"],
            row["input_tokens"], row["adapter_params"], row["condition"],
        ),
    )["condition"]


def record_id(run):
    return digest({name: value for name, value in run.items() if name != "run_id"})


def config_id(k, rank, adapter):
    setting = {"model": MODEL_ID, "revision": MODEL_REVISION, "k": k, "rank": rank,
               "adapter": adapter, "settings": SETTINGS}
    if k:
        setting.update(retriever=EMBED_ID, embedding_revision=EMBED_REVISION)
    return digest(setting)[:16]


class DesignStudio:
    def __init__(self, *, bank=None, reference=None):
        self.events = {e["id"]: e for e in (load_events() if bank is None else bank)}
        self.reference = load_json(ASSETS / "reference.json") if reference is None else reference
        self._runtime = None
        self._embedder = None

    @property
    def runtime(self):
        if self._runtime is None:
            from .design_models import QwenDesign
            self._runtime = QwenDesign()
        return self._runtime

    @property
    def embedder(self):
        if self._embedder is None:
            from .design_models import EvidenceEncoder
            self._embedder = EvidenceEncoder()
        return self._embedder

    def ranking(self, event, live=False):
        if live:
            return self.embedder.ranking(event)
        version = document_version(event)
        return self.reference["rankings"][version]

    def make_inputs(self, event, k, selection=rank_documents, *, live_search=False):
        scores = self.ranking(event, live=live_search) if k else []
        selected = selection(scores, k) if k else []
        if selected != rank_documents(scores, k):
            raise ValueError("선택한 문서 ID의 순서가 코사인 점수와 맞지 않습니다.")
        docs = evidence_for(event, selected)
        return {"ranked": scores, "used_docs": selected, "context": format_sources(docs),
                "prompt": prompt(event, docs), "document_version": document_version(event)}

    def run(self, event, condition, axis="top_k", *, phase="class", steps=4,
            mode="reference", selection=rank_documents):
        config = configs(axis)[condition]
        k, rank = config["k"], config["rank"]
        if mode not in ("live", "reference"):
            raise ValueError("실행 경로는 live 또는 reference입니다.")
        reads = self.make_inputs(event, k, selection, live_search=mode == "live")
        if mode == "live":
            generated = self.runtime.generate(reads["prompt"], rank, steps)
        else:
            key = reference_key(event, k, rank, steps)
            generated = self.reference["generations"][key]
            if generated["prompt_sha256"] != digest(reads["prompt"]):
                raise ValueError("기록의 프롬프트와 현재 입력이 다릅니다.")
        entry = {
            "event": event["id"], "variant": event.get("variant", "original"), "phase": phase,
            "condition": condition, **config, "training_steps": steps if rank else 0,
            "mode": mode, "model": MODEL_ID, "revision": MODEL_REVISION,
            "retriever": EMBED_ID if k else None, "embedding_revision": EMBED_REVISION if k else None,
            "settings": SETTINGS, **reads, **generated,
        }
        entry["config_id"] = config_id(k, rank, entry["adapter_sha256"])
        entry["run_id"] = record_id(entry)
        return entry

    def experiment(self, axis, *, mode="reference", selection=rank_documents):
        common = [self.run(event, condition, axis, phase="official", steps=24,
                           mode=mode, selection=selection)
                  for event in self.events.values() if event["split"] == "official"
                  for condition in ("A", "B", "C")]
        valid = [self.run(event, condition, axis, phase="validation", steps=24,
                          mode=mode, selection=selection)
                 for event in self.events.values() if event["split"] == "validation"
                 for condition in CHOICES]
        return common, valid

    def make_run(self, axis, hypothesis, mask, *, mode="reference", selection=rank_documents,
                 chooser=choose_default):
        common, valid = self.experiment(axis, mode=mode, selection=selection)
        table = summarize(valid, self.events)
        chosen = chooser(table)
        event = self.events[UPDATE_EVENT]
        revised = updated_event(self.events)
        app_runs = [self.run(case, chosen, axis, phase="app", steps=24, mode=mode, selection=selection)
                    for case in (event, revised)]
        ranks = (4, 2) if axis == "lora_rank" else (4,)
        training = {str(rank): (self.runtime.training[(rank, 24)] if mode == "live"
                                else self.reference["training"][f"{rank}:24"]) for rank in ranks}
        return {
            "version": VERSION, "axis": axis, "hypothesis": hypothesis, "mode": mode,
            "data_sha256": digest(list(self.events.values())),
            "split": {split: [event["id"] for event in self.events.values() if event["split"] == split]
                      for split in SPLITS},
            "mask": mask_witness(self.reference["token_example"], mask),
            "training": training, "choice": chosen,
            "observations": [], "official": common, "validation": valid,
            "app": app_runs, "app_state": [preview(x, e) for x, e in zip(app_runs, (event, revised))],
        }


def csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def bundle_files(manifest, events):
    records = manifest["official"] + manifest["validation"] + manifest["app"]
    revised = updated_event(events)
    details = [
        row
        for run in records
        for row in field_rows(run, revised if run["variant"] == "updated" else events[run["event"]])
    ]
    reports = [{**row, "phase": phase}
               for phase in ("official", "validation")
               for row in summarize(manifest[phase], events)]
    run_meta = {key: value for key, value in manifest.items() if key not in ("official", "validation", "app")}
    return {
        "run.json": json.dumps(run_meta, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
        "generations.json": json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
        "validation.csv": csv_bytes(details),
        "summary.csv": csv_bytes(reports),
    }


def to_zip(manifest, events, screenshots=None):
    screenshots = screenshots or {}
    manifest["capture"] = "captured" if screenshots else "failed_or_unavailable"
    buf = io.BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in bundle_files(manifest, events).items():
            archive.writestr(name, content)
        for name, content in screenshots.items():
            archive.writestr("screenshots/" + name, content)
    return buf.getvalue()
