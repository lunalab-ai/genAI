"""W13B visitor-card experiments and the no-server comparison export."""
from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .exhibit_studio import FIELDS, MODEL_ID, REVISION, ROOT, UNKNOWN
from .exhibit_studio import dataset, digest, inspect_output, load_json, messages, to_json

SCHEMA = "w13b-cards-v2"
MEMO_SCHEMA = "w13b-memo-v2"
COLAB = ("https://colab.research.google.com/github/lunalab-ai/genAI/blob/"
         "2026-fall-w13b-r3/notebooks/student/w13b_exhibit_r2.ipynb")
VALIDATION = ("BAG-07", "BAG-10", "TEE-11")
AUDIENCES = ("공통", "처음 방문한 학생", "리폼에 관심 있는 학생")
TEMPLATE = Path(__file__).with_name("assets") / "w13b-r2" / "studio.html"


def lesson_cases():
    by_id = {case["id"]: case for case in dataset() if case["split"] == "validation"}
    return [by_id[key] for key in VALIDATION]


def memo_case(memo):
    """Turn an editable memo into a new, ungraded exhibit case."""
    if memo.get("schema") != MEMO_SCHEMA:
        raise ValueError("W13B 운영 메모 JSON을 사용하세요.")
    title = str(memo.get("title", "")).strip()
    material = str(memo.get("material", "")).strip()
    audience = memo.get("audience", "공통")
    if not 1 <= len(title) <= 80 or not 1 <= len(material) <= 650 or audience not in AUDIENCES:
        raise ValueError("작품명, 재료·리폼 과정, 관람객을 확인하세요.")
    fields = memo.get("facts")
    if not isinstance(fields, dict) or set(fields) != set(FIELDS):
        raise ValueError("종류·장소·관람료·예약 네 항목을 넣으세요.")
    facts = {field: str(fields[field]).strip() or UNKNOWN for field in FIELDS}
    if facts["종류"] == UNKNOWN or facts["장소"] == UNKNOWN:
        raise ValueError("종류와 장소는 입력하세요.")
    if any(len(value) > 60 or re.search(r"[\r\n|=]", value) for value in facts.values()):
        raise ValueError("사실값은 한 줄로 입력하고 |와 =는 빼세요.")
    case = {"title": title, "material": material, "facts": facts,
            "audience": audience, "split": "practice"}
    case["id"] = "MEMO-" + digest(case)[:10]
    return case


def draft_prompt(trainer, case):
    dialogue = messages(case)
    dialogue[0]["content"] = (
        "한국어 리폼 전시 관람 안내 카드입니다. 운영 메모로 아래 정확히 두 줄을 쓰세요. "
        "첫 줄: 종류=값 | 장소=값 | 관람료=값 | 예약=값. 빠진 사실은 확인 필요. "
        "둘째 줄: 재료가 어떻게 바뀌었는지 한 문장, 무엇을 살펴볼지 한 문장. "
        "운영 메모 제목을 반복하지 말고 자료에 없는 이야기를 덧붙이지 마세요. "
        "말을 건넬 사람: " + case["audience"] + "."
    )
    return trainer.tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)


def measures(raw, case):
    """Extract observable checks from the raw text instead of editing the output."""
    legacy = inspect_output(raw, case)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    narrative = "\n".join(lines[1:]) if len(lines) > 1 else ""
    strict = bool(len(lines) == 2 and legacy["format_ok"])
    longest, streak, previous = 0, 0, None
    for word in re.findall(r"[가-힣A-Za-z0-9]+", narrative.lower()):
        streak = streak + 1 if word == previous else 1
        longest = max(longest, streak)
        previous = word
    matches = sum(f["status"] == "일치" for f in legacy["fields"])
    return {"fields": legacy["fields"], "matches": matches,
            "strict": strict, "line_count": len(lines), "repetition": longest,
            "narrative": narrative}


def row_for(record, case, *, name, steps, temperature, rule):
    raw = record["result"]["raw"] if "result" in record else record["raw"]
    m = measures(raw, case)
    issues = []
    if m["matches"] != len(FIELDS):
        issues.append("운영 메모와 다른 값 또는 누락")
    if not m["strict"]:
        issues.append(f"두 줄 형식 확인 · 현재 {m['line_count']}줄")
    if m["repetition"] >= 3:
        issues.append(f"같은 단어 {m['repetition']}회 연속")
    nll = record["nll_sum"] / record["response_tokens"] if "nll_sum" in record else None
    approved = not issues if rule is None else bool(rule(m["matches"], m["strict"], m["repetition"]))
    return {"name": name, "steps": steps, "temperature": temperature, "rank": record["rank"],
            "raw": raw, "nll": nll, "run_id": record["run_id"],
            "adapter_sha256": record.get("adapter_sha256"), "issues": issues, **m,
            "approved": approved}


def replay_items():
    """Re-score the exact previous CPU runs without loading model weights."""
    reference = load_json(ROOT / "reference.json")
    runs = {rank: {r["case_id"]: r for r in rows} for rank, rows in reference["evaluation"].items()}
    items = []
    for case in lesson_cases():
        key = case["id"]
        rows = [row_for(runs[str(rank)][key], case, name=f"{name} · rank {rank}",
                        steps=24, temperature=None, rule=None)
                for name, rank in (("A", 2), ("B", 4))]
        items.append(comparison(case, "rank-replay", "rank · 24회 실제 수업 기록",
                                "동일 사례 · Greedy · 24회 대 24회", rows, "replay"))
    return items


def comparison(case, kind, title, caption, rows, origin="live"):
    return {"key": case["id"], "kind": kind, "title": title, "caption": caption,
            "rows": rows, "origin": origin, "case": case}


def validate_design(design):
    if not isinstance(design, list) or len(design) != 2:
        return False
    return ([r.get("rank") for r in design] == [2, 4] and
            [r.get("steps") for r in design] == [4, 4] and
            [r.get("alpha") for r in design] == [4, 8])


def fit_pair(trainer, train, design, mask, folder):
    if not validate_design(design):
        raise ValueError("A/B의 rank·steps·alpha를 확인하세요.")
    work = Path(folder)
    reports = {}
    for spec in design:
        rank, steps = spec["rank"], spec["steps"]
        print(f"rank {rank} · {steps}회 업데이트 시작", flush=True)
        report = trainer.fit(train, work / f"rank-{rank}", rank=rank, steps=steps, mask=mask)
        reports[rank] = report
        print(f"{report['trainable']:,}개 모수 · {report['training_seconds']:.1f}초", flush=True)
    a, b = reports[2], reports[4]
    ids = lambda r: [v["event"] for v in r["updates"]]
    if not a["base_before"] == a["base_after"] == b["base_before"] == b["base_after"] or ids(a) != ids(b):
        raise RuntimeError("기본 가중치 또는 학습 사례 순서가 달라졌습니다.")
    return reports


def rank_items(trainer, cases, rule, steps=4):
    left = trainer.evaluate(cases, 2, steps=steps)
    right = trainer.evaluate(cases, 4, steps=steps)
    items = []
    for case, a, b in zip(cases, left, right, strict=True):
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise RuntimeError("rank 비교의 프롬프트가 서로 다릅니다.")
        rows = [row_for(r, case, name=f"{name} · rank {r['rank']}", steps=steps,
                        temperature=None, rule=rule)
                for name, r in (("A", a), ("B", b))]
        items.append(comparison(case, "rank-live", f"rank · 내 {steps}회 학습",
                                f"동일 사례 · Greedy · {steps}회 대 {steps}회", rows))
    return items


def valid_sampling(settings, temperature):
    return isinstance(settings, dict) and settings == {
        "do_sample": True, "temperature": temperature, "top_k": 0,
        "top_p": 1.0, "max_new_tokens": 112,
    }


def temperature_item(trainer, case, rank, steps, make_settings, rule, *, audience_prompt=False):
    prompt = draft_prompt(trainer, case) if audience_prompt else trainer.prompt(case)
    original = trainer.generation_settings
    rows = []
    try:
        for name, temperature in (("A", 0.35), ("B", 0.85)):
            settings = make_settings(temperature)
            if not valid_sampling(settings, temperature):
                raise ValueError("temperature 외의 생성 설정을 비교표와 맞추세요.")
            trainer.torch.manual_seed(42)
            trainer.generation_settings = settings
            result = trainer.generate(prompt, rank, steps)
            options = {"rank": rank, "steps": steps, "seed": 42, "decoding": settings}
            result.update(rank=rank, run_id=digest({"case": case["id"], "raw": result["raw"],
                "prompt_sha256": result["prompt_sha256"], "adapter_sha256": result["adapter_sha256"],
                "options": options})[:16])
            rows.append(row_for(result, case, name=f"{name} · T {temperature}", steps=steps,
                                temperature=temperature, rule=rule))
    finally:
        trainer.generation_settings = original
    if rows[0]["adapter_sha256"] != rows[1]["adapter_sha256"]:
        raise RuntimeError("생성 설정 비교의 어댑터가 다릅니다.")
    caption = f"rank {rank} · {steps}회 학습 어댑터 · 시드 42 · 두 조건에서 같은 메모와 프롬프트"
    return comparison(case, "temperature", "temperature · 내 생성", caption, rows)


def export_data(items):
    return {"schema": SCHEMA, "model": MODEL_ID, "revision": REVISION, "items": items}


def screen_html(items, initial=None):
    data = to_json({**export_data(items), "initial": initial, "colab": COLAB})
    escaped = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return TEMPLATE.read_text(encoding="utf-8").replace("__SHOWCASE_JSON__", escaped)


def write_artifacts(folder, items, initial=None):
    """Store result JSON, an interactive offline HTML page, and their ZIP."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data, page = folder / "w13b-my-results.json", folder / "w13b-cards.html"
    data.write_bytes((to_json(export_data(items)) + "\n").encode("utf-8"))
    page.write_bytes(screen_html(items, initial).encode("utf-8"))
    bundle = folder / "w13b-cards.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
        archive.write(page, "index.html")
        archive.write(data, data.name)
        archive.writestr("읽어주세요.txt", "압축을 풀고 index.html을 여세요. 화면에는 저장된 두 모델 결과가 표시됩니다. 새 메모는 Colab에서 생성합니다.\n")
    return data, page, bundle
