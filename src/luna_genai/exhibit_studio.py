"""CPU learning and reproducible output validation for the W13B exhibit studio."""
from __future__ import annotations

import csv
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .design_models import QwenDesign
from .event_design import response_mask
from .popup_studio import MODEL_ID, REVISION

ROOT = Path(__file__).with_name("assets") / "w13b"
FIELDS = ("종류", "장소", "관람료", "예약")
UNKNOWN = "확인 필요"
RANKS = (2, 4)
STEPS = 24
DECODING = {"max_new_tokens": 128, "do_sample": False}
FORMAT = "w13b-exhibit-1"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(value):
    return sha256(to_json(value).encode("utf-8")).hexdigest()


def dataset():
    cases = load_json(ROOT / "exhibits.json")
    assert len(set(row["id"] for row in cases)) == 33
    assert sum(row["split"] == "train" for row in cases) == 24
    assert sum(row["split"] == "validation" for row in cases) == 9
    return cases


def fact_line(facts):
    return " | ".join(f"{field}={facts[field]}" for field in FIELDS)


def messages(case):
    field_source = "\n".join(f"{field}: {value}" for field, value in case["facts"].items()
                             if value != UNKNOWN)
    instruction = (
        "캠퍼스 리폼 전시의 한국어 작품 안내문을 작성하세요. 자료의 값을 그대로 복사하세요. "
        "자료가 없는 항목은 확인 필요라고 쓰세요. "
        "첫 줄은 종류=값 | 장소=값 | 관람료=값 | 예약=값 형식, "
        "두 번째 줄은 작품을 소개하는 짧은 한국어 문장 하나입니다. 다른 말을 덧붙이지 마세요."
    )
    content = f"작품명: {case['title']}\n작품 설명: {case['material']}\n운영 메모:\n{field_source}"
    return [{"role": "system", "content": instruction}, {"role": "user", "content": content}]


def inspect_output(raw, case):
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    claims = {}
    if lines:
        for segment in lines[0].split("|"):
            key, sep, value = segment.partition("=")
            if sep and key.strip() in FIELDS and key.strip() not in claims:
                claims[key.strip()] = value.strip()
    checks = []
    for field in FIELDS:
        match = claims.get(field) == case["facts"][field]
        status = "일치" if match else "자료 밖 값" if claims.get(field) not in (None, "", UNKNOWN) else "누락·미확정"
        checks.append({"field": field, "claim": claims.get(field, ""), "reference": case["facts"][field],
                       "status": status, "source": case["id"] + "-MEMO" if case["facts"][field] != UNKNOWN else ""})
    format_ok = (
        len(lines) >= 2 and len(lines[0].split("|")) == 4 and tuple(claims) == FIELDS
        and all(claims.values()) and bool(lines[1].strip())
    )
    return {"fields": checks, "format_ok": format_ok, "raw": raw,
            "intro": lines[1] if format_ok else "",
            "publishable": format_ok and all(item["status"] == "일치" for item in checks)}


class ExhibitTrainer(QwenDesign):
    def __init__(self, threads=2):
        super().__init__(threads=threads, model_id=MODEL_ID, model_revision=REVISION,
                         generation_settings=DECODING)

    def prompt(self, case):
        return self.tokenizer.apply_chat_template(messages(case), tokenize=False,
                                                  add_generation_prompt=True)

    def tokens(self, case):
        prefix = self.tokenizer.encode(self.prompt(case), add_special_tokens=False)
        target = fact_line(case["facts"]) + "\n" + case["target"]
        response = self.tokenizer.encode(target, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return {"event": case["id"], "prefix": prefix, "response": response}

    def evaluate(self, cases, rank):
        if not cases or any(row["split"] == "train" for row in cases):
            raise ValueError("검증 또는 최종 평가 분할을 선택하세요.")
        if rank:
            self.activate(rank, STEPS)
        from contextlib import nullcontext
        frozen = self.model.disable_adapter() if self.installed and not rank else nullcontext()
        nll = []
        with frozen, self.torch.inference_mode():
            for case in cases:
                example = self.tokens(case)
                prefix, response = example["prefix"], example["response"]
                loss = self.model(input_ids=self.torch.tensor([prefix + response]),
                                  labels=self.torch.tensor([response_mask(prefix, response)]),
                                  use_cache=False).loss
                nll.append((round(float(loss) * len(response), 6), len(response)))
        # Batch only a few prompts at once to bound CPU memory.
        batch = 3
        outputs = []
        for offset in range(0, len(cases), batch):
            prompts = [self.prompt(case) for case in cases[offset:offset+batch]]
            outputs.extend(self.generate_batch(prompts, rank, STEPS))
        rows = []
        for case, result, (total, tokens) in zip(cases, outputs, nll):
            row = self.record(case, result, rank)
            row.update(nll_sum=total, response_tokens=tokens)
            rows.append(row)
        return rows

    def record(self, case, result, rank):
        info = inspect_output(result["raw"], case)
        row = {
            "case_id": case["id"], "rank": int(rank), "result": info,
            "model": MODEL_ID, "revision": REVISION, "mode": "live",
            "prompt_sha256": result["prompt_sha256"],
            "adapter_sha256": result.get("adapter_sha256"),
        }
        row["run_id"] = digest(row)[:16]
        return row

    def generate_case(self, case, rank):
        return self.record(case, self.generate(self.prompt(case), rank, STEPS), rank)


def summarize(rows):
    return {
        "rank": rows[0]["rank"],
        "cases": len(rows), "fields": len(rows) * len(FIELDS),
        "match": sum(f["status"] == "일치" for row in rows for f in row["result"]["fields"]),
        "unsupported": sum(f["status"] == "자료 밖 값" for row in rows for f in row["result"]["fields"]),
        "invalid": sum(not row["result"]["format_ok"] for row in rows),
        "nll": round(sum(row["nll_sum"] for row in rows) / sum(row["response_tokens"] for row in rows), 5),
    }


def select(rows):
    valid = [row for row in rows if not row["unsupported"] and not row["invalid"]]
    return min(valid, key=lambda x: (-x["match"], x["rank"]))["rank"] if valid else None


def html_for(case, record, comparison, mode):
    payload = to_json({"case": case, "record": record, "comparison": comparison, "mode": mode})
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return (ROOT / "studio.template.html").read_text(encoding="utf-8").replace("__STUDIO_DATA__", payload)


def comparison_csv(rows):
    out = StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def bundle(path, training, evidence, case, record, mode="live"):
    metrics = [summarize(evidence[str(rank)]) for rank in RANKS]
    manifest = {"format": FORMAT, "mode": mode, "model": MODEL_ID, "revision": REVISION,
                "decoding": DECODING, "selected": select(metrics), "case_id": case["id"],
                "applied_rank": record["rank"], "run_id": record["run_id"]}
    screen = html_for(case, record, metrics, mode)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in {
            "manifest.json": manifest, "training.json": training, "evaluation.json": evidence, "app-run.json": record,
        }.items():
            archive.writestr(name, to_json(content))
        archive.writestr("verification.csv", comparison_csv(metrics))
        archive.writestr("index.html", screen)
    return manifest
