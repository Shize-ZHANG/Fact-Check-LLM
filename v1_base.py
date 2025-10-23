#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, time, re
from typing import Iterator, Dict, Any, Optional
from tqdm import tqdm

# pip install openai>=1.0.0
from openai import OpenAI

LABELS = ["supports", "refutes", "NOT ENOUGH INFO"]

SYSTEM_PROMPT = (
    "你是严格的事实核验分类器。只依据输入的中文陈述句本身，不查阅任何外部资料。\n"
    "请将该陈述句的事实状态判为以下三者之一，并且只输出其中一个标签（不要解释）：\n"
    "supports（支持为真）\n"
    "refutes（反驳为假）\n"
    "NOT ENOUGH INFO（信息不足，无法判断）"
)
USER_TMPL = "陈述句：{claim}\n只输出一个：supports / refutes / NOT ENOUGH INFO"

def read_jsonl(path:str)->Iterator[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def normalize_label(s: str) -> str:
    t = s.strip()
    tl = t.lower()
    tu = t.upper()
    if tl == "supports": return "supports"
    if tl == "refutes": return "refutes"
    if tu == "NOT ENOUGH INFO": return "NOT ENOUGH INFO"
    # 容错中文答案
    if "支持" in t or "为真" in t or "為真" in t: return "supports"
    if "反驳" in t or "反駁" in t or "为假" in t or "為假" in t: return "refutes"
    if "不确定" in t or "不確定" in t or "信息不足" in t or "資訊不足" in t: return "NOT ENOUGH INFO"
    # 容错英文包含
    if re.search(r"\bsupports?\b", tl): return "supports"
    if re.search(r"\brefutes?\b", tl): return "refutes"
    if "not enough info" in tl: return "NOT ENOUGH INFO"
    return "NOT ENOUGH INFO"

def build_client(base_url:str, api_key:str)->OpenAI:
    return OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=api_key)

def chat_once(client:OpenAI, model:str, claim:str, max_retries:int=3)->str:
    msgs = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":USER_TMPL.format(claim=claim)}
    ]
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=msgs,
                temperature=0.0, top_p=1.0, max_tokens=8
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if i+1==max_retries: raise
            time.sleep(1.0 + 1.5*i)
    return "NOT ENOUGH INFO"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 JSONL（train/dev/test 均可）")
    ap.add_argument("--output", required=True, help="输出 JSONL（包含 claim / predicted_label / label）")
    ap.add_argument("--base_url", default=f"http://127.0.0.1:{os.environ.get('VLLM_PORT','8009')}",
                    help="vLLM OpenAI 端点，如 http://127.0.0.1:8009")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY","sk-xxx"))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME","qwen3-32b"))
    ap.add_argument("--limit", type=int, default=0, help="调试用，>0 仅推理前 N 条")
    args = ap.parse_args()

    client = build_client(args.base_url, args.api_key)

    n=0; n_has_label=0; n_correct=0
    with open(args.output, "w", encoding="utf-8") as wf:
        for i, ex in enumerate(tqdm(read_jsonl(args.input), desc="Infer")):
            if args.limit and i>=args.limit: break
            claim = ex.get("claim","").strip()
            if not claim:
                continue
            raw = chat_once(client, args.model, claim)
            pred = normalize_label(raw)
            if pred not in LABELS:
                pred = "NOT ENOUGH INFO"

            gold = ex.get("label", None)
            if isinstance(gold, str):
                gold_norm = normalize_label(gold)
                # 与要求一致，回写 gold 为其原始值；但用于计算用 gold_norm
                n_has_label += 1
                if pred == gold_norm:
                    n_correct += 1
            else:
                gold_norm = None

            out = {
                "claim": claim,
                "predicted_label": pred,
                "label": gold if gold is not None else None
            }
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            n+=1

    print(f"[Done] wrote {n} lines to {args.output}")
    if n_has_label>0:
        acc = n_correct / n_has_label
        print(f"[Eval] accuracy = {acc:.4f}  ({n_has_label} labeled examples)")

if __name__ == "__main__":
    main()
