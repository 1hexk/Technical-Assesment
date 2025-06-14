import json, re, math, pandas as pd
from medical_claims import ask

EVAL_FILE = "eval_questions_100.jsonl"

def equal(exp, got, abs_tol=1e-2, rel_tol=1e-3):
    def first_num(t):
        m = re.search(r'-?\d[\d,]*(?:\.\d+)?', str(t))
        return None if not m else float(m.group().replace(",", ""))
    n1, n2 = first_num(exp), first_num(got)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) <= abs_tol or abs(n1 - n2) / (abs(n1) + 1e-9) <= rel_tol
    if isinstance(exp, list):
        tok = lambda txt: {x.strip().lower() for x in re.split(r'[,\n|;]+', str(txt)) if x.strip()}
        return tok(exp) == tok(got)
    return str(exp).strip() == str(got).strip()

cases = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8")]

out = []
for c in cases:
    ans, sql, valid, dt = ask(c["nl"])
    out.append({
        "id": c["id"],
        "nl": c["nl"],
        "ground_truth": c["answer"],
        "llm_answer": ans,
        "sql_used": sql,
        "sql_valid": valid,
        "correct": valid and equal(c["answer"], ans),
        "latency_s": dt
    })

df = pd.DataFrame(out)
df.to_csv("evaluation_full.csv", index=False)
df[~df["correct"]].to_csv("evaluation_failures.csv", index=False)
pd.Series({
    "n": len(df),
    "sql_valid_pct": 100 * df["sql_valid"].mean(),
    "accuracy_pct": 100 * df["correct"].mean(),
    "avg_latency": df["latency_s"].mean(),
    "p95_latency": df["latency_s"].quantile(0.95)
}).to_csv("evaluation_summary.csv", header=False)
print("evaluation complete")