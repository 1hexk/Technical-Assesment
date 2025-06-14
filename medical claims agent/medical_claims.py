import duckdb, pandas as pd, re, sqlparse, time, math
from llama_cpp import Llama

EXCEL = "HealthCare_Claims.xlsx"
DB = "health.duckdb"
TABLE = "claims_tbl"
CTX = 8192
TOPK = 50

df = pd.read_excel(EXCEL)

df = df.rename(columns={"name": "Patient_name"})

str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

df["Gender"] = (
    df["Gender"]
    .fillna("Not Specified")
    .replace({"M": "Male", "F": "Female"})
)

for col in ["admission_type", "medication", "test_resultS", "MedicalCondition"]:
    if col in df.columns:
        df[col] = df[col].str.title()

df["Length_of_Stay"] = (df["discharge_date"] - df["admission_date"]).dt.days

con = duckdb.connect(DB)
con.execute(f"DROP TABLE IF EXISTS {TABLE}")
con.register("tmp", df)
con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM tmp")
con.close()

db = duckdb.connect(DB)
cols = [r[1] for r in db.execute(f"PRAGMA table_info('{TABLE}')").fetchall()]
col_list = ", ".join(f'"{c}"' for c in cols)

SQL_LLM = Llama(model_path="models/arctic.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, chat_format="chatml")
ANS_LLM = Llama(model_path="models/Meta-Llama-3-8B-Instruct-IQ3_M.gguf", n_gpu_layers=-1, n_ctx=CTX, chat_format="chatml")

SYSTEM_SQL = {
    "role": "system",
    "content": (
        "You are an expert DuckDB 0.10 SQL generator. "
        "Your ONLY task is to translate the user-question into a single, read-only "
        "SELECT statement that follows all constraints."
    )
}

SYSTEM_ANS = {
    "role": "system",
    "content": (
        "You are a senior data-analyst. Given a small table of result rows "
        "and the user’s question, deliver:\n"
        "• a concise detailed explanation drawn strictly from the rows.\n"
        "Never show SQL or invent numbers."
    )
}

def chat_sql(p, **k):
    m = [SYSTEM_SQL, {"role": "user", "content": p}]
    return SQL_LLM.create_chat_completion(m, **k)["choices"][0]["message"]["content"].strip()

def chat_ans(p, **k):
    m = [SYSTEM_ANS, {"role": "user", "content": p}]
    return ANS_LLM.create_chat_completion(m, **k)["choices"][0]["message"]["content"].strip()

FORBID = ("insert", "update", "delete", "drop", "alter", "create")

def extract_sql(t):
    for pat in [r"(?i)<SQL>([\s\S]+?)</SQL>", r"```sql([\s\S]+?)```", r"(?i)(SELECT[\s\S]+)"]:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(";")
    return None

def safe(q):
    if any(k in q.lower() for k in FORBID):
        return False
    p = sqlparse.parse(q)
    return len(p) == 1 and p[0].get_type() == "SELECT"

def rows_to_text(d, lim=TOPK):
    if d.empty:
        return "<empty set>"
    d = d.head(lim)
    return "\n".join("; ".join(f"{c}: {r[c]}" for c in d.columns) for _, r in d.iterrows())

EXAMPLES_SQL = """
-- Q: How many claims exist?
<SQL>
SELECT COUNT(*) FROM claims_tbl;
</SQL>
-- Q: Average length of stay for female patients?
<SQL>
SELECT AVG("Length_of_Stay") FROM claims_tbl WHERE "Gender" = 'Female';
</SQL>
-- Q: Total billing amount in 2024?
<SQL>
SELECT SUM("billing_amount") FROM claims_tbl WHERE YEAR("admission_date") = 2024;
</SQL>
"""

EXAMPLES_SQL += """
-- Q: كم عدد المطالبات لـ 'شركة القلعة للتأمين' ؟
<SQL>
SELECT COUNT(*) FROM claims_tbl
WHERE "insurance_provider_name" = 'شركة القلعة للتأمين';
</SQL>

-- Q: Median billing amount (all claims)?
<SQL>
SELECT MEDIAN(billing_amount) FROM claims_tbl;
</SQL>
"""


lookup_gender = " | ".join(df["Gender"].dropna().unique())
lookup_adm = " | ".join(df["admission_type"].dropna().unique()[:10])
lookup_med  = " | ".join(df["medication"].dropna().unique()[:8])
lookup_test = " | ".join(df["test_resultS"].dropna().unique()[:8])
lookup_cond = " | ".join(df["MedicalCondition"].dropna().unique()[:8])


LOOKUP_NOTE = f"""
Valid filters →
• Gender: {lookup_gender}
• admission_type: {lookup_adm}
• MedicalCondition: {lookup_cond}
• medication: {lookup_med}
• test_resultS: {lookup_test}
Default date field = admission_date unless question says discharge_date.
"""

PROMPT_SQL = f"""
You are a DuckDB 0.10 SQL generator.
{EXAMPLES_SQL}
{LOOKUP_NOTE}
Hard rules:
• Only table "{TABLE}" columns: {col_list}
• Single read-only SELECT inside <SQL> … </SQL>, no extra text.
Question: {{q}}
"""

PROMPT_ANS = """
Columns: {cols}
Rows:
{rows}
Question: {q}
Answer (≤3 sentences, or single value if one cell):
"""

def execute_safe(s):
    try:
        return db.sql(s).df(), None
    except Exception as e:
        return None, str(e)

def answer_question(q, max_try=3):
    t0 = time.time()
    sql = None
    for i in range(max_try):
        prompt = PROMPT_SQL.format(q=q) if i == 0 else f"Previous failed. {PROMPT_SQL.format(q=q)}"
        resp = chat_sql(prompt, max_tokens=512)
        sql = extract_sql(resp)
        if sql and safe(sql):
            break
        sql = None
    if sql is None:
        return "NO_SQL", None, False, time.time() - t0
    rows, err = execute_safe(sql)
    if err:
        return f"SQL_ERROR: {err}", sql, False, time.time() - t0
    if rows.shape == (1, 1):
        return str(rows.iat[0, 0]), sql, True, time.time() - t0
    ans = chat_ans(PROMPT_ANS.format(cols=", ".join(rows.columns), rows=rows_to_text(rows), q=q), max_tokens=128, temperature=0.0)
    return ans.strip(), sql, True, time.time() - t0

def ask(q):
    return answer_question(q)

if __name__ == "__main__":
    while True:
        q = input(">> ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        print(ask(q))
