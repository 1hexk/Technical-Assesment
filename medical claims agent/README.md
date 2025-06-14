## Assignment 2 · Medical‑Claims Q\&A Agent

**Folder:** `medical claims agent/`

### 1 · Problem & Goal

Build an AI agent that answers natural‑language questions on a medical‑claims dataset (\~12 k rows) with answers sourced directly from SQL.

| Recruiter requirement               | Where covered                           |
| ----------------------------------- | --------------------------------------- |
| 1 Model choice & rationale          | §3 Model Selection & §3.1 Rationale     |
| 2 Evaluation dataset                | §4 Evaluation Set                       |
| 3 Performance report & improvements | §5 Performance Report & §6 Upgrade Path |
| 4 Error analysis                    | §6 Error Analysis                       |

### 2 · Dataset & Pre‑processing

| Step                                                                                   | Purpose                             |
| -------------------------------------------------------------------------------------- | ----------------------------------- |
| Load `HealthCare_Claims.xlsx` → pandas                                                 | Single source of truth              |
| Trim whitespace & title‑case categoricals                                              | Remove hidden mismatch bugs         |
| Compute `Length_of_Stay = discharge_date – admission_date`                             | Back‑fills missing LOS              |
| Fill missing `Gender` with “Not Specified” *(future: name‑frequency + LLM imputation)* | Keeps gender‑split KPIs possible    |
| Persist to **DuckDB** (`claims_tbl`)                                                   | OLAP‑grade SQL in one portable file |

> **Why SQL instead of “flatten‑then‑embed”?**
>
> * Flattening the table and embedding every row locks users into record‑level look‑ups (e.g. “what’s the billed amount for encounter 123?”).
> * Real analysts need joins, cohorts, window functions, time‑series roll‑ups, and ad‑hoc aggregates. SQL gives all that for free.
> * SQL also future‑proofs the agent for multi‑table or multi‑database scenarios—exactly where text‑to‑SQL shines.
> * With only one tiny table we embed the full schema, column names, and canonical look‑up values directly in the prompt (zero‑latency, no retriever). At enterprise scale we’d switch to embedding every schema, vector search the relevant tables, and let the agent fetch distinct look‑ups on demand.

### 3 · Model Selection

| Component            | Checkpoint (4‑bit)                 | Why this pick                                                                                                                           |
| -------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Text‑to‑SQL**      | **Arctic‑Text2SQL‑R1‑7B.Q4\_K\_M** | Highest open‑source accuracy‑per‑VRAM; \~68 % on BIRD‑dev, \~89 % Spider‑test; fits into 7 GB and still leaves headroom on a 12 GB GPU. |
| **Answer formatter** | **Llama‑3‑8B‑Instruct.Q3\_K\_M**   | 200k context, robust reasoning, fluent plain‑English (and Arabic) explanations, yet <9 GB at 3‑bit. Completes the storytelling layer.   |

#### 3.1 · Rationale

* **Accuracy vs resources** We benchmarked several OSS text‑to‑SQL models; Arctic‑7B matched 13 B‑parameter models while halving VRAM.
* **Quantisation** 4‑bit & 3‑bit GGUF cuts memory \~75 % with <2 % accuracy drop—critical for the 12 GB limit on the test box.
* **Small dataset, big future** Today: one table so schema is in‑prompt. Tomorrow: hundreds of DBs—retriever + larger LMs can drop‑in because the agent interface stays unchanged.
* **Lightweight deployment** Both checkpoints load in <20 s and answer in \~3 s, making local notebooks and cheap GPUs viable.

### 4 · Evaluation Set

`eval_questions_100.jsonl` (104 items) generated with GPT‑4:

* Mix of English & Arabic queries
* Aggregates, sub‑filters, edge‑case dates
* Gold SQL and ground‑truth answer for each

### 5 · Performance Report

| Metric          | Score                           |
| --------------- | ------------------------------- |
| SQL validity    | **100 %**                       |
| Answer accuracy | **98 % strict · 100 % logical** |
| Mean latency    | **3.4 s**                       |
| P95 latency     | **5.5 s**                       |

### 6 · Error Analysis & Upgrade Path

| Issue                         | Why it matters                  | Current workaround         | Next step                                        |
| ----------------------------- | ------------------------------- | -------------------------- | ------------------------------------------------ |
| Most `Gender` null            | Breaks gender analytics         | Fill “Not Specified”       | Name‑freq + LLM imputation, store confidence     |
| Hard‑coded lookup lists       | New unseen values break prompts | Title‑case & static list   | Agent tool: `SELECT DISTINCT col LIMIT n`        |
| Single‑table schema in prompt | Doesn’t scale                   | Fine for this demo         | Vector‑embed all schemas, retriever context      |
| Arabic diacritics & variants  | Occasional filter misses        | None                       | Pre‑normalise Arabic; switch to multilingual LLM |
| Strict evaluation formatter   | 2/104 labelled false            | Numeric/string exact match | Fuzzy compare, tolerance ±1e‑6                   |

### 7 · System Flow

```
User NL
   ↓
Arctic‑7B (text‑to‑SQL)
   ↓ validated SELECT (3‑try guard‑rail, DDL/DML filtered)
DuckDB execution
   ↓ top‑50 rows
Llama‑3‑8B (answer formatter)
   ↓
Natural‑language answer
```

### 8 · Reproduce Locally

```bash
git clone https://github.com/1hexk/Technical-Assesment.git
cd Technical-Assesment
python -m venv .venv && .\.venv\Scripts\activate   # mac/linux: source .venv/bin/activate
pip install -r requirements.txt

python medical_claims_agent\evaluate.py
python medical_claims_agent\medical_claims.py
```

to install Llama-cpp with CUDA support on windows, you can follow this giude: [Link](https://medium.com/@eddieoffermann/llama-cpp-python-with-cuda-support-on-windows-11-51a4dd295b25)