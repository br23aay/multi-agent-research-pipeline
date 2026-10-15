# Multi-Agent Research Pipeline

A **LangGraph** multi-agent pipeline that turns a research question into a structured, **quality-gated** report — with a **human approval gate** before anything is finalised. Built to demonstrate agent orchestration, self-correction loops, and human-in-the-loop control rather than a single monolithic prompt.

> Status: working prototype (single-file graph). Runs end to end against the Groq API.

---

## What it does

Five specialised agent nodes, wired as a stateful graph:

| Node | Role |
|---|---|
| **Router** | Classifies the query as `technical`, `business`, or `general` |
| **Researcher** | Role-specific research; incorporates prior quality feedback on retries |
| **Synthesizer** | Structures findings into Summary → Key Points → Recommendations |
| **Quality Check** | Scores the report 1–10 on completeness + actionability, returns feedback |
| **Human Review** | LangGraph `interrupt()` — pauses for an `approve` / `revise` decision |

**Control flow:** if the quality score is below 7 the report loops back to the Researcher with the feedback, up to **3 iterations**, then proceeds to human review with a warning. State is persisted through an `InMemorySaver` checkpointer so the graph can be **interrupted and resumed** at the human-review step.

---

## Stack

- **LangGraph** — stateful multi-agent graph, conditional edges, interrupts, checkpointing
- **LangChain** — chat model abstraction
- **Groq** — `llama-3.1-8b-instant` (fast, low-cost inference)
- **Python** — typed `TypedDict` state schema

---

## Run

```bash
pip install langgraph langchain-groq langchain
setx GROQ_API_KEY "your_key"      # or $env:GROQ_API_KEY in PowerShell
python research_pipeline.py
```

The pipeline prompts for a research question, runs router → researcher → synthesizer → quality-gate, then pauses for your approval before printing the final report.

---

## Design notes

- **Self-correction over one-shot:** the quality gate makes the system critique and retry its own work, with an explicit iteration cap so it can't loop forever.
- **Human-in-the-loop by construction:** approval is a first-class graph node (`interrupt`), not an afterthought — the graph genuinely pauses and resumes on the human's decision.
- **Explainable routing:** classification and quality decisions are logged at each step.
- **Rate-limit aware:** deliberate pauses between heavy calls keep it inside free-tier limits.

---

## Roadmap

- Add a real **web-search / retrieval tool** to the Researcher (currently model-knowledge only).
- Swap `InMemorySaver` for a **persistent checkpointer** (SQLite/Postgres) so sessions survive restarts.
- **Multi-model routing** (e.g. a stronger model for synthesis, a cheaper one for classification).
- A small **evaluation harness** to track report quality across runs.
- Streaming output and a lightweight web UI.

---

## Author

**Bharadwaj Rachuri** — ML & AI Engineer · [br23aay.github.io](https://br23aay.github.io) · [github.com/br23aay](https://github.com/br23aay)

*Independent project, built and run locally. No credentials are committed — the Groq API key is read from the environment.*
