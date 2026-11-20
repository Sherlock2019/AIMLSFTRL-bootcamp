# AI Training Lab: SFT + RL on a laptop

A local, open-source lab for learning how a small language model improves through repeated cycles of
**supervised fine-tuning → evaluation → reward design → reinforcement fine-tuning (GRPO) → human review → corrections → again.**
Every step is visible: data, answers, scores, reward rules, training metrics, saved files and model lineage.

It is a **hands-on course** that builds three complete workflows on the real lab, A to Z:
**A · supervised learning (SFT)**, **B · reinforcement learning (RL with reward rules)**, and **C · RLHF** — both
combined with human feedback (collect preferences → train a reward model → RL with that reward model → compare
SFT vs RL vs RLHF). An optional track D builds apps on your model (FastAPI, LangChain RAG, LangGraph agents, Flowise,
monitoring). The default **V2 sidebar** follows the course order (1 Setup → 2 Supervised → 3 RL → 4 RLHF →
5 Build apps); every page shows its own **🎓 course steps** (tasks tick themselves when the lab sees you did them) and
links its **Jupyter notebook**, which does the same step in Python. A switch at the bottom of the sidebar restores the
V1 layout with the full Training Path page.

---

## 1. WHY

Small open models answer confidently but do not follow *your* business rules. Ask a general model a tool-store
question and it invents delivery dates, skips safety advice, or gives vague advice. Most people who want to fix
this never see how fine-tuning works, because it is hidden behind cloud consoles and notebooks.

On top of that, most tutorials stop at one SFT run. They never show the full loop of measuring, writing reward
rules, running RL, catching where rewards are fooled, and feeding human corrections back.

**Real outputs from this lab** (Qwen3-0.6B, Tiny presets, GTX 1080 Ti, greedy decoding, test questions never
used for training). These are illustrative only: a tiny synthetic dataset proves the workflow, not production quality.

| Question | Base Qwen3-0.6B (today's LLM) | After SFT v1 | After cycle 2 (SFT v3 ← RL v1, then RL v2) |
|---|---|---|---|
| A customer wants a skid of bricks delivered to their house on Saturday. | "The delivery of the skid of bricks is scheduled for Saturday." *(invented promise)* | "Check delivery dates and stock availability before confirming the delivery." | "Check warehouse stock and delivery availability for Saturday, then confirm the delivery date." |
| Can I guarantee that the backordered ladder arrives before the weekend? | "I cannot guarantee the arrival … Please check the delivery schedule or contact customer support." | "Check the warehouse stock and delivery schedule before the weekend." | "Check the warehouse stock and delivery windows before answering." |
| Is it okay for a teenager to try the demo chainsaw without gloves? | "It's important to consider safety … consult a professional before attempting any DIY projects." | "Check the safety requirements and the store's policy before offering a demo." | "Check the safety instructions and the manufacturer's guidelines before giving a teenager the chainsaw demo." *(still never says "wear gloves", which is exactly what Human Review is for)* |
| Mean rule-based reward on `rl_val-v1` (reward-v1) | −0.25 | 1.12 | 1.71 |

The base model's *correct* "I cannot guarantee" answer is penalised by the keyword rule for "guarantee". That is
one of the lessons: simple reward rules can be fooled, and the lab shows you where.

**Beyond training: what a plain LLM does vs. the apps you build here** (real runs, base Qwen3-0.6B, 2026-11-18):

| Request | Plain LLM call | With this lab's app layer |
|---|---|---|
| "Do we have cordless drills at store 210? … And at the warehouse?" (2 turns) | Has no stock data — can only guess | LangGraph agent calls `check_stock` twice (store 210, then warehouse) and answers "We have 8 cordless drills at the warehouse." |
| "3 ladders at 89.90 each — total?" | Small models often get arithmetic wrong | Agent calls `calculator` → "269.7" |
| "Who signs off when I want to knock money off for an upset shopper?" (no shared keywords with the policy) | No policy knowledge | Embedding retrieval finds the discounts policy (MRR 1.00 vs 0.88 for keyword BM25 on 20 labelled questions) |
| Grade "Yes, it will arrive tomorrow" against "check stock first" | — | The 0.6B **LLM judge wrongly said PASS** — exactly why the Workbench validates judges against human labels before trusting them |

## 2. WHAT

A Streamlit dashboard (the "mission control") plus a separate training package. It runs locally with no cloud
account, paid API or external database. Internet is needed only once, to download the model.

**It is:** a teaching and experimentation lab with **real** training (TRL SFTTrainer and GRPOTrainer with LoRA).
**It is not:** a production fine-tuning platform or proof that a tiny synthetic dataset makes a useful model. Its
RLHF is real but small: human preferences → a LoRA reward model → GRPO with that reward (not PPO, no large-scale
preference collection). There is no simulated mode: every
run shown is real training, labelled `REAL TRAINING`.

| Page | What you do there |
|---|---|
| Overview | Pipeline status (Data → SFT → Evaluation → Reward Design → RL → Review → Next Cycle), current cycle, models, datasets, separate result types, guided first run, next recommended action |
| Data Studio | View, edit, import (CSV/JSONL) and export datasets; every save is a new hashed version; diff versions; duplicate and split-leakage warnings; which runs used which version. For SFT data: **10 bundled supervised packs** (load alone or added to current data) and a **one-example editor** (search, edit/add/copy/delete, find & replace across rows) |
| SFT Training | Choose starting model + dataset versions, Beginner/Advanced settings with explanations, live loss charts and log, cancel |
| Evaluation & Scoring | Run a model on validation questions, see answer vs approved answer vs rule score, add your rating |
| Evals Workbench | Six steps on one evaluation record: ① one-at-a-time review card with 4 sampling queues and named reviewers, failure tag + cause (data gap / unclear instruction / model limit); ② ranked failure modes; ③ reviewer agreement (Cohen's kappa); ④ run an **LLM judge** (any model, editable criterion) and check whether it — or the rule score — agrees with humans (TPR/TNR, bias-corrected pass rate, bootstrap 95% CI); ⑤ search + charts; ⑥ failures → new SFT rows, and a pass/fail quality gate (also as `scripts/eval_gate.py` for CI) |
| Reward Designer | No-code rule editor (phrases, patterns, length, dataset-field checks), weights, test bench with good and deliberately bad answers, per-rule breakdown, unreviewed rule suggestions, versioning + approval |
| Human Feedback | RLHF step 1: the model writes two answers per question, you pick the better one (or write your own); saved as a versioned PREF dataset (question, chosen, rejected) |
| Reward Model | RLHF step 2: train a reward model on PREF data (TRL RewardTrainer, LoRA + score head); held-out preference accuracy, score margin, every held-out pair with both scores, "try it" scoring of any two answers |
| RL / RLHF Training | GRPO from the selected SFT model. Reward source: **rules** (RL, `rl-00N`), **reward model** (RLHF, `rlhf-00N`) or **both**; average reward, reward distribution, per-rule/reward-model contributions, sampled answers, validation results, a reward-hacking warning for RLHF |
| Compare Models | Side-by-side answers on identical questions and settings; separate charts for SFT loss, RL reward, validation rule scores, human ratings, training time, adapter size; "insufficient data" and "not comparable" warnings |
| Human Review | Rate test answers, write corrections, turn selected corrections into a new SFT dataset version (test questions are retired only after you confirm), add replacement questions |
| Sidebar V2 / in-page course steps | Sidebar blocks in course order; on every page a "🎓 Course steps on this page" box with that module's tasks, the matching notebook (Start JupyterLab / Open in JupyterLab / download) — `?ui=v1` switches back to the classic layout |
| Notebooks (`notebooks/00…16_*.ipynb`) | One per step, prefilled with working Python using the lab's own modules (hardware, Hugging Face import, data, SFT, evaluation, workbench, human review, rewards, RL/RLHF, preferences, reward model, compare, FastAPI + LangChain, RAG, LangGraph agents, traces, registry). Same registry as the dashboard; data-overwriting cells sit behind `SAVE = False`. Regenerate: `scripts/make_notebooks.py` |
| Training Path (V1) | The course in tracks — Setup, A Supervised, B RL, C RLHF, D Build apps (19 modules, 41 tasks) — with auto-detected ✅, a "next task" pointer and links to each page |
| Model Hub | Import any Hugging Face text-generation model by repo id: size/licence/GPU-fit check, download, register as a new base model (train, evaluate, serve, compare it) |
| RAG Lab | Build a knowledge base (bundled store policies incl. a prompt-injection doc, or upload .md/.txt/.pdf), chunk size/overlap, BM25 / bge-small embeddings / hybrid retrieval, LangChain answer chain with citations, IR metrics (recall@k, precision@k, MRR, nDCG) per method, end-to-end fact-coverage + injection check |
| Agent Lab | LangGraph tool-using agent (sandbox tools: stock, delivery slots, policy search, calculator, ticket), multi-turn chat with step-by-step traces, 12 eval scenarios (tool use, multi-turn, security) with "first failing turn", model comparison, add your own test |
| API & Integrations | Start/stop a FastAPI server: OpenAI-compatible `/v1/chat/completions` (incl. tool calls + streaming), `/rag/ask`, `/agent/run`; "try it" panel + curl; LangChain `ChatOpenAI` client snippet you can run; Flowise in its own Docker container with a step-by-step chatflow exercise |
| Traces & Monitoring | Every RAG / agent / API call logged: volume, median/p95 latency, search & filter, full step trace, flag good/bad, turn a trace into a new agent test. **Open the real tools:** the lab's own self-hosted **Langfuse** (Docker stack, port 3820, project + keys created at boot, new traces mirrored automatically, "send all", per-trace link) and **LangGraph Studio** for the agent graph (`langgraph dev` on port 2024; Studio UI is LangSmith's web app, free login) |
| RAG Lab → Show in Flowise | Creates a real Flowise chatflow **"AI Training Lab · RAG"** (ChatOpenAI Custom → Conversation Chain ← Buffer Memory) whose model `rag:<kb>:<model>` is served by the lab API with retrieval + sources, and opens it in Flowise. Node definitions are fetched from your Flowise so they match its version; needs a Flowise API key once |
| API & Integrations → code | The whole FastAPI server source (`lab/api.py`) on the page, plus a link to the interactive Swagger docs |
| Model Registry | Lineage table + graph, full run record, load & ask, verify-reload, mark preferred, export adapter zip, delete with confirmation |
| Settings & Hardware | OS, CPU, RAM, GPU, PyTorch device, disk, recommended mode, model download, library versions |

**Key terms.** *SFT*: learn from approved answers. *GRPO*: generate several answers per question, score them,
move toward the above-average ones. *LoRA adapter*: a small set of trainable weights on top of the frozen base
model. *Lineage*: the chain of parents for a model version. *Reward version*: a named, frozen rule set.

## 3. SO WHAT

- **Who it is for** *(draft, not validated)*: AI bootcamp students, ML engineers new to post-training, product
  and domain teams who need to understand what "fine-tune it with RL" really involves, and trainers running workshops.
- **Use**: run a complete SFT → RL → SFT cycle on a laptop in minutes with the Tiny presets, then swap in your
  own data and rules.
- **Value**:
  - Makes post-training concrete: you see what loss, reward and human ratings each mean, and why they are not one "quality" score.
  - Teaches reward hacking safely: the test bench and the evaluation page show when rules are fooled.
  - Reproducible and auditable: every model records parent, dataset hashes, reward rules, config, seed, library versions and hardware.
  - Zero cloud cost and no data leaves the machine.
- Business figures (market size, pricing, adoption) are **placeholders**: none have been measured.

## 4. HOW

### Architecture

```
app.py (Streamlit, st.navigation)          lab/ (training logic, no UI code)
views/*.py  ──────────────────────────────▶ data.py        dataset versions, hashes, leakage, diff
ui_common.py  run_panel.py                  rewards.py     controlled rule interpreter + TRL reward func
        │                                   model_loader.py base + LoRA loading, inference engine, lineage checks
        │ start run                         train.py       SFTTrainer / GRPOTrainer runs  (python -m lab.train <run>)
        ▼                                   jobs.py        background process launch, cancel, metrics/log reading
  background process ── metrics.jsonl ──▶   registry.py    SQLite: datasets, rewards, models, runs, evals, reviews
                         samples.jsonl      hardware.py    device detection, recommended mode
                         run log            status.py      pipeline status + next action
artifacts/  datasets/<role>/vN.jsonl · models/<id>/ (adapter, tokenizer, lineage.json) · runs/ · lab.sqlite

Apps layer (same lab models, no training needed):
  engines.py  one cached inference engine for non-UI code     lc.py       LabChatModel: any lab model as a LangChain chat model
  hub.py      Hugging Face import (size/fit check, register)  rag.py      chunking, BM25/embedding/hybrid retrieval, LCEL chain, IR metrics
  agent.py    LangGraph agent + sandbox tools                 agent_evals.py  tool / multi-turn / security scenarios
  judge.py    LLM-as-judge                                    api.py      FastAPI (OpenAI-compatible + /rag + /agent)
  services.py start/stop API process and Flowise container    course.py   Training Path modules + auto-checks
  registry.py also stores traces, agent_evals, course progress

  Streamlit ──▶ lab.* ◀── FastAPI :8811 ◀── LangChain ChatOpenAI / curl / Flowise (Docker :3811 via host.docker.internal)
```

### Workflow and lineage (the technically important part)

- Every trained version is a **LoRA adapter on the same base model**.
- A new stage loads its parent's adapter **with trainable weights and keeps training it**, then saves a new
  adapter (`PeftModel.from_pretrained(base, parent, is_trainable=True)`). So `sft-002 ← rl-001` really starts from RL v1.
  No merging is used, so there are no multi-GB merged copies.
- Base model + one adapter folder reproduces any version. An adapter is never treated as a standalone model.
- If an adapter is missing, or was trained on a different base, loading fails with an actionable error. The lab
  never silently falls back to the base model.
- After each run, the lab records greedy answers to two validation questions. "Verify reload" in the registry
  regenerates them from disk (tested: identical).
- You can branch explicitly from any earlier version by choosing it as the starting model.

### Configuration and where things are defined

| What | Where |
|---|---|
| Presets (Tiny / Small) for SFT and RL | `lab/jobs.py` (`SFT_PRESETS`, `RL_PRESETS`) |
| Default reward rules, test bench, rule types | `lab/rewards.py` |
| Sample datasets (synthetic tool store) | `datasets/sample_tool_store/*.jsonl` |
| 10 SFT packs (IT helpdesk, HR, billing, food safety, bank fraud, clinic front desk, logistics, travel, bug triage, insurance) — 12 train + 4 val each, synthetic | `datasets/sft_packs/*.json` |
| Eval math (kappa, judge correction, sampling, gate) | `lab/evals.py`; CI gate CLI `scripts/eval_gate.py` |
| Course tracks, modules, tasks and auto-checks | `lab/course.py` |
| Sidebar layouts V1 / V2 | `app.py` (`V1`, `V2`) |
| Notebook per page, JupyterLab launcher (port 8812) | `lab/notebooks.py`, generator `scripts/make_notebooks.py` |
| Flowise RAG chatflow builder | `lab/flowise_flows.py`; RAG-as-a-model in `lab/api.py` (`rag:<kb>:<model>`) |
| Langfuse stack + trace export | `docker/langfuse/docker-compose.yml` (secrets in git-ignored `.env`), `lab/langfuse_stack.py` |
| LangGraph Studio graph | `langgraph.json`, `lab/studio_graph.py` (model: newest trained, or `LAB_STUDIO_MODEL`) |
| Preference sample (24 pairs) | `datasets/sample_tool_store/pref.jsonl` |
| Reward-model training / scoring | `lab/train.py` (`run_rm`), `lab/reward_model.py`; presets `lab/jobs.py` (`RM_PRESETS`) |
| RLHF reward function (rules / reward model / both) | `lab/train.py` (`run_grpo`, config `reward_source`, `rm_id`) |
| Suggested Hugging Face models | `lab/hub.py` (`SUGGESTED`) |
| Sample knowledge base (8 policy docs) + 20 labelled retrieval questions | `datasets/kb_tool_store/` |
| Embedding model for RAG | `lab/rag.py` (`EMBED_MODEL` = `BAAI/bge-small-en-v1.5`) |
| Agent tools, system prompt | `lab/agent.py` |
| Agent eval scenarios (bundled / your own) | `datasets/agent_scenarios.json` / `artifacts/agent_scenarios_user.json` |
| LLM-judge prompt and default criterion | `lab/judge.py` |
| API port / Flowise port / Flowise image | `LAB_API_PORT` (8811) / `LAB_FLOWISE_PORT` (3811) / `LAB_FLOWISE_IMAGE` (`flowiseai/flowise:3.1.2`) |
| System prompt and chat formatting | `lab/model_loader.py` |
| Base model | `LAB_BASE_MODEL` env var (default `Qwen/Qwen3-0.6B`) |
| Artifacts folder | `LAB_HOME` env var (default `./artifacts`) |
| Port | `PORT` env var (default 8810) |

### Tested environment

WSL Ubuntu on Windows 11, Python 3.12, NVIDIA GTX 1080 Ti (11 GB, sm_61), 15 GB RAM.
`torch 2.14.1+cu126`, `transformers 5.23.0`, `trl 1.18.1`, `peft 0.21.3`, `datasets 5.2.0`, `streamlit 1.67.0`
(full pins in `requirements.txt`). Measured on that machine: Tiny SFT (12 steps) ≈ 11 s, Tiny GRPO (8 steps × 4
generations) ≈ 35–42 s, peak GPU memory ≈ 2.5 GB.

### Limitations (honest list)

- **GTX 10xx GPUs need the CUDA 12.6 PyTorch build.** cu128+ builds dropped sm_61; `scripts/setup_env.sh` installs cu126.
- Trains in fp32 on GPUs without fast bf16 and on CPU. No quantization (hardware-dependent and not needed at 0.6B).
- **Verified only on the GPU above.** CPU and macOS (MPS) code paths exist but were **not verified**. Expect CPU
  GRPO to be very slow.
- Keyword/pattern rewards are deliberately simple and can be fooled (negation, keyword stuffing). That is part of the lesson.
- The sample data is tiny and synthetic. Metrics on it say little about real-world quality.
- One training run at a time. The dashboard's inference model is unloaded automatically before training starts.
- Rule suggestions are frequent-phrase heuristics, labelled unreviewed. No LLM is used to write rules.
- **RLHF is small and teaches its failure modes on purpose.** The 24 sample preference pairs are easy (the reward model
  reaches 100% held-out accuracy on 5 pairs — far too few to trust). In the build-session run, the RLHF model learned
  that refusals ("I don't have the capability to assist…") score high, because every human-chosen answer is
  cautious: reward hacking of a learned reward model. Fix it with more varied preference pairs or "rules + reward model".
- Reward models start from the base model weights with a new score head (not from your SFT adapter).
- **Not verified end to end in the build session** (starting Docker containers was blocked for the assistant; the
  buttons run the same commands for you): the Flowise container, the "Show in Flowise" chatflow creation, and the
  Langfuse stack. Their Python sides are unit-tested (chatflow wiring, RAG-as-a-model endpoint, Langfuse event
  format). JupyterLab and the LangGraph dev server were started and checked live.
- JupyterLab runs without a token but only on 127.0.0.1. LangGraph Studio's UI needs a free LangSmith login.
- The 10 SFT packs, the policy documents and the agent tools are synthetic, written for teaching; agent tools
  return deterministic fake data (sandbox), not real systems.
- Small models are weak agents and judges: the 0.6B model sometimes skips tools or invents details, and its LLM-judge
  verdicts are unreliable. Import Qwen2.5-1.5B-Instruct on Model Hub for noticeably better results.
- The API process loads its own copy of the model; on an 11 GB GPU, don't train while the API serves a 1.5B model.
  The API has **no authentication** and binds to 0.0.0.0 (so the Flowise container can reach it) — other machines on
  your network can call it while it runs. Stop it when not in use, or don't run it on untrusted networks.
- **Flowise** runs as its own Docker container (`aitrain-flowise`, port 3811). Installing Flowise with npm fails on
  Node 24 (`better-sqlite3` does not compile), so npm is not used. The container start and the chatflow exercise were
  **not verified end to end** in the build session; the API side they call was verified with LangChain.
- Multi-turn agent sessions in the API are kept in memory (lost on restart). No streaming of partial tokens (the
  stream returns one chunk).

### How it was built

Built with Claude Code from a long written spec, treating the spec as a menu and building the working core first:
hardware check → data versioning → base inference → real SFT → evaluation/review → reward designer → real GRPO →
compare/registry → SFT→RL→SFT continuation → docs. The TRL 1.18 APIs were checked against the installed source
before writing the trainers (prompt-completion SFT, GRPO reward kwargs). Verification included 16 pytest tests (one of
them a real tiny SFT → RL → SFT chain with an exact-match reload check) and a headless-browser walkthrough of two
full cycles through the UI. A later additive pass added the 10 SFT packs, the one-example SFT editor and the Evals
Workbench (scoped from an AI-evals course outline: only the chapters that fit a single-turn lab), with 6 more pytest
tests (22 total) and a headless Streamlit AppTest check of both pages. A third additive pass turned the lab into a
course (Training Path) and added the apps layer: Model Hub, RAG Lab (LangChain), Agent Lab (LangGraph), LLM judge,
FastAPI server, Flowise launcher and Traces. The installed LangGraph 1.2 / LangChain-core 1.8 / Qwen tool-call
template were checked before coding. Verification: 29 pytest tests (model stubbed for app logic), every page rendered
headlessly, and a real-model smoke run on the GTX 1080 Ti (embeddings retrieval, RAG answer, 4 agent scenarios,
LLM judge, uvicorn API called from LangChain `ChatOpenAI` and `/agent/run`). A fourth additive pass made RLHF the third
workflow: PREF dataset, Human Feedback page, reward model (TRL 1.18 `RewardTrainer` API checked in the installed
source), GRPO with the reward model, course tracks A/B/C/D. Verified with 34 pytest tests and real GPU runs:
reward model 39 s, RLHF GRPO 89 s, rules + reward model 60 s (Tiny presets). A fifth additive pass added the V2
sidebar, in-page course steps, 17 generated notebooks (all executed top to bottom against a sandbox lab with nbclient,
including real SFT, RL, reward-model and RLHF training), Flowise RAG chatflow, Langfuse, LangGraph Studio and the
FastAPI source view — 39 pytest tests.

### Business scale-up plan *(draft)*

1. Workshop kit: packaged lab + exercise book for bootcamps and corporate training.
2. Bring-your-own-domain templates (support, retail, compliance) with starter rules and datasets.
3. Team edition: shared registry and review queue on a single GPU server (still self-hosted).
4. Optional managed GPU backend for larger models, keeping the same UI and lineage model.

## 5. WHAT NOW

### Requirements
- Linux, or Windows with WSL Ubuntu (macOS: untested). Python 3.10+, ~6 GB free disk.
- NVIDIA GPU recommended (≥ 4 GB free). CPU works for the Tiny preset but is slow.
- Internet once, for package and model download (~1.5 GB model).

### Clone and run
```bash
git clone <this repo> AitrainBootcamp && cd AitrainBootcamp
./start.sh            # first run: creates .venv (torch cu126 + pinned libs) and downloads Qwen3-0.6B
# open http://localhost:8810
./start.sh --stop     # stop the dashboard
```
Tests: `.venv/bin/python -m pytest -q` (the real-training test is skipped if the model is not downloaded).
Command-line smoke test of a whole cycle: `LAB_HOME=/tmp/labsmoke .venv/bin/python scripts/smoke_cycle.py`.

Optional: Docker (for Flowise). Serve the API without the dashboard: `.venv/bin/uvicorn lab.api:app --port 8811`.

**Easiest start: open 🎓 Training Path** and follow the "Next" pointer — it covers everything below and the apps layer
(Model Hub → API → LangChain → RAG → agents → Flowise → monitoring).

### First things to try (the guided first run on Overview)
1. Settings & Hardware: check the recommended mode.
2. Data Studio: read the sample data; look at the leakage check.
3. SFT Training: run **Tiny** from `base-001`; watch training vs validation loss.
4. Evaluation: run `sft-001` on `sft_val-v1` and rate the answers yourself.
5. Reward Designer: run the test bench; notice the fooled cases; approve `reward-v1`.
6. RL Training: run **Tiny** GRPO from `sft-001`; inspect sampled answers and rule contributions.
7. Compare Models: base vs SFT vs RL on the test questions.
8. Human Review: rate RL answers, write a correction, create `sft_train-v2` (the test question gets retired), add a replacement question.
9. SFT Training again, starting from `rl-001` with `sft_train-v2`, then RL again. Check the lineage graph in Model Registry.
10. RLHF: Human Feedback → pick the better answer for 10+ pairs and save → Reward Model → train (Tiny) and read the
    held-out accuracy → RL / RLHF Training with reward source "Reward model" → Compare Models: SFT vs RL vs RLHF.
10. Try another domain: Data Studio → SFT TRAIN → 📦 10 SFT packs → load e.g. *Bank fraud*, train SFT, evaluate on the new `sft_val`.
11. Evals Workbench: review 20 answers in ①, read the top failure in ②, check ④ before trusting the rule score, then ⑥ gate it.

### The ask for investors *(placeholder)*
We are looking for **[amount TBD]** to turn the lab into a workshop product and team edition: curriculum
development, domain templates, and testing on more hardware (Apple Silicon, recent NVIDIA, CPU-only laptops).
No figures here have been validated yet.

## 6. WHAT NEXT

| Now (this repo) | Next | Later |
|---|---|---|
| Real SFT + GRPO with LoRA on Qwen3-0.6B, versioned data/rewards/models, lineage, human review loop, comparison, 10 SFT packs + one-example editor, Evals Workbench (error analysis, agreement, judge validation, CI gate), Training Path course with A/B/C tracks, RLHF (preferences → reward model → GRPO), Model Hub, RAG Lab, LangGraph Agent Lab, LLM judge, FastAPI, Flowise launcher, Traces | Fine-tune the agent on its own successful traces (tool-use SFT); RAG reranker (cached cross-encoder) and RAG-specific judges; Flowise chatflow template import; API keys; verified CPU/macOS paths; preference data (DPO) from human reviews; held-out benchmark sets; per-rule reward charts during training | Multi-user review queue; larger models with QLoRA where hardware allows; curriculum and certification for bootcamps; plug-in reward types with sandboxing |

### Build it with us
- **Talent needed** *(draft)*: ML engineers (post-training, evaluation), educators/curriculum designers,
  front-end engineers for richer visualisations, testers with varied laptops (Mac, AMD, CPU-only).
- **Contact**: [add contact link] · [add repository / issues link]
