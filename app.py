"""Adaptive RAG Studio: side-by-side demo of adaptive vs. fixed-baseline RAG.

Layout: a query bar on top, the two answers next to each other, a metric table
that compares them row by row, and tabbed detail (signals, passages, routing
matrix, method) underneath.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

from src.cascade import (
    CascadedAdaptivePipeline,
    CascadedConfig,
    CascadedResult,  # noqa: F401  (kept for type reference)
)
from src.chunking import chunk_documents  # noqa: F401
from src.config import DEFAULT_EMBEDDING_MODELS, CandidateConfig, ExperimentConfig  # noqa: F401
from src.data import create_corpus  # noqa: F401
from src.embeddings import SentenceEmbeddingRegistry
from src.evaluation import evaluate_answer, normalize_answer
from src.generation import Phi3Generator
from src.retrieval import FaissIndexManager
from src.types import Document, RAGExample  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_ROOT / "cache" / "dataset" / "corpus.json"

SAMPLE_QUERIES = [
    ("P=NP prize", "What is the prize offered for finding a solution to P=NP?"),
    ("Super Bowl 50 uniforms", "What color were the Bronco's uniforms in Super Bowl 50?"),
    ("Newtonian gravitation", "How might gravity effects be observed differently according to Newton?"),
    ("PPO release date", "When was PPO developed?"),
    ("Out-of-corpus: 2+2", "2+2 ="),
]

# ─────────────────────────────── Styling ───────────────────────────────────
#  ink #0f1a2b   muted #596779   line #e3e7ed   accent #0e5a63 (adaptive)
#  slate #6b7788 (fixed baseline)   good #1f7a4d   bad #a33b32
CSS = """
:root {
    --bg: #fbfbfc;
    --surface: #ffffff;
    --ink: #0f1a2b;
    --muted: #596779;
    --faint: #8794a5;
    --line: #e3e7ed;
    --line-soft: #f0f2f6;
    --accent: #0e5a63;
    --accent-soft: #e7f1f2;
    --slate: #6b7788;
    --good: #1f7a4d;
    --bad: #a33b32;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 32px 32px !important;
    font-family: "IBM Plex Sans", -apple-system, "Segoe UI", Roboto, sans-serif !important;
    color: var(--ink) !important;
    background: var(--bg) !important;
}
.gradio-container .block, .gradio-container .form { box-shadow: none !important; }
.gradio-container .prose, .gradio-container .md { color: var(--ink); }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* ── Top bar and hero ── */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    padding: 22px 0;
    border-bottom: 1px solid var(--line);
}
.brand { display: flex; align-items: baseline; }
.brand-name { font-family: var(--serif); font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; }
.brand-sub {
    margin-left: 14px; padding-left: 14px;
    border-left: 1px solid var(--line);
    font-size: 0.84rem; color: var(--muted);
}
.env { display: flex; gap: 24px; margin: 0; padding: 0; list-style: none; }
.env li { font-size: 0.78rem; color: var(--muted); }
.env b { font-family: var(--mono); font-weight: 500; color: var(--ink); margin-left: 6px; }
.env-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--good); margin-right: 6px; vertical-align: 1px; }

.hero { padding: 44px 0 26px; }
.hero h1 {
    font-family: var(--serif);
    font-size: 2.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.12;
    margin: 0 0 12px;
    max-width: 20ch;
}
.hero p { margin: 0; max-width: 60ch; font-size: 1.02rem; line-height: 1.6; color: var(--muted); }

/* ── Query bar ── */
.query-row { gap: 12px !important; align-items: stretch !important; }
#query-box { background: transparent !important; border: none !important; padding: 0 !important; }
#query-box textarea {
    font-family: inherit !important;
    font-size: 1.05rem !important;
    line-height: 1.5 !important;
    padding: 16px 18px !important;
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
}
#query-box textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(14, 90, 99, .14) !important;
}
.run-btn {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 0.96rem !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    min-height: 56px !important;
    transition: background .15s ease !important;
}
.run-btn:hover { background: #0a464d !important; border-color: #0a464d !important; }

.chip-row { gap: 8px !important; flex-wrap: wrap !important; margin: 12px 0 8px !important; }
.chip-label { font-size: 0.8rem; color: var(--muted); margin-right: 4px; align-self: center; }
.chip-btn {
    background: transparent !important;
    border: 1px solid var(--line) !important;
    border-radius: 999px !important;
    padding: 3px 13px !important;
    min-width: 0 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    box-shadow: none !important;
    transition: border-color .15s ease, background .15s ease !important;
}
.chip-btn:hover { border-color: var(--accent) !important; background: var(--accent-soft) !important; }

.source-acc { margin-top: 6px !important; }
.hint { font-size: 0.84rem; color: var(--muted); line-height: 1.55; margin: 0 0 8px; }

/* ── Results ── */
.empty-state {
    margin-top: 28px;
    padding: 44px 24px;
    text-align: center;
    color: var(--muted);
    font-size: 0.92rem;
    border: 1px dashed var(--line);
    border-radius: 12px;
}
.results { margin-top: 30px; }
.verdict {
    border-left: 3px solid var(--accent);
    padding: 2px 0 2px 16px;
    margin-bottom: 22px;
}
.verdict-label { font-size: 0.78rem; font-weight: 600; color: var(--accent); margin-bottom: 3px; }
.verdict-text { font-size: 1.02rem; line-height: 1.55; color: var(--ink); max-width: 80ch; }

.answers { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
@media (max-width: 820px) { .answers { grid-template-columns: 1fr; } }
.ans {
    background: var(--surface);
    border: 1px solid var(--line);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    padding: 18px 22px 20px;
    display: flex;
    flex-direction: column;
}
.ans.fixed { border-top-color: var(--slate); }
.ans-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
.ans-name { font-size: 0.92rem; font-weight: 600; }
.ans-cfg {
    font-family: var(--mono); font-size: 0.74rem; color: var(--muted);
    background: var(--line-soft); padding: 3px 9px; border-radius: 5px;
}
.ans-text {
    font-family: var(--serif);
    font-size: 1.2rem;
    line-height: 1.6;
    color: var(--ink);
    flex-grow: 1;
    min-height: 96px;
    margin: 0;
    overflow-wrap: anywhere;
}
.ans-meta { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line-soft); font-size: 0.8rem; color: var(--muted); }

/* ── Comparison table ── */
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
.cmp { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.cmp th {
    text-align: left; font-weight: 600; font-size: 0.8rem; color: var(--muted);
    padding: 12px 20px; background: var(--bg); border-bottom: 1px solid var(--line); white-space: nowrap;
}
.cmp td { padding: 13px 20px; border-bottom: 1px solid var(--line-soft); }
.cmp tr:last-child td { border-bottom: none; }
.cmp .num { text-align: right; font-family: var(--mono); font-size: 0.86rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.cmp th.num { text-align: right; font-family: inherit; }
.cmp .metric { font-weight: 500; }
.cmp .metric small { display: block; font-weight: 400; font-size: 0.76rem; color: var(--muted); margin-top: 1px; }
.cmp .better { color: var(--good); font-weight: 600; }
.cmp .worse { color: var(--bad); font-weight: 600; }
.cmp .same { color: var(--faint); }
.swatch { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 8px; }

/* ── Detail tabs ── */
.gradio-container .tabs { margin-top: 30px; }
.gradio-container [role="tab"] { font-weight: 500 !important; font-size: 0.9rem !important; color: var(--muted) !important; }
.gradio-container [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-color: var(--accent) !important; }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 4px 22px; }
.signal-row {
    display: grid;
    grid-template-columns: minmax(200px, 1.2fr) 2fr 64px;
    align-items: center;
    gap: 18px;
    padding: 15px 0;
    border-bottom: 1px solid var(--line-soft);
}
.signal-row:last-child { border-bottom: none; }
.signal-name { font-size: 0.9rem; font-weight: 600; }
.signal-desc { font-size: 0.78rem; color: var(--muted); margin-top: 2px; line-height: 1.45; }
.track { height: 6px; background: var(--line-soft); border-radius: 3px; overflow: hidden; }
.fill { height: 100%; border-radius: 3px; }
.signal-val { text-align: right; font-family: var(--mono); font-size: 0.86rem; font-variant-numeric: tabular-nums; }
.signal-row.total { background: var(--bg); margin: 0 -22px; padding: 16px 22px; border-top: 1px solid var(--line); border-radius: 0 0 12px 12px; }
.signal-row.total .track { height: 8px; }
.signal-row.total .signal-val { color: var(--accent); font-weight: 600; }
@media (max-width: 700px) {
    .signal-row { grid-template-columns: 1fr 56px; }
    .signal-row .track { grid-column: 1 / -1; order: 3; }
}
.note { font-size: 0.8rem; color: var(--muted); margin: 12px 2px 0; }

.passage {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.passage-meta { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; }
.passage-text { font-family: var(--serif); font-size: 1rem; line-height: 1.7; }

.matrix { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.matrix th {
    text-align: left; font-weight: 600; font-size: 0.78rem; color: var(--muted);
    padding: 12px 16px; background: var(--bg); border-bottom: 1px solid var(--line); white-space: nowrap;
}
.matrix td { padding: 11px 16px; border-bottom: 1px solid var(--line-soft); white-space: nowrap; }
.matrix tr:last-child td { border-bottom: none; }
.matrix .num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.matrix th.num { text-align: right; }
.matrix code { font-family: var(--mono); font-size: 0.82rem; background: none; padding: 0; color: inherit; }
.matrix tr.winner td { background: var(--accent-soft); font-weight: 600; }
.matrix tr.pruned td { color: var(--faint); }
.tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; }
.tag.winner { background: var(--accent); color: #fff; }
.tag.kept { background: var(--line-soft); color: var(--ink); }
.tag.pruned { color: var(--faint); border: 1px solid var(--line); font-weight: 500; }

.app-footer {
    margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--line);
    font-size: 0.78rem; color: var(--faint);
}
"""

FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"


# ─────────────────────────────── App state ─────────────────────────────────
class AppState:
    def __init__(self) -> None:
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = ExperimentConfig(device=self.device)
        self.candidates = self.config.candidates()
        self.embeddings = SentenceEmbeddingRegistry(self.config.device, 64)
        self.generator: Phi3Generator | None = None
        self.default_indexes: FaissIndexManager | None = None
        self.corpus_docs: list[Document] = []
        self._load_default_corpus()
        self.gold_answers: dict[str, list[str]] = self._load_gold_answers()
        # Warm generator during startup
        self.get_generator()

    def _load_gold_answers(self) -> dict[str, list[str]]:
        gold_map: dict[str, list[str]] = {}
        try:
            from src.data import load_squad_subset

            rag_examples, gold = load_squad_subset(500, 42, cache_dir=PROJECT_ROOT / "cache")
            for ex in rag_examples:
                if ex.query_id in gold:
                    gold_map[normalize_answer(ex.question)] = list(gold[ex.query_id].answers)
        except Exception:
            pass

        # Seed sample queries
        gold_map[normalize_answer("What is the prize offered for finding a solution to P=NP?")] = ["$1,000,000", "US$1,000,000"]
        gold_map[normalize_answer("What color were the Bronco's uniforms in Super Bowl 50?")] = ["white", "matching white jerseys", "white uniforms"]
        gold_map[normalize_answer("How might gravity effects be observed differently according to Newton?")] = ["at larger distances"]
        gold_map[normalize_answer("When was PPO developed?")] = ["2017", "July 2017"]
        gold_map[normalize_answer("2+2 =")] = ["4"]
        gold_map[normalize_answer("2+2")] = ["4"]
        return gold_map

    def _load_default_corpus(self) -> None:
        if CORPUS_PATH.exists():
            payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
            self.corpus_docs = [Document(**item) for item in payload]
        else:
            self.corpus_docs = [
                Document(
                    document_id="doc_p_np",
                    text=(
                        "The question of whether P equals NP is one of the most important open questions in theoretical computer science. "
                        "The P versus NP problem is one of the Millennium Prize Problems proposed by the Clay Mathematics Institute. "
                        "There is a US$1,000,000 prize for resolving the problem."
                    ),
                    title="Computational_complexity_theory",
                    source_example_ids=("q1",),
                ),
                Document(
                    document_id="doc_sb50",
                    text=(
                        "The Broncos elected to wear their road white jerseys with matching white pants in Super Bowl 50. "
                        "The Broncos' decision to wear white meant the Panthers would wear their standard home uniform: black jerseys with silver pants."
                    ),
                    title="Super_Bowl_50",
                    source_example_ids=("q2",),
                ),
            ]
        self.default_indexes = FaissIndexManager(
            self.corpus_docs,
            self.embeddings,
            self.config.chunk_overlap,
            PROJECT_ROOT / "cache",
        )
        self.default_indexes.build_all(self.candidates)

    def get_generator(self) -> Phi3Generator:
        if self.generator is None:
            self.generator = Phi3Generator(
                self.config.generator_model_id,
                self.config.device,
                max_new_tokens=self.config.max_new_tokens,
            )
        return self.generator


STATE = AppState()


# ─────────────────────────────── HTML builders ─────────────────────────────
def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: float) -> int:
    return min(100, max(0, round(value * 100)))


def empty_state(message: str) -> str:
    return f'<div class="empty-state">{esc(message)}</div>'


def build_header() -> str:
    device = STATE.device.upper()
    model = STATE.config.generator_model_id.split("/")[-1]
    return f"""
<div class="topbar">
  <div class="brand">
    <span class="brand-name">Adaptive RAG Studio</span>
    <span class="brand-sub">Reference-free retrieval tuning</span>
  </div>
  <ul class="env">
    <li><span class="env-dot"></span>Device<b>{esc(device)}</b></li>
    <li>Generator<b>{esc(model)}</b></li>
    <li>Configurations<b>{len(STATE.candidates)}</b></li>
  </ul>
</div>
<div class="hero">
  <h1>Adaptive retrieval, measured against a fixed baseline</h1>
  <p>Ask a question and both pipelines answer it. The adaptive router picks its retrieval setup per query using signals that need no reference answer, and the table below shows how each system did.</p>
</div>
"""


def _delta(a: float, b: float, fmt: str, higher_is_better: bool, unit: str = "") -> str:
    diff = a - b
    if abs(diff) < 1e-9:
        return '<td class="num same">same</td>'
    good = diff > 0 if higher_is_better else diff < 0
    cls = "better" if good else "worse"
    return f'<td class="num {cls}">{format(diff, fmt)}{unit}</td>'


def build_comparison(adaptive_res: Any, fixed_res: Any, question: str) -> str:
    refs = STATE.gold_answers.get(normalize_answer(question))

    a_r, f_r = adaptive_res.retrieval_confidence, fixed_res.retrieval_confidence
    a_c, f_c = adaptive_res.context_consistency, fixed_res.context_consistency
    a_t, f_t = adaptive_res.total_latency, fixed_res.total_latency
    a_n, f_n = adaptive_res.llm_generations_count, fixed_res.llm_generations_count

    # Verdict: report what actually happened, whichever system came out ahead.
    timing = f"Adaptive took {a_t:.2f} s and the baseline {f_t:.2f} s."
    a_f1 = f_f1 = None
    if refs:
        a_f1 = evaluate_answer(adaptive_res.answer, refs)["f1"] * 100
        f_f1 = evaluate_answer(fixed_res.answer, refs)["f1"] * 100
        if a_f1 > f_f1 + 0.05:
            verdict = f"Adaptive RAG matched the reference answer more closely, with token F1 of {a_f1:.1f}% against {f_f1:.1f}% for the fixed baseline."
        elif f_f1 > a_f1 + 0.05:
            verdict = f"The fixed baseline matched the reference answer more closely, with token F1 of {f_f1:.1f}% against {a_f1:.1f}% for adaptive RAG."
        else:
            verdict = f"Both systems scored the same token F1 against the reference answer ({a_f1:.1f}%)."
    else:
        verdict = (
            "No reference answer exists for this query, so the comparison uses reference-free signals. "
            f"Evidence grounding is {a_c:.3f} for adaptive RAG and {f_c:.3f} for the fixed baseline."
        )

    def answer_card(name: str, cls: str, res: Any, label: str) -> str:
        model = STATE.config.embedding_models.get(res.selected_embedding_key, res.selected_embedding_key)
        return f"""
<article class="ans {cls}">
  <div class="ans-head"><span class="ans-name">{esc(name)}</span><span class="ans-cfg">{esc(res.selected_candidate_id)}</span></div>
  <p class="ans-text">{esc(res.answer)}</p>
  <div class="ans-meta">{esc(model)}, chunk {res.selected_chunk_size}, top-{res.selected_top_k}. {esc(label)}</div>
</article>"""

    a_route = "Stage 1 early exit" if adaptive_res.is_early_exit else "Stage 2 evaluated"
    cards = (
        answer_card("Adaptive RAG", "adaptive", adaptive_res, a_route)
        + answer_card("Fixed baseline", "fixed", fixed_res, "Static configuration")
    )

    rows = [
        (
            '<td class="metric">Configuration</td>'
            f'<td class="num">{esc(adaptive_res.selected_candidate_id)}</td>'
            f'<td class="num">{esc(fixed_res.selected_candidate_id)}</td>'
            '<td class="num same">-</td>'
        ),
        (
            '<td class="metric">Retrieval confidence (R)<small>Similarity of retrieved passages to the question</small></td>'
            f'<td class="num">{a_r:.3f}</td><td class="num">{f_r:.3f}</td>' + _delta(a_r, f_r, "+.3f", True)
        ),
        (
            '<td class="metric">Evidence grounding (C)<small>Share of the answer supported by retrieved text</small></td>'
            f'<td class="num">{a_c:.3f}</td><td class="num">{f_c:.3f}</td>' + _delta(a_c, f_c, "+.3f", True)
        ),
    ]
    if a_f1 is not None and f_f1 is not None:
        rows.append(
            '<td class="metric">Token F1 vs. reference<small>Overlap with the reference answer</small></td>'
            f'<td class="num">{a_f1:.1f}%</td><td class="num">{f_f1:.1f}%</td>' + _delta(a_f1, f_f1, "+.1f", True, " pts")
        )
    rows.append(
        '<td class="metric">Latency</td>'
        f'<td class="num">{a_t:.2f} s</td><td class="num">{f_t:.2f} s</td>' + _delta(a_t, f_t, "+.2f", False, " s")
    )
    rows.append(
        '<td class="metric">Generator calls</td>'
        f'<td class="num">{a_n}</td><td class="num">{f_n}</td>' + _delta(float(a_n), float(f_n), "+.0f", False)
    )

    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f"""
<section class="results">
  <div class="verdict">
    <div class="verdict-label">Result</div>
    <div class="verdict-text">{esc(verdict)} {esc(timing)}</div>
  </div>
  <div class="answers">{cards}</div>
  <div class="table-wrap"><table class="cmp">
    <thead><tr>
      <th>Metric</th>
      <th class="num"><span class="swatch" style="background:var(--accent)"></span>Adaptive RAG</th>
      <th class="num"><span class="swatch" style="background:var(--slate)"></span>Fixed baseline</th>
      <th class="num">Difference</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table></div>
</section>
"""


def build_signals(adaptive_res: Any) -> str:
    rows = [
        ("Retrieval confidence (R)", "Mean cosine similarity of the retrieved passages.", adaptive_res.retrieval_confidence, "#0e5a63"),
        ("Semantic confidence (1 - U)", "Agreement across stochastic generations. High agreement means low model uncertainty.", adaptive_res.semantic_confidence, "#1f7a4d"),
        ("Evidence grounding (C)", "Share of answer tokens contained in the retrieved evidence.", adaptive_res.context_consistency, "#4a5fa8"),
    ]
    out = ['<div class="card">']
    for name, desc, val, color in rows:
        out.append(
            f'<div class="signal-row">'
            f'<div><div class="signal-name">{esc(name)}</div><div class="signal-desc">{esc(desc)}</div></div>'
            f'<div class="track"><div class="fill" style="width:{pct(val)}%;background:{color}"></div></div>'
            f'<div class="signal-val">{val:.4f}</div></div>'
        )
    j = adaptive_res.combined_score
    out.append(
        f'<div class="signal-row total">'
        f'<div><div class="signal-name">Composite objective (J)</div>'
        f'<div class="signal-desc">J = (R + (1 - U) + C) / 3. The router picks the configuration with the highest J.</div></div>'
        f'<div class="track"><div class="fill" style="width:{pct(j)}%;background:#0f1a2b"></div></div>'
        f'<div class="signal-val">{j:.4f}</div></div></div>'
        '<p class="note">Signals are computed for the adaptive pipeline only. The fixed baseline uses one static configuration and does not optimize J.</p>'
    )
    return "".join(out)


def build_evidence(chunks: list[str]) -> str:
    if not chunks:
        return empty_state("No passages were retrieved for this query.")
    cards = []
    for i, chunk_text in enumerate(chunks, 1):
        clean = " ".join(chunk_text.split())
        cards.append(
            f'<div class="passage">'
            f'<div class="passage-meta"><span>Passage {i} of {len(chunks)}</span>'
            f"<span>{len(clean.split())} words</span></div>"
            f'<div class="passage-text">{esc(clean)}</div></div>'
        )
    return "".join(cards)


def build_matrix(candidates: list[Any]) -> str:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "-"

    rows = []
    for c in candidates:
        if c.is_winner:
            if c.semantic_uncertainty is not None and c.semantic_uncertainty > 0:
                tag = '<span class="tag winner">Selected (Stage 2 Winner)</span>'
            else:
                tag = '<span class="tag winner">Selected (Early Exit)</span>'
            cls = "winner"
        elif c.retained_for_stage_2:
            cls, tag = "kept", '<span class="tag kept">Evaluated in stage 2 (Runner-up)</span>'
        else:
            cls, tag = "pruned", '<span class="tag pruned">Pruned in stage 1</span>'
        rows.append(
            f'<tr class="{cls}">'
            f"<td><code>{esc(c.candidate_id)}</code></td>"
            f"<td>{esc(c.embedding_key)}</td>"
            f'<td class="num">{c.chunk_size}</td>'
            f'<td class="num">{c.top_k}</td>'
            f'<td class="num">{c.retrieval_confidence:.3f}</td>'
            f'<td class="num">{fmt(c.semantic_uncertainty)}</td>'
            f'<td class="num">{fmt(c.context_consistency)}</td>'
            f'<td class="num">{fmt(c.combined_score)}</td>'
            f"<td>{tag}</td></tr>"
        )
    return (
        '<p class="hint">Every configuration the adaptive router considered. '
        "R is retrieval confidence, U is uncertainty, C is grounding and J is the composite score.</p>"
        '<div class="table-wrap"><table class="matrix"><thead><tr>'
        "<th>Configuration</th><th>Embedding</th>"
        '<th class="num">Chunk</th><th class="num">Top-K</th>'
        '<th class="num">R</th><th class="num">U</th><th class="num">C</th><th class="num">J</th>'
        "<th>Outcome</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


# ─────────────────────────────── Pipeline ──────────────────────────────────
def extract_text_from_file(file_obj: Any) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        path = Path(file_obj)
    elif isinstance(file_obj, dict) and "path" in file_obj:
        path = Path(file_obj["path"])
    elif isinstance(file_obj, list) and file_obj:
        return extract_text_from_file(file_obj[0])
    elif hasattr(file_obj, "name"):
        path = Path(file_obj.name)
    elif hasattr(file_obj, "orig_name"):
        path = Path(file_obj.orig_name)
    else:
        path = Path(str(file_obj))

    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            pages = [t.strip() for t in (p.extract_text() for p in reader.pages) if t and t.strip()]
            extracted = "\n".join(pages).strip()
            print(f"[Upload] Extracted {len(pages)} pages ({len(extracted.split())} words) from PDF: {path.name}")
            return extracted
        except Exception as e:
            print(f"[Upload] Error reading PDF: {e}")
            return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        print(f"[Upload] Read text file ({len(txt.split())} words): {path.name}")
        return txt
    except Exception as e:
        print(f"[Upload] Error reading text file: {e}")
        return ""


def run_pipeline(
    question: str,
    custom_doc_text: str,
    uploaded_file: Any,
) -> tuple[str, str, str, str]:
    if not question or not question.strip():
        return (
            empty_state("Enter a question above to run both pipelines."),
            empty_state("Objective signals appear here after you run a query."),
            empty_state("Retrieved passages appear here after you run a query."),
            empty_state("The routing matrix appears here after you run a query."),
        )

    custom_content = (custom_doc_text or "").strip()
    if not custom_content and uploaded_file is not None:
        custom_content = extract_text_from_file(uploaded_file)

    generator = STATE.get_generator()

    if custom_content:
        doc_hash = hashlib.sha256(custom_content.encode("utf-8")).hexdigest()[:16]
        if hasattr(STATE, "custom_indexes_cache") and doc_hash in STATE.custom_indexes_cache:
            active_indexes = STATE.custom_indexes_cache[doc_hash]
        else:
            custom_doc = Document(
                document_id=f"doc_{doc_hash}",
                text=custom_content,
                title="Custom_Upload",
                source_example_ids=("custom_1",),
            )
            custom_indexes = FaissIndexManager(
                [custom_doc],
                STATE.embeddings,
                STATE.config.chunk_overlap,
                PROJECT_ROOT / "cache" / "custom_temp",
            )
            custom_indexes.build_all(STATE.candidates)
            if not hasattr(STATE, "custom_indexes_cache"):
                STATE.custom_indexes_cache = {}
            STATE.custom_indexes_cache[doc_hash] = custom_indexes
            active_indexes = custom_indexes
    else:
        active_indexes = STATE.default_indexes

    pipeline = CascadedAdaptivePipeline(
        indexes=active_indexes,
        generator=generator,
        embeddings=STATE.embeddings,
        candidates=STATE.candidates,
        cascade_config=CascadedConfig(),
    )

    fixed_res = pipeline.run_query(question, mode="fixed")
    adaptive_res = pipeline.run_query(question, mode="cascaded")

    return (
        build_comparison(adaptive_res, fixed_res, question),
        build_signals(adaptive_res),
        build_evidence(adaptive_res.retrieved_chunks),
        build_matrix(adaptive_res.all_candidates),
    )


# ─────────────────────────────── UI ────────────────────────────────────────
METHOD_MD = r"""
#### The problem
A standard RAG system commits to one fixed setup: one embedding model, one chunk size, one retrieval depth. Different questions need different setups, and tuning against reference answers (BLEU, ROUGE) is impossible at inference time because no reference exists.

#### Reference-free objective
Three signals estimate answer quality without a reference:

1. **Retrieval confidence (R):** mean cosine similarity of the top-K chunks in the FAISS index.
2. **Semantic uncertainty (U):** how much answers diverge across stochastic generations, measured as semantic entropy in embedding space.
3. **Evidence grounding (C):** how much of the generated answer is contained in the retrieved context.

They combine into one composite objective score:

$$J = \frac{R + (1-U) + C}{3}$$

#### Two-stage cascade with early exit
- **Stage 1, FAISS filtering.** All configurations are queried against the vector indexes. If one shows overwhelming retrieval confidence, or the query is clearly outside the corpus, the router exits early with a single generation call.
- **Stage 2, generation and selection.** Otherwise the leading configuration from each retriever family is evaluated, and the one with the highest J is selected.
"""


def create_demo() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "-apple-system", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#fbfbfc",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e3e7ed",
        block_shadow="none",
        block_radius="10px",
        input_radius="8px",
        button_large_radius="10px",
    )

    head = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&display=swap" rel="stylesheet">'
    )

    with gr.Blocks(
        title="Adaptive RAG Studio",
        theme=theme,
        css=CSS,
        head=head,
        js=FORCE_LIGHT_JS,
    ) as demo:
        gr.HTML(build_header())

        # ── Query bar ──
        with gr.Row(elem_classes=["query-row"], equal_height=True):
            question_box = gr.Textbox(
                placeholder="Ask a question, for example: What is the prize offered for solving P=NP?",
                lines=2,
                show_label=False,
                container=False,
                scale=6,
                elem_id="query-box",
            )
            submit_btn = gr.Button(
                "Run comparison",
                variant="primary",
                size="lg",
                scale=1,
                min_width=180,
                elem_classes=["run-btn"],
            )

        with gr.Row(elem_classes=["chip-row"]):
            gr.HTML('<span class="chip-label">Try an example</span>')
            chips = [
                (gr.Button(label, size="sm", elem_classes=["chip-btn"]), text)
                for label, text in SAMPLE_QUERIES
            ]
        for chip, text in chips:
            chip.click(fn=lambda t=text: t, outputs=[question_box])

        with gr.Accordion("Search your own document instead of the default corpus", open=False, elem_classes=["source-acc"]):
            gr.HTML(
                '<p class="hint">By default the SQuAD corpus is searched. Upload a PDF or text file, or paste text, '
                "and both pipelines will answer from that document instead.</p>"
            )
            with gr.Row():
                file_input = gr.File(label="Upload PDF, TXT or Markdown", file_types=[".pdf", ".txt", ".md"])
                custom_text_input = gr.Textbox(
                    label="Or paste document text",
                    placeholder="Paste document text here...",
                    lines=6,
                )

        # ── Results ──
        comparison_output = gr.HTML(
            value=empty_state("Pick an example or type a question, then run the comparison.")
        )

        with gr.Tabs():
            with gr.TabItem("Objective signals"):
                signals_output = gr.HTML(value=empty_state("Objective signals appear here after you run a query."))
            with gr.TabItem("Retrieved passages"):
                evidence_output = gr.HTML(value=empty_state("Retrieved passages appear here after you run a query."))
            with gr.TabItem("Routing matrix"):
                matrix_output = gr.HTML(value=empty_state("The routing matrix appears here after you run a query."))
            with gr.TabItem("Method"):
                gr.Markdown(
                    METHOD_MD,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )

        gr.HTML('<div class="app-footer">Adaptive RAG Studio. Reference-free adaptive retrieval compared with a fixed baseline.</div>')

        run_inputs = [question_box, custom_text_input, file_input]
        run_outputs = [comparison_output, signals_output, evidence_output, matrix_output]
        submit_btn.click(fn=run_pipeline, inputs=run_inputs, outputs=run_outputs)
        question_box.submit(fn=run_pipeline, inputs=run_inputs, outputs=run_outputs)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    print("=" * 60)
    print("  Adaptive RAG Studio")
    print("  Local URL: http://127.0.0.1:7860")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )