"""
AI Agents Inference Benchmarking System — Streamlit dashboard.

Run from the "Data Analysis with SQL Queries" folder with:
    streamlit run app/streamlit_app.py

Three tabs, one per project objective:
  1. Standardized Pipeline — a landing view: the pipeline shape and what's
     held identical between the two frameworks.
  2. Live Execution — watch, step by step, how CrewAI (one looping agent)
     and LangGraph (an explicit assistant/tools graph) actually run a
     question, in real time, from real execution events.
  3. Run Your Own Question — type any question, get it answered by both
     frameworks, and see accuracy, consistency, latency, and LLM calls.
All benchmarking logic lives in the real project files (the CrewAI/LangGraph
scripts, utils/metrics_collector.py) and in app/{framework_loader,runner,
config_validate}.py, which call those real files rather than reimplementing
them.
"""
import os
import sys
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from app import config_validate, known_questions, runner
from app.framework_loader import dataset_status, DB_PATH, QUESTIONS_PATH, EXPECTED_RESULTS_PATH

st.set_page_config(page_title="CrewAI vs LangGraph", page_icon="◆", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    _p = dict(
        accent="#6FA8E8", accent_soft="#16314F", text_main="#E7EFFA", text_sub="#9BB0C9",
        border_c="#28415F", good="#52C98A", bad="#E8746F",
        page_bg="linear-gradient(160deg, #0B1626 0%, #101F36 50%, #0B1A2E 100%)",
        glass_bg="rgba(19,35,59,0.55)", glass_bg2="rgba(19,35,59,0.4)", glass_border="rgba(255,255,255,0.08)",
        shadow="0 8px 28px rgba(0,0,0,0.35)",
    )
else:
    _p = dict(
        accent="#4A90D9", accent_soft="#EAF4FC", text_main="#17293D", text_sub="#5A7184",
        border_c="#D8E8F6", good="#2E9E6B", bad="#D9534F",
        page_bg="linear-gradient(160deg, #F3F8FD 0%, #EAF4FC 45%, #F7FBFE 100%)",
        glass_bg="rgba(255,255,255,0.55)", glass_bg2="rgba(255,255,255,0.4)", glass_border="rgba(255,255,255,0.65)",
        shadow="0 8px 28px rgba(74,144,217,0.12)",
    )

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
  --accent: {_p['accent']}; --accent-soft: {_p['accent_soft']}; --text-main: {_p['text_main']};
  --text-sub: {_p['text_sub']}; --border-c: {_p['border_c']}; --good: {_p['good']}; --bad: {_p['bad']};
  --glass: {_p['glass_bg']}; --glass2: {_p['glass_bg2']}; --gborder: {_p['glass_border']}; --shadow: {_p['shadow']};
}}
html, body, [class*="css"] {{font-family: 'Inter', sans-serif !important;}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"] {{
  background: {_p['page_bg']} !important;
}}
.block-container {{padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; padding-left: 2.2rem; padding-right: 2.2rem;}}
[data-testid="stVerticalBlock"] {{gap: 0.6rem;}}
.card, .result-card, .verdict-banner {{width: 100%;}}
h1, h2, h3, p, span, label, .stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{color: var(--text-main) !important;}}
.eyebrow {{color: var(--accent) !important; font-size: 12.5px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 2px;}}
h1 {{font-weight: 800 !important; letter-spacing: -0.02em;}}
.app-sub {{color: var(--text-sub) !important; font-size: 15.5px; margin-top: -4px; margin-bottom: 1.2rem;}}
.section-head {{font-size: 20px; font-weight: 700; color: var(--text-main) !important; margin: 10px 0 10px; display:flex; align-items:center; gap:8px;}}
.section-head .dot {{width:8px; height:8px; border-radius:50%; background: var(--accent); display:inline-block;}}
.card, .result-card, .verdict-banner {{
  background: var(--glass) !important; border: 1px solid var(--gborder); border-radius: 16px;
  padding: 20px 22px; margin: 12px 0; box-shadow: var(--shadow);
  backdrop-filter: blur(16px) saturate(160%); -webkit-backdrop-filter: blur(16px) saturate(160%);
}}
.fw-label {{font-size: 14px; font-weight: 700; color: var(--accent) !important; letter-spacing: .02em; margin-bottom: 10px; display:flex; align-items:center; gap:8px;}}
.flow-row {{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom: 8px;}}
.flow-tools {{display:flex; gap:10px; margin: 12px 0 0 64px; flex-wrap:wrap;}}
.node {{padding:9px 13px; border-radius:10px; background: var(--glass2) !important; border:1.5px solid var(--gborder);
       font-size:12.5px; font-weight:600; color: var(--text-sub) !important; text-align:center; transition: all .25s;
       backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);}}
.node.active {{border-color: var(--accent); background: var(--accent-soft) !important; color: var(--text-main) !important; box-shadow: 0 0 0 3px {_p['accent_soft']};}}
.node.done {{color: var(--text-main) !important; border-color: var(--good);}}
.agent-node {{padding:13px 18px; border-radius:14px; background: var(--glass2) !important; border:2px solid var(--gborder);
             font-size:13.5px; font-weight:700; color: var(--text-sub) !important; text-align:center;
             backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);}}
.agent-node.active {{border-color: var(--accent); background: var(--accent-soft) !important; color: var(--text-main) !important;}}
.agent-node.done {{border-color: var(--good); color: var(--text-main) !important;}}
.node-sub {{font-size:10.5px; font-weight:400; color: var(--text-sub) !important; display:block; margin-top:2px;}}
.node-count {{font-size:10.5px; margin-left:3px; color: var(--accent) !important; font-weight:700;}}
.flow-arrow {{color: var(--border-c); font-size:17px;}}
.flow-arrow-loop {{color: var(--accent) !important; font-size:16px; font-weight:800;}}
.flow-caption {{font-size:11.5px; color: var(--text-sub) !important; margin-top: 8px;}}
.timeline {{margin-top: 14px;}}
.tl-row {{display:flex; align-items:center; gap:10px; padding:7px 0; font-size: 13px; color: var(--text-sub) !important;
          border-bottom: 1px solid var(--gborder);}}
.tl-row:last-child {{border-bottom: none;}}
.tl-row.latest {{color: var(--text-main) !important; font-weight: 600;}}
.tl-num {{width:20px; height:20px; border-radius:50%; background: var(--glass2); border:1px solid var(--gborder);
          display:flex; align-items:center; justify-content:center; font-size:10.5px; font-weight:700;
          color: var(--text-sub); flex-shrink:0;}}
.tl-row.latest .tl-num {{background: var(--accent); color:#fff; border-color: var(--accent);}}
.metric-row {{display:flex; gap: 22px; margin: 10px 0 12px;}}
.metric-box {{flex:1;}}
.metric-box .v {{font-size: 21px; font-weight: 800; color: var(--text-main) !important;}}
.metric-box .l {{font-size: 11.5px; color: var(--text-sub) !important;}}
.verdict-good {{color: var(--good) !important; font-weight: 700;}}
.verdict-bad {{color: var(--bad) !important; font-weight: 700;}}
.verdict-muted {{color: var(--text-sub) !important;}}
.shared-chip {{display:inline-flex; align-items:center; gap:7px; padding:9px 16px; margin:4px 8px 4px 0;
               border-radius: 22px; background: var(--glass) !important; border:1px solid var(--gborder);
               font-size:13.5px; font-weight:600; color: var(--text-main) !important;
               backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);}}
.shared-chip .tick {{color: var(--good) !important; font-weight:800;}}
.stTextInput input, .stTextArea textarea {{
  background-color: var(--glass) !important; color: var(--text-main) !important; border-color: var(--gborder) !important;
}}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{color: var(--text-sub) !important; opacity: 1 !important;}}
.stButton button {{background: var(--glass) !important; color: var(--text-main) !important; border-color: var(--gborder) !important;
                   border-radius: 10px !important; backdrop-filter: blur(10px);}}
.stButton button[kind="primary"] {{background: var(--accent) !important; color: #FFFFFF !important; border-color: var(--accent) !important;}}
[data-testid="stExpander"] {{background: var(--glass) !important; border-color: var(--gborder) !important; border-radius: 14px; box-shadow: var(--shadow);}}
[data-baseweb="tab-list"] {{background-color: transparent !important; gap: 4px;}}
[data-baseweb="tab"] {{background: var(--glass) !important; border-radius: 10px 10px 0 0 !important; color: var(--text-sub) !important;}}
[aria-selected="true"][data-baseweb="tab"] {{color: var(--accent) !important; font-weight:700 !important;}}
code {{background-color: var(--accent-soft) !important; color: var(--text-main) !important;}}
hr {{border-color: var(--border-c) !important;}}
</style>
""", unsafe_allow_html=True)


def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


title_col, toggle_col = st.columns([6, 1])
with title_col:
    st.markdown('<div class="eyebrow">Agent orchestration benchmark</div>', unsafe_allow_html=True)
    st.title("CrewAI vs LangGraph")
with toggle_col:
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.toggle("Dark", key="dark_mode")
st.markdown('<div class="app-sub">A standardized, live, and measured comparison of two agent frameworks on SQL data-analysis tasks.</div>', unsafe_allow_html=True)

status = dataset_status()
with st.expander("Settings", expanded=not (status["db_exists"] and bool(get_groq_key()))):
    existing_key = get_groq_key()
    if existing_key:
        st.caption("GROQ_API_KEY loaded.")
    else:
        pasted_key = st.text_input("Groq API key", type="password")
        if pasted_key:
            os.environ["GROQ_API_KEY"] = pasted_key
            st.caption("Key set for this session.")
    db_ok = "✅" if status["db_exists"] else "❌"
    q_ok = "✅" if status["questions_exists"] else "❌"
    st.caption(f"{db_ok} database  ·  {q_ok} question set  ·  see app/data_setup.py if either is missing")

groq_key = get_groq_key()
can_run = status["db_exists"] and status["questions_exists"] and bool(groq_key)

if status["questions_exists"]:
    spec = importlib.util.spec_from_file_location("questions_only", QUESTIONS_PATH)
    qmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qmod)
    sample_questions = [v["question"] for v in qmod.queries.values()]
else:
    sample_questions = known_questions.RECOVERED_QUESTIONS

expected_df = None
if EXPECTED_RESULTS_PATH.exists():
    try:
        expected_df = pd.read_csv(EXPECTED_RESULTS_PATH)
    except Exception:
        expected_df = None


def get_expected(q: str):
    if expected_df is None:
        return None
    match = expected_df[expected_df["question"].str.strip().str.lower() == q.strip().lower()]
    return match.iloc[0]["answer"] if not match.empty else None


def missing_reqs():
    missing = []
    if not status["db_exists"]:
        missing.append("database")
    if not status["questions_exists"]:
        missing.append("question set")
    if not groq_key:
        missing.append("API key")
    return missing


tab1, tab2, tab3 = st.tabs([
    "1 · Standardized Pipeline",
    "2 · Live Execution",
    "3 · Run Your Own Question",
])

# ==========================================================================
# TAB 1 — Standardized Pipeline (landing view)
# ==========================================================================
with tab1:
    pipe_steps = ["Question", "Config", "CrewAI / LangGraph", "SQL Tools", "Database", "Metrics"]
    pipe_html = '<div class="flow-row">' + '<div class="flow-arrow">→</div>'.join(
        f'<div class="node done">{s}</div>' for s in pipe_steps
    ) + '</div>'
    st.markdown(f'<div class="card">{pipe_html}</div>', unsafe_allow_html=True)

    config = config_validate.load_config()
    model_name = config.get("model", {}).get("name", "—")
    temp = config.get("model", {}).get("temperature", "—")
    iters = config.get("benchmarks", {}).get("iterations", "—")
    nq = config.get("benchmarks", {}).get("total_questions", "—")

    chips = [
        f"Same model · {model_name}",
        f"Same temperature · {temp}",
        "Same database",
        "Same SQL tools",
        f"Same question set · {nq} questions",
        f"Same iterations · {iters}",
    ]
    chip_html = "".join(f'<span class="shared-chip"><span class="tick">✓</span>{c}</span>' for c in chips)
    st.markdown(f'<div class="card">{chip_html}</div>', unsafe_allow_html=True)

# ==========================================================================
# TAB 2 — Live Execution (the illustration)
# ==========================================================================
with tab2:
    st.markdown('<div class="section-head"><span class="dot"></span>Watch it execute</div>', unsafe_allow_html=True)
    st.caption("Pick a question — both frameworks run it live, and every real step is drawn as it happens.")

    if "illustrate_q" not in st.session_state:
        st.session_state.illustrate_q = sample_questions[0]

    pick_col, run_illu_col = st.columns([4, 1])
    with pick_col:
        st.session_state.illustrate_q = st.selectbox(
            "Pick a question", sample_questions,
            index=sample_questions.index(st.session_state.illustrate_q) if st.session_state.illustrate_q in sample_questions else 0,
            label_visibility="collapsed",
        )
    with run_illu_col:
        illu_clicked = st.button("▶ Run illustration", disabled=not can_run, type="primary", use_container_width=True)
    if not can_run:
        st.caption(f"Can't run yet — missing: {', '.join(missing_reqs())}. See Settings above.")

    def render_crew_diagram(container, state):
        final = state.get("final", False)
        agent_cls = "agent-node done" if final else ("agent-node active" if state.get("started") else "agent-node")
        ans_cls = "node done" if final else "node"
        tool_html = ""
        for key, label in [("list_tables_tool", "List tables"), ("get_schema_tool", "Get schema"), ("execute_sql_tool", "Execute SQL")]:
            cls = "node done" if key in state["done"] else ("node active" if key == state.get("active") else "node")
            tool_html += f'<div class="{cls}">{label}</div>'
        tl_html = '<div class="timeline">' + "".join(
            f'<div class="tl-row{" latest" if i == len(state["log"]) - 1 else ""}"><span class="tl-num">{i+1}</span>{line}</div>'
            for i, line in enumerate(state["log"])
        ) + '</div>'
        html = f"""
        <div class="flow-row">
          <div class="node done">Question</div><div class="flow-arrow">→</div>
          <div class="{agent_cls}">Agent<span class="node-sub">internal ReAct loop</span></div><div class="flow-arrow">→</div>
          <div class="{ans_cls}">Answer</div>
        </div>
        <div class="flow-tools">{tool_html}</div>
        {tl_html}
        """
        container.markdown(html, unsafe_allow_html=True)

    def render_lang_diagram(container, state):
        counts = state["counts"]
        active = state.get("active")
        final = state.get("final", False)
        start_cls = "node done" if counts["assistant"] > 0 else "node"
        a_cls = "node active" if active == "assistant" else ("node done" if counts["assistant"] > 0 else "node")
        t_cls = "node active" if active == "tools" else ("node done" if counts["tools"] > 0 else "node")
        end_cls = "node done" if final else "node"
        tl_html = '<div class="timeline">' + "".join(
            f'<div class="tl-row{" latest" if i == len(state["log"]) - 1 else ""}"><span class="tl-num">{i+1}</span>{line}</div>'
            for i, line in enumerate(state["log"])
        ) + '</div>'
        html = f"""
        <div class="flow-row">
          <div class="{start_cls}">START</div><div class="flow-arrow">→</div>
          <div class="{a_cls}">Assistant<span class="node-count">×{counts['assistant']}</span></div>
          <div class="flow-arrow-loop">⇄</div>
          <div class="{t_cls}">Tools<span class="node-count">×{counts['tools']}</span></div>
          <div class="flow-arrow">→</div><div class="{end_cls}">END</div>
        </div>
        {tl_html}
        """
        container.markdown(html, unsafe_allow_html=True)

    CREW_LOG_LABELS = {
        "start": "Agent receives the question",
        "list_tables_tool": "Calls <code>list_tables_tool</code> — discovers the table",
        "get_schema_tool": "Calls <code>get_schema_tool</code> — reads columns & sample rows",
        "execute_sql_tool": "Calls <code>execute_sql_tool</code> — runs the generated SQL",
        "done": "Returns the final answer",
    }

    def make_crew_tracker(container):
        state = {"started": False, "active": None, "done": set(), "final": False, "log": []}
        render_crew_diagram(container, state)

        def on_event(kind, label):
            if kind == "start":
                state["started"] = True
                state["log"].append(CREW_LOG_LABELS["start"])
            elif kind == "tool":
                state["active"] = label
                state["done"] = state["done"] | {label}
                state["log"].append(CREW_LOG_LABELS.get(label, f"Calls <code>{label}</code>"))
            elif kind == "done":
                state["final"] = True
                state["active"] = None
                state["log"].append(CREW_LOG_LABELS["done"])
            render_crew_diagram(container, state)
        return on_event

    def make_lang_tracker(container):
        state = {"counts": {"assistant": 0, "tools": 0}, "active": None, "final": False, "log": []}
        render_lang_diagram(container, state)

        def on_event(kind, label):
            if kind == "start":
                state["log"].append("Graph starts at <code>START</code>")
            elif kind == "assistant":
                state["counts"]["assistant"] += 1
                state["active"] = "assistant"
                state["log"].append(f"Assistant node fires (#{state['counts']['assistant']}) — {label.replace('Assistant: ', '')}")
            elif kind == "tool":
                state["counts"]["tools"] += 1
                state["active"] = "tools"
                state["log"].append(f"Tools node fires (#{state['counts']['tools']}) — executes the requested tool")
            elif kind == "done":
                state["final"] = True
                state["active"] = None
                state["log"].append("Graph reaches <code>END</code> — final answer")
            render_lang_diagram(container, state)
        return on_event

    if illu_clicked and can_run:
        q = st.session_state.illustrate_q
        st.markdown(f"**Question:** {q}")
        exec_col1, exec_col2 = st.columns(2)
        with exec_col1:
            st.markdown('<div class="fw-label">● CrewAI — agent loop</div>', unsafe_allow_html=True)
            crew_panel = st.empty()
        with exec_col2:
            st.markdown('<div class="fw-label">● LangGraph — state graph</div>', unsafe_allow_html=True)
            lang_panel = st.empty()

        crew_on_event = make_crew_tracker(crew_panel)
        lang_on_event = make_lang_tracker(lang_panel)
        crew_result = runner.run_single_question_crewai(q, groq_key, on_event=crew_on_event)
        lang_result = runner.run_single_question_langgraph(q, groq_key, on_event=lang_on_event)

        illu_expected = get_expected(q)

        def illu_tick(numerical_answer):
            if illu_expected is None or numerical_answer in (None, "Error retrieving numeric answer"):
                return ""
            try:
                correct = float(numerical_answer) == float(illu_expected)
            except (ValueError, TypeError):
                return ""
            return ' <span class="verdict-good">✓</span>' if correct else ' <span class="verdict-bad">✗</span>'

        res_col1, res_col2 = st.columns(2)
        for col, r in [(res_col1, crew_result), (res_col2, lang_result)]:
            with col:
                if r.get("status") != "success":
                    st.markdown(f'<div class="result-card">Run failed: {r.get("error", "unknown error")}</div>', unsafe_allow_html=True)
                    continue
                html = f"""
                <div class="result-card">
                  <div class="metric-row">
                    <div class="metric-box"><div class="v">{r['elapsed_s']:.1f}s</div><div class="l">elapsed time</div></div>
                    <div class="metric-box"><div class="v">{r.get('llm_call_count', '—')}</div><div class="l">LLM calls</div></div>
                  </div>
                  <div style="color:var(--text-sub); font-size:13px; margin-bottom:4px">Final answer:</div>
                  <div style="font-size:15px; color:var(--text-main)">{r.get('raw_answer', '')[:400]}</div>
                  <div style="color:var(--text-sub); font-size:13px; margin-top:8px">Extracted value: <code>{r.get('numerical_answer')}</code>{illu_tick(r.get('numerical_answer'))}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
                sqls = r.get("sql_queries") or []
                if sqls:
                    st.code(sqls[-1], language="sql")

# ==========================================================================
# TAB 3 — Run Your Own Question (evaluation)
# ==========================================================================
with tab3:
    st.markdown('<div class="section-head"><span class="dot"></span>Run your own question</div>', unsafe_allow_html=True)
    st.caption("Type any question about the sales data — both frameworks answer it, scored on accuracy, consistency, latency, and LLM calls.")

    if "question_box" not in st.session_state:
        st.session_state.question_box = ""

    pick_col, _ = st.columns([4, 1])
    with pick_col:
        picked = st.selectbox(
            "Or pick one of the benchmark questions", ["— pick a benchmark question —"] + sample_questions,
            label_visibility="visible",
        )
        if picked != "— pick a benchmark question —" and picked != st.session_state.question_box:
            st.session_state.question_box = picked
            st.rerun()

    in_col, btn_col = st.columns([5, 1])
    with in_col:
        question = st.text_input(
            "Question", value=st.session_state.question_box,
            placeholder="e.g. What is the maximum discount percentage?",
            label_visibility="collapsed",
        )
    with btn_col:
        run_clicked = st.button("Run ▶", disabled=not can_run, type="primary", use_container_width=True)

    check_consistency = st.checkbox(
        "Also check consistency (runs each framework twice — 2 extra calls, uses more of your rate limit)",
        value=False,
    )

    if not can_run:
        st.caption(f"Can't run yet — missing: {', '.join(missing_reqs())}. See Settings above.")

    if run_clicked and can_run:
        st.session_state.question_box = question
        q = question.strip() if question.strip() else sample_questions[0]
        expected = get_expected(q)

        spinner_text = "Running both frameworks twice each (checking consistency)..." if check_consistency else "Running both frameworks..."
        with st.spinner(spinner_text):
            run1_crew = runner.run_single_question_crewai(q, groq_key)
            run1_lang = runner.run_single_question_langgraph(q, groq_key)
            if check_consistency:
                run2_crew = runner.run_single_question_crewai(q, groq_key)
                run2_lang = runner.run_single_question_langgraph(q, groq_key)
            else:
                run2_crew = run2_lang = None

        def verdict(numerical_answer, expected_val):
            if expected_val is None or numerical_answer in (None, "Error retrieving numeric answer"):
                return None
            try:
                return float(numerical_answer) == float(expected_val)
            except (ValueError, TypeError):
                return None

        def consistency(r1, r2):
            if r2 is None or r1.get("status") != "success" or r2.get("status") != "success":
                return None, None
            a1, a2 = r1.get("numerical_answer"), r2.get("numerical_answer")
            try:
                same = float(a1) == float(a2)
            except (ValueError, TypeError):
                same = a1 == a2
            return same, abs(r1["elapsed_s"] - r2["elapsed_s"])

        st.markdown(f"**Question:** {q}")

        run_rows = []
        for label, r1, r2 in [("CrewAI", run1_crew, run2_crew), ("LangGraph", run1_lang, run2_lang)]:
            st.markdown(f'<div class="fw-label">● {label}</div>', unsafe_allow_html=True)
            if r1.get("status") != "success":
                st.markdown(f'<div class="result-card">Run failed: {r1.get("error", "unknown error")}</div>', unsafe_allow_html=True)
                run_rows.append({"Framework": label, "Answer": "—", "Correct": "—", "Latency (s)": None, "LLM calls": None, "Consistency": "failed"})
                continue
            v = verdict(r1.get("numerical_answer"), expected)
            same, spread = consistency(r1, r2)
            tick_html = (' <span class="verdict-good">✓</span>' if v is True else
                         ' <span class="verdict-bad">✗</span>' if v is False else '')
            v_html = ('<span class="verdict-good">✓ correct</span>' if v is True else
                      '<span class="verdict-bad">✗ incorrect</span>' if v is False else
                      '<span class="verdict-muted">not verifiable</span>')
            if not check_consistency:
                c_html = '<span class="verdict-muted">not checked this run — enable the checkbox above</span>'
                c_raw = "not checked"
            else:
                c_html = ('<span class="verdict-good">✓ stable across 2 runs</span>' if same is True else
                          '<span class="verdict-bad">✗ answer changed between runs</span>' if same is False else
                          '<span class="verdict-muted">unavailable</span>')
                c_raw = "stable" if same is True else ("changed" if same is False else "unavailable")
            avg_calls = None
            calls_label = "LLM calls"
            if r2 is not None and r1.get("llm_call_count") not in (None, "Unavailable") and r2.get("llm_call_count") not in (None, "Unavailable"):
                try:
                    avg_calls = round((r1["llm_call_count"] + r2["llm_call_count"]) / 2, 1)
                    calls_label = "LLM calls (avg of 2)"
                except TypeError:
                    avg_calls = r1.get("llm_call_count")
            calls_val = avg_calls if avg_calls is not None else r1.get("llm_call_count")
            sqls = r1.get("sql_queries") or []
            spread_html = f'<div class="metric-box"><div class="v">{spread:.1f}s</div><div class="l">latency spread (2 runs)</div></div>' if spread is not None else ""
            html = f"""
            <div class="result-card">
              <div class="metric-row">
                <div class="metric-box"><div class="v">{r1['elapsed_s']:.1f}s</div><div class="l">latency</div></div>
                <div class="metric-box"><div class="v">{calls_val if calls_val is not None else '—'}</div><div class="l">{calls_label}</div></div>
                {spread_html}
              </div>
              <div style="margin-bottom:8px">Accuracy vs. ground truth: {v_html}</div>
              <div style="margin-bottom:8px">Consistency: {c_html}</div>
              <div style="color:var(--text-sub); font-size:13px">Answer: <code>{r1.get('numerical_answer')}</code>{tick_html}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
            if sqls:
                st.code(sqls[-1], language="sql")
            run_rows.append({
                "Framework": label, "Answer": r1.get("numerical_answer"),
                "Correct": "yes" if v is True else ("no" if v is False else "not verifiable"),
                "Latency (s)": round(r1["elapsed_s"], 2), "LLM calls": calls_val, "Consistency": c_raw,
            })

        results_df = pd.DataFrame(run_rows)
        if len(results_df) == 2 and results_df["Latency (s)"].notna().all():
            st.markdown('<div class="section-head"><span class="dot"></span>This run, compared</div>', unsafe_allow_html=True)
            chart_c1, chart_c2 = st.columns(2)
            chart_color = "#6FA8E8" if st.session_state.dark_mode else "#4A90D9"
            with chart_c1:
                st.caption("Latency (s)")
                st.bar_chart(results_df.set_index("Framework")[["Latency (s)"]], color=chart_color)
            with chart_c2:
                st.caption("LLM calls")
                st.bar_chart(results_df.set_index("Framework")[["LLM calls"]], color=chart_color)

            faster = results_df.loc[results_df["Latency (s)"].idxmin(), "Framework"]
            fewer_calls = results_df.loc[results_df["LLM calls"].idxmin(), "Framework"] if results_df["LLM calls"].notna().all() else None
            both_correct = (results_df["Correct"] == "yes").all()
            insight_bits = [f"**{faster}** was faster on this question."]
            if fewer_calls:
                insight_bits.append(f"**{fewer_calls}** used fewer LLM calls.")
            insight_bits.append("Both frameworks answered correctly." if both_correct else "Not both frameworks answered correctly — check the cards above.")
            st.markdown(" ".join(insight_bits))

            st.markdown("##### Raw result")
            st.dataframe(results_df, hide_index=True, use_container_width=True)
