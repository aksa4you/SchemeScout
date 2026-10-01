"""SchemeScout - multilingual government scheme finder agent.

Pipeline (LangGraph):
  extract -> (ask_user | eligibility) -> (explain -> review -> [retry] -> act | no_match)

Env vars: GEMINI_API_KEY, TAVILY_API_KEY, optional GEMINI_MODEL, OUTPUT_DIR.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, TypedDict

import requests
from google import genai
from langgraph.graph import END, StateGraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

try:  # optional: load .env for local runs
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

log = logging.getLogger("schemescout")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
MAX_ATTEMPTS = 2  # explanation attempts before giving up on the reviewer

DISCLAIMER = (
    "Indicative result only. Eligibility rules are simplified; please confirm "
    "on the official portal before applying."
)
FLAG_NOTE = (
    "\n\nNote: some details could not be fully verified. "
    "Please check the official website."
)

# ---------------------------------------------------------------- LLM helpers
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def ask(prompt: str) -> str:
    return get_client().models.generate_content(model=MODEL, contents=prompt).text


def ask_json(prompt: str, retries: int = 1) -> dict:
    for attempt in range(retries + 1):
        raw = ask(prompt + "\nReturn ONLY valid JSON. No markdown, no commentary.")
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            log.warning("Bad JSON from model (attempt %d): %.80s", attempt + 1, raw)
    return {}


# ---------------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    user_text: str
    profile: dict
    missing: List[str]
    matches: List[dict]
    near_misses: List[dict]
    context: str
    explanation: str
    review: str
    flagged: bool
    attempts: int
    pdf_path: str
    trace: List[str]


def add_trace(state, msg):
    return state.get("trace", []) + [msg]


# ------------------------------------------------------------ profile + rules
REQUIRED = ["age", "state", "occupation", "annual_income"]

OCC_MAP = {
    "kisan": "farmer",
    "agricultural": "farmer",
    "agriculture": "farmer",
    "cultivator": "farmer",
    "farmer": "farmer",
}

CRITERION_LABEL = {
    "min_age": "minimum age",
    "max_age": "maximum age",
    "occupation": "occupation",
    "land": "land holding",
    "max_income": "income limit",
}


def load_schemes() -> list:
    with open(BASE_DIR / "schemes.json", encoding="utf-8") as f:
        return json.load(f)


SCHEMES = load_schemes()


def _num(value, cast):
    if value in (None, ""):
        return None
    try:
        return cast(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def coerce_profile(p: dict) -> dict:
    return {
        "age": _num(p.get("age"), int),
        "state": (str(p["state"]).strip() or None) if p.get("state") else None,
        "occupation": (str(p["occupation"]).strip() or None) if p.get("occupation") else None,
        "annual_income": _num(p.get("annual_income"), int),
        "land_acres": _num(p.get("land_acres"), float),
    }


def normalize_occ(text):
    t = (text or "").lower()
    for key, val in OCC_MAP.items():
        if key in t:
            return val
    return t


def check_scheme(profile, rules):
    failed = []
    age = profile.get("age")
    if "min_age" in rules and (age is None or age < rules["min_age"]):
        failed.append("min_age")
    if "max_age" in rules and (age is None or age > rules["max_age"]):
        failed.append("max_age")
    if "occupation" in rules:
        if normalize_occ(profile.get("occupation")) not in rules["occupation"]:
            failed.append("occupation")
    if "min_land_acres" in rules:
        if (profile.get("land_acres") or 0) < rules["min_land_acres"]:
            failed.append("land")
    if "max_income" in rules:
        income = profile.get("annual_income")
        if income is None or income > rules["max_income"]:
            failed.append("max_income")
    return (len(failed) == 0, failed)


# ----------------------------------------------------------------------- nodes
def extract_profile_node(state: AgentState) -> AgentState:
    raw = ask_json(
        f"""Extract a profile from this text (Hindi or English).
Keys: age (int), state (str, in English), occupation (str, in English, e.g. farmer, student),
annual_income (int, INR), land_acres (float). Use null if unknown.
Text: {state['user_text']}"""
    )
    profile = coerce_profile(raw)
    missing = [k for k in REQUIRED if profile.get(k) in (None, "")]
    return {
        "profile": profile,
        "missing": missing,
        "trace": add_trace(state, f"Extracted profile: {profile}"),
    }


def ask_user_node(state: AgentState) -> AgentState:
    q = ask(
        f"""Politely ask the user, in the same language as their message, to share
these missing details: {state['missing']}.
User message: {state['user_text']}"""
    )
    return {
        "explanation": q,
        "matches": [],
        "near_misses": [],
        "trace": add_trace(state, f"Missing info: {state['missing']}"),
    }


def eligibility_node(state: AgentState) -> AgentState:
    matches, near = [], []
    for s in SCHEMES:
        ok, failed = check_scheme(state["profile"], s["rules"])
        if ok:
            matches.append(s)
        elif len(failed) == 1:
            near.append({**s, "missing_criterion": failed[0]})
    return {
        "matches": matches,
        "near_misses": near,
        "trace": add_trace(state, f"Matched {len(matches)}, near-miss {len(near)}"),
    }


def web_search_tool(query: str) -> str:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
        json={"query": query, "max_results": 3},
        timeout=15,
    )
    resp.raise_for_status()
    return "\n".join(r["content"] for r in resp.json().get("results", []))


def explain_node(state: AgentState) -> AgentState:
    names = ", ".join(m["name"] for m in state["matches"])
    trace = state.get("trace", [])
    attempts = state.get("attempts", 0) + 1

    context = state.get("context")
    if not context:
        try:
            context = web_search_tool(f"{names} India eligibility documents required")
            trace = trace + ["Search tool OK"]
        except Exception as e:
            context = ""
            trace = trace + [f"Search tool unavailable: {type(e).__name__}"]

    feedback = ""
    if state.get("flagged"):
        feedback = (
            "\nA fact-checker rejected your previous answer. Remove every claim "
            f"not supported by the context or profile. Feedback: {state.get('review', '')}"
        )

    text = ask(
        f"""Explain simply, in the same language as the user's message, why this
person may qualify for: {names}. Profile: {state['profile']}.
Use ONLY this context for facts about documents or benefits, and say
'not sure' if the context is empty:
{context}{feedback}
User message: {state['user_text']}"""
    )
    return {
        "context": context,
        "explanation": text,
        "attempts": attempts,
        "trace": trace + [f"Explanation drafted (attempt {attempts})"],
    }


def reviewer_node(state: AgentState) -> AgentState:
    verdict = ask(
        f"""You are a strict fact-checker. Reply 'APPROVED' if every factual claim
in the ANSWER is supported by the CONTEXT or the PROFILE. Otherwise reply
'FLAGGED:' followed by the unsupported claims.
CONTEXT: {state.get('context', '')}
PROFILE: {state['profile']}
ANSWER: {state['explanation']}"""
    ).strip()
    flagged = not verdict.upper().startswith("APPROVED")
    return {
        "review": verdict,
        "flagged": flagged,
        "trace": add_trace(state, f"Review: {verdict[:80]}"),
    }


def no_match_node(state: AgentState) -> AgentState:
    near = [
        {"name": n["name"], "missing": CRITERION_LABEL.get(n["missing_criterion"], n["missing_criterion"])}
        for n in state.get("near_misses", [])
    ]
    text = ask(
        f"""In the same language as the user's message, kindly tell them that no
scheme in our list matches their profile right now. If this list is not
empty, mention these near matches and the one criterion each is missing: {near}.
Do not invent any other schemes or facts.
User message: {state['user_text']}"""
    )
    return {"explanation": text, "trace": add_trace(state, "No matches")}


def _safe(text: str) -> str:
    """Helvetica cannot draw Devanagari; replace unsupported characters."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def action_node(state: AgentState) -> AgentState:
    explanation = state["explanation"]
    if state.get("flagged"):
        explanation += FLAG_NOTE

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    def line(text, font="Helvetica", size=11, gap=16):
        nonlocal y
        for part in simpleSplit(_safe(text), font, size, width - 2 * margin):
            if y < margin:
                c.showPage()
                y = height - margin
            c.setFont(font, size)
            c.drawString(margin, y, part)
            y -= gap

    line("Your Matched Schemes", "Helvetica-Bold", 16, 26)
    for m in state["matches"]:
        line(f"- {m['name']}", "Helvetica-Bold")
        line(f"  Apply / details: {m['source']}")
    y -= 10
    # reportlab cannot shape Devanagari, so only include Latin-script text.
    if explanation.isascii():
        line("Why you may qualify", "Helvetica-Bold", 13, 22)
        for para in explanation.split("\n"):
            line(para or " ")
    else:
        line("Explanation is available in your language in the app response.")
    y -= 10
    line(DISCLAIMER, "Helvetica-Oblique", 9, 13)
    c.save()

    return {
        "explanation": explanation,
        "pdf_path": str(path),
        "trace": add_trace(state, f"PDF created: {path.name}"),
    }


# ----------------------------------------------------------------------- graph
def route_after_extract(state: AgentState) -> str:
    return "ask_user" if state["missing"] else "eligibility"


def route_after_eligibility(state: AgentState) -> str:
    return "explain" if state["matches"] else "no_match"


def route_after_review(state: AgentState) -> str:
    if state.get("flagged") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "explain"
    return "act"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("extract", extract_profile_node)
    g.add_node("ask_user", ask_user_node)
    g.add_node("eligibility", eligibility_node)
    g.add_node("explain", explain_node)
    g.add_node("review", reviewer_node)
    g.add_node("act", action_node)
    g.add_node("no_match", no_match_node)

    g.set_entry_point("extract")
    g.add_conditional_edges(
        "extract", route_after_extract, {"ask_user": "ask_user", "eligibility": "eligibility"}
    )
    g.add_conditional_edges(
        "eligibility", route_after_eligibility, {"explain": "explain", "no_match": "no_match"}
    )
    g.add_edge("explain", "review")
    g.add_conditional_edges("review", route_after_review, {"explain": "explain", "act": "act"})
    g.add_edge("ask_user", END)
    g.add_edge("no_match", END)
    g.add_edge("act", END)
    return g.compile()


graph = build_graph()


def run_agent(user_text: str) -> dict:
    return graph.invoke({"user_text": user_text, "trace": []})


if __name__ == "__main__":
    out = run_agent("Main 45 saal ka kisan hoon, UP mein 2 acre zameen hai, salana aay 1 lakh")
    print("PROFILE:", out.get("profile"))
    print("\nEXPLANATION:\n", out.get("explanation"))
    print("\nREVIEW:", out.get("review"))
    print("PDF:", out.get("pdf_path"))
    print("\nTRACE:")
    for step in out.get("trace", []):
        print(" -", step)

