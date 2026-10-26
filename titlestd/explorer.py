"""The architecture page: each stage of a run, each component's job, counts from a state.

Owner: feat/architecture-page (after the build).
Governed by: DESIGN.md 1.6 (the flow), 1.10 (waves), 1.11 (architecture),
2.2 (what is in the mock), 2.5 (the free modes), 2.8 (the two browser
libraries of the architecture page); docs/build-decisions.md Q16 (the rows of
the table), Q21 (the audited titles of a path), Q40 (the answer labels), S08
(the schema of the state file is titlestd/store.py's), S13 (the settings of a
command), "Diagram rules not to copy"; static/vendor/README.md (how the page
loads the two libraries).

Start it from the repository root, after a free run:

    python3 -m titlestd run
    python3 -m titlestd.explorer

It serves the state file of that run, out/state.sqlite, at
http://127.0.0.1:7654/ until it is stopped with Ctrl+C; --out-dir names the
output folder of another run. There is a page for
the overview, for each stage of a run (set-up, Wave 1, Wave 2, the close) and
for each component. A stage page shows the stage's flowchart, with the
components that act in it styled and listed as text; a component page shows
its job, module, design sections, record entries and the stages it acts in.
Both show counts read from the state. docs/architecture.md holds the four
flowcharts and the start command.

Decided here, where the record is silent:

- What it refuses. It answers only at 127.0.0.1:7654: a request whose Host
  header is anything else gets 403, the guard against a page of another site
  that a renamed address points here. Every method but GET gets 405 with
  "Allow: GET", HEAD included. It starts only on a free run's state: a
  missing file, a file that is not a state of this program and a state whose
  llm setting is not one of the two free modes, offline and replay (2.5),
  stop it with one line and exit status 1. The free modes are a list of
  their own, not "every mode but anthropic": set-up records the settings
  before any scoring (Q37), so a state with no llm setting holds no answer,
  and a mode added to settings.LLMS later is refused until this list names
  it. A state with a model answer labelled "live" (contracts.LABEL_LIVE,
  Q40) is refused too. A later run may replace the file while the page is
  open, so every page checks the same again and answers 503 with a fixed
  text when the state no longer passes. htmx swaps no error into the page,
  so static/app.js then loads the address in full, and the 503 or a 404 is
  seen.
- Read-only. Each request opens the file with SQLite's read-only mode and
  closes it again, so the page never writes the state, and shows the state as
  it is now.
- A run that did not finish. A run writes its final decisions, audits,
  golden-set scores, review queue and answer labels in one step at its end
  (store.save_run); one that stopped, or one still under way, leaves the
  state as it stood at its last completed step. The page shows that state
  as it is, and the heading of every table of counts says so. The labels of
  the answers are read from the model answers, each of which carries its
  own: the answer labels a finished run writes are the same set (Q40), and
  a run that stopped has none.
- Counts and code-defined labels only. Every query returns COUNT(*), and
  every value it compares with is a parameter taken from a table in the code
  (the stages, the routes, the occasions of titlestd.pipeline and so on). A
  value the state holds is never shown: an audit outcome or a setting is
  shown by counting the rows that hold each value the code knows, and
  showing the code's label of the one found. No title, description, quote,
  persona or role id reaches a page.
- S08 makes the schema store.py's business and not a contract for any other
  part. The page is the one part that reads it, by the names store.py gives
  its tables and columns, so a change to the schema changes the page's
  queries with it. The integration tests run the real pipeline and hold the
  page's counts to that run and to each other (each set of counts that
  splits a total adds up to it), so such a change fails there rather than
  showing zeros.
- The golden-set passes. Wave 1 and the close put the golden titles through
  the ML match, the LLM and the rules with no agent (titlestd.pipeline), and
  their answers and routes are recorded like any other. The counts of the
  rules leave those two passes out and show them on a row of their own, so
  the LLM answers sent to the agent are the agent's reviews. In Wave 2 the
  golden titles are scored for release with the rest of the long tail.
- One escape. _e is the only caller of html.escape, and every value put into
  a page passes through it; markup this module built is marked as such
  (_Html) and passes through unchanged. The flowcharts are escaped too:
  Mermaid reads a chart from the element's HTML and decodes it first
  (static/vendor/README.md).
- A page is always the whole document. A link is a real href; htmx fetches
  the same address and takes the panel and the navigation out of it
  (hx-select, hx-select-oob), so a response never depends on a request
  header, and history, with no cache, gets the same page.
- One thread per request. wsgiref's server takes one connection at a time,
  and a connection a browser opens ahead and leaves idle would hold up every
  other request; the page's server takes each in a thread of its own, and
  each request opens its own connection to the state.
- A module of its own. The page starts as python3 -m titlestd.explorer, not
  as a command of python3 -m titlestd, because cli.py is a shared file of
  rule 2. It takes --config and the setting flags of every command (S13),
  through titlestd.settings, and uses only out_dir. A bad command line or
  setting exits 2.
- The flowcharts are constants, equal to the fences of docs/architecture.md.
  Each draws one stage of titlestd.pipeline: its component nodes are exactly
  the components acting in it, styled with the class "acting"; the other
  nodes are inputs, lists and outcomes. Every label is fixed text: no
  directive, no click, no HTML. None draws a rule of "Diagram rules not to
  copy": there are two machine paths, the ML match runs from Wave 1 on, a
  title marked "no suitable role" goes to the catalog owner and never to a
  generic role, a look-up hit goes to no model and not through the rules,
  and the audit samples 60 with the lines 2 and 12. The audited titles of a
  path take their labels as Approved, and its outcome applies to the rest
  (Q21).
- The components are the steps of 1.6 as the mock builds them, plus the five
  parts the mock adds around them: the golden set, the generic family role
  of the close, the model switch and the state file, and the report that
  prints the table. Learning (1.6 step 10) is the component of
  titlestd/lookup.py that takes people's decisions; the look-up is the same
  module's exact match. Each job is text from DESIGN.md, word for word.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import socketserver
import sqlite3
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from wsgiref.simple_server import WSGIServer, make_server

from titlestd import pipeline, store
from titlestd.contracts import (
    LABEL_LIVE,
    LABEL_STAND_IN,
    AuditOutcome,
    ChooseKind,
    HumanAnswer,
    MachinePath,
    ReviewKind,
    Route,
    Stage,
    Status,
    TableRow,
    TitlestdError,
    recorded_label,
)
from titlestd.settings import (
    EMBEDDERS,
    LLMS,
    SettingsError,
    add_setting_arguments,
    settings_from_arguments,
)

HOST = "127.0.0.1"
PORT = 7654
ADDRESS = f"http://{HOST}:{PORT}/"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# static/vendor/README.md, "How the page loads them".
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
HTMX_CONFIG_META = (
    '<meta name="htmx-config" content=\'{"includeIndicatorStyles":false,'
    '"globalViewTransitions":false,"defaultSettleDelay":0,"allowScriptTags":false,'
    '"allowEval":false,"historyCacheSize":0}\'>'
)

# The tables of the state file, as titlestd/store.py makes them.
TABLES = (
    "settings",
    "thresholds",
    "title_rows",
    "lookup",
    "examples",
    "scorings",
    "ml_shortlist",
    "model_answers",
    "shown_roles",
    "human_decisions",
    "golden_scores",
    "audits",
    "audit_samples",
    "decisions",
    "review_queue",
    "answer_labels",
)

# The files the page loads, by address: the only files it serves.
_CONTENT_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
STATIC_FILES = {
    f"/static/{name}": (name, _CONTENT_TYPES[Path(name).suffix])
    for name in ("app.css", "app.js", "vendor/htmx.min.js", "vendor/mermaid.min.js")
}


class ExplorerError(TitlestdError):
    """The state file is missing, cannot be read, or is not a free run's."""


# --- What a page counts ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Count:
    """One number of a page: COUNT(*) of a query, with its parameters."""

    label: str
    sql: str
    params: tuple[str | int, ...] = ()


@dataclass(frozen=True, slots=True)
class Found:
    """One code label of a page: the labels of the options whose query counts a row.

    Each option is a query, its parameters and the label shown when it counts
    at least one row. No option found shows "none".
    """

    label: str
    options: tuple[tuple[str, tuple[str | int, ...], str], ...]


Fact = Count | Found

_TABLES_MADE = (
    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
    f" AND name IN ({', '.join('?' for _ in TABLES)})"
)
# Each model answer carries its label, so a run that stopped holds them too; a
# finished run also writes them in answer_labels, the same set (Q40).
_LABELLED = "SELECT COUNT(*) FROM model_answers WHERE source = ?"
_LABELLED_LIKE = "SELECT COUNT(*) FROM model_answers WHERE source LIKE ?"
_SETTING = "SELECT COUNT(*) FROM settings WHERE name = ? AND value = ?"
_NAMED = "SELECT COUNT(*) FROM title_rows WHERE has_person_name = 1"
_NO_FAMILY = "SELECT COUNT(*) FROM title_rows WHERE family_usable = 0"
_ON_WAVE1_LIST = "SELECT COUNT(*) FROM title_rows WHERE on_wave1_list = 1"
_IN_GOLDEN_SET = "SELECT COUNT(*) FROM title_rows WHERE in_golden_set = 1"
_GOLDEN_LEARNED = (
    "SELECT COUNT(*) FROM title_rows AS t WHERE in_golden_set = 1 AND ("
    "EXISTS (SELECT 1 FROM lookup AS l"
    " WHERE l.family_code = t.family_code AND l.cleaned_title = t.cleaned_title)"
    " OR EXISTS (SELECT 1 FROM examples AS e"
    " WHERE e.family_code = t.family_code AND e.cleaned_title = t.cleaned_title))"
)
_LOOKUP_JOINED = "SELECT COUNT(*) FROM lookup WHERE joined_at = ?"
_EXAMPLES_JOINED = "SELECT COUNT(*) FROM examples WHERE joined_at = ?"
_SCORED = "SELECT COUNT(*) FROM scorings WHERE stage = ?"
_ROUTED = "SELECT COUNT(*) FROM scorings WHERE stage = ? AND route = ?"
_ANSWERS = "SELECT COUNT(*) FROM model_answers WHERE stage = ? AND step = ?"
_ANSWERED = "SELECT COUNT(*) FROM model_answers WHERE step = ? AND answer = ?"
# An answer of the golden-set passes, which call no agent: a golden title's
# answer at one of the two stages given (_GOLDEN_PASS_STAGES).
_IN_GOLDEN_PASS = (
    "a.stage IN (?, ?) AND EXISTS (SELECT 1 FROM title_rows AS t WHERE t.in_golden_set = 1"
    " AND t.family_code = a.family_code AND t.cleaned_title = a.cleaned_title)"
)
_RULED = (
    "SELECT COUNT(*) FROM model_answers AS a WHERE a.step = ? AND a.route = ?"
    f" AND NOT ({_IN_GOLDEN_PASS})"
)
_GOLDEN_PASS = f"SELECT COUNT(*) FROM model_answers AS a WHERE a.step = ? AND {_IN_GOLDEN_PASS}"
_DECIDED = "SELECT COUNT(*) FROM human_decisions WHERE occasion = ?"
_DECIDED_AS = "SELECT COUNT(*) FROM human_decisions WHERE occasion = ? AND answer = ?"
_ANSWERED_BY_PEOPLE = "SELECT COUNT(*) FROM human_decisions WHERE answer = ?"
_SIMULATED = "SELECT COUNT(*) FROM human_decisions WHERE simulated = 1"
_AUDITED = "SELECT COUNT(*) FROM audit_samples WHERE path = ?"
_AUDIT_OUTCOME = "SELECT COUNT(*) FROM audits WHERE path = ? AND outcome = ?"
_IN_ROW = "SELECT COUNT(*) FROM decisions WHERE table_row = ?"
_WITH_STATUS = "SELECT COUNT(*) FROM decisions WHERE status = ?"
_FREE_SETTING = "SELECT COUNT(*) FROM settings WHERE name = ? AND value IN (?, ?)"
# One query per table: the state file's own page, and every plain count of a table.
_ROWS_IN = {table: "SELECT COUNT(*) FROM " + table for table in TABLES}

# A replayed answer's label (contracts.recorded_label), as a LIKE pattern.
_RECORDED_PATTERN = recorded_label("%", "%")
# The llm settings of a free run (2.5): the deterministic stand-in and recorded answers.
_FREE_LLMS = ("offline", "replay")
_CHOOSE, _REVIEW = store.STEP_CHOOSE, store.STEP_REVIEW
# The stages after set-up: the models score titles and people's decisions are learned.
_LATER_STAGES = (Stage.WAVE1, Stage.WAVE2, Stage.CLOSE)
# The stages whose golden-set pass calls no agent (titlestd.pipeline); in Wave 2
# the golden titles are scored for release with the rest of the long tail.
_GOLDEN_PASS_STAGES = (Stage.WAVE1, Stage.CLOSE)


def _setting(label: str, name: str, values: Iterable[str]) -> Found:
    return Found(label, tuple((_SETTING, (name, value), value) for value in values))


def _audit(path: MachinePath, name: str) -> tuple[Count, Found]:
    """How many titles of a path were audited, and the outcome of its audit."""
    outcomes = tuple((_AUDIT_OUTCOME, (path, each), each.value) for each in AuditOutcome)
    return (
        Count(f"Audited on the {name} path", _AUDITED, (path,)),
        Found(f"Outcome of the {name} path", outcomes),
    )


def _learned(stage: Stage, where: str = "") -> tuple[Count, Count]:
    """The look-up entries and the examples that joined at a stage."""
    return (
        Count(f"Look-up entries learned{where}", _LOOKUP_JOINED, (stage,)),
        Count(f"Examples learned{where}", _EXAMPLES_JOINED, (stage,)),
    )


# The facts more than one page shows, each under one label.
_LLM_SETTING = _setting("LLM setting", "llm", LLMS)
_EMBEDDER_SETTING = _setting("Embedder setting", "embedder", EMBEDDERS)
_ANSWER_LABELS = Found(
    "Answers came from",
    (
        (_LABELLED, (LABEL_STAND_IN,), "stand-ins"),
        (_LABELLED_LIKE, (_RECORDED_PATTERN,), "a recording"),
    ),
)
_TITLE_ROWS = Count("Title rows", _ROWS_IN["title_rows"])
_GOLDEN_TITLES = Count("Titles in the golden set", _IN_GOLDEN_SET)
_SEED_ENTRIES = Count("Entries from the seed look-up", _LOOKUP_JOINED, (Stage.SETUP,))
_LOOKUP_HITS = Count("Look-up hits", _IN_ROW, (TableRow.LOOKUP,))
_WAVE1_APPROVALS = Count("Wave 1 approvals", _DECIDED, (pipeline.OCCASION_WAVE1,))
_GOLDEN_LABELS = Count("Golden-set labels", _DECIDED, (pipeline.OCCASION_GOLDEN,))
_AUDIT_LABELS = Count("Audit labels", _DECIDED, (pipeline.OCCASION_AUDIT,))
_QUEUE_DECISIONS = Count("Review-queue decisions", _DECIDED, (pipeline.OCCASION_QUEUE,))
# A person could not decide at the close only (Q01).
_UNDECIDED = Count(
    "Could not decide", _DECIDED_AS, (pipeline.OCCASION_QUEUE, HumanAnswer.UNDECIDED)
)
_AUDITS = (
    *_audit(MachinePath.MODELS_AGREE, "models-agree"),
    *_audit(MachinePath.AGENT_SETTLED, "agent-settled"),
    _AUDIT_LABELS,
)
_GOLDEN_TITLES_LEARNED = Count("Golden-set titles learned", _GOLDEN_LEARNED)
_GENERIC = Count("Given a generic family role", _IN_ROW, (TableRow.GENERIC,))
_LEGACY = Count("Left on a legacy title", _IN_ROW, (TableRow.LEGACY,))
_QUEUE_FILE = Count("Titles in the review-queue file", _ROWS_IN["review_queue"])


# --- The components ----------------------------------------------------------------

# DESIGN.md 2.6, the first line: what one run does.
RUN_TEXT = (
    "One command runs set-up, Wave 1, Wave 2 and the close on the mock data and prints one table."
)

# DESIGN.md sections a page cites, with their headings.
DESIGN_SECTIONS = {
    "1.1": "The idea in four lines",
    "1.2": "Vocabulary",
    "1.4": "Inputs and what each is used for",
    "1.6": "The flow",
    "1.7": "The rules",
    "1.8": "What people always decide",
    "1.9": "Confidence, the random audit and statuses",
    "1.10": "Waves",
    "2.1": "Purpose and size",
    "2.2": "In and out",
    "2.4": "Settled choices",
    "2.5": "Models and settings",
    "2.6": "What a run shows",
}


@dataclass(frozen=True, slots=True)
class Component:
    """One part of the program, as its page shows it.

    node is its name in the flowcharts, module the file that builds it.
    design lists (section, detail) pairs:
    the sections of DESIGN.md it builds. job is text from DESIGN.md, one
    paragraph per item, word for word. entries are those of
    docs/build-decisions.md that govern it, each named in its module's
    docstring.
    """

    slug: str
    node: str
    name: str
    module: str
    design: tuple[tuple[str, str], ...]
    entries: tuple[str, ...]
    job: tuple[str, ...]
    facts: tuple[Fact, ...]


COMPONENTS = (
    Component(
        slug="prepare",
        node="prepare",
        name="Prepare",
        module="titlestd/prepare.py",
        design=(("1.6", "step 1"), ("1.4", "")),
        entries=("Q11", "Q14", "Q26"),
        job=(
            "Clean each title (lower-case, strip punctuation, locations and requisition "
            "numbers, expand a small abbreviation list such as sr to senior), mask names, "
            "group to one row per title and family.",
            "Names are masked before any text reaches a model. A title that contains a "
            "person's name goes to a person and to no model.",
        ),
        facts=(
            _TITLE_ROWS,
            Count("With a person's name", _NAMED),
            Count("With no usable family code", _NO_FAMILY),
        ),
    ),
    Component(
        slug="look-up",
        node="lookup",
        name="Look-up",
        module="titlestd/lookup.py",
        design=(("1.6", "step 2"), ("2.4", "The seed look-up")),
        entries=("Q09", "Q13"),
        job=(
            "Exact match on cleaned title plus family in the look-up table. A hit takes that "
            "role with no model.",
            "The seed look-up exists so that the look-up step has hits to show.",
        ),
        facts=(
            _LOOKUP_HITS,
            _SEED_ENTRIES,
            Count("Entries at the end of the run", _ROWS_IN["lookup"]),
        ),
    ),
    Component(
        slug="ml-match",
        node="ml_match",
        name="ML match",
        module="titlestd/matcher.py",
        design=(("1.6", "step 3"),),
        entries=("Q04", "Q12", "Q18"),
        job=(
            "An off-the-shelf embedding model turns the title and description into a vector. "
            "A nearest-neighbor match against every catalog role except the generic family "
            "roles (its name and one-line definition) and every human-approved title returns "
            "a shortlist of ten roles, a first choice, and a margin (the gap between first "
            "and second).",
        ),
        facts=tuple(
            Count(f"Titles matched in {stage.value}", _SCORED, (stage,)) for stage in _LATER_STAGES
        ),
    ),
    Component(
        slug="llm-fallback",
        node="fallback",
        name="LLM fallback",
        module="titlestd/fallback.py",
        design=(("1.6", "step 4"), ("2.4", "Role score")),
        entries=("Q11", "Q15", "Q18"),
        job=(
            "The ML model's choice may stand alone only if the margin clears a cut-off and "
            "the chosen role has at least three human-approved examples. Otherwise the title "
            "goes to the LLM.",
            "The LLM sees the masked record and the ten roles, numbered and shuffled. It "
            'must answer with a number from 1 to 10, or "no match", or "insufficient data", '
            "plus a short quote from the record as evidence.",
            "The ML-alone route is built and off by default.",
        ),
        facts=(
            *(
                Count(f"LLM answers in {stage.value}", _ANSWERS, (stage, _CHOOSE))
                for stage in _LATER_STAGES
            ),
            Count("Answered with a number", _ANSWERED, (_CHOOSE, ChooseKind.ROLE)),
            Count("Answered no match", _ANSWERED, (_CHOOSE, ChooseKind.NO_MATCH)),
            Count("Answered insufficient data", _ANSWERED, (_CHOOSE, ChooseKind.INSUFFICIENT_DATA)),
            Count("No usable answer", _ANSWERED, (_CHOOSE, ChooseKind.INVALID)),
        ),
    ),
    Component(
        slug="rules",
        node="rules",
        name="Rules",
        module="titlestd/rules.py",
        design=(("1.6", "step 5"), ("1.7", "")),
        entries=("Q04", "Q15", "Q20", "Q39"),
        job=(
            "Code checks every proposal (table in 1.7).",
            "If several fail, a person beats the agent. Everything the agent returns is "
            "checked by the same rules except agreement. If it fails any, or the agent "
            "escalates, the title goes to a person.",
        ),
        facts=(
            *(
                Count(label, _RULED, (step, route, *_GOLDEN_PASS_STAGES))
                for label, step, route in (
                    ("LLM answers that passed every check", _CHOOSE, Route.ACCEPT),
                    ("LLM answers sent to the agent", _CHOOSE, Route.AGENT),
                    ("LLM answers sent to a person", _CHOOSE, Route.PERSON),
                    ("Agent answers that passed every check", _REVIEW, Route.ACCEPT),
                    ("Agent answers sent to a person", _REVIEW, Route.PERSON),
                )
            ),
            Count(
                "LLM answers of the golden-set passes, only scored",
                _GOLDEN_PASS,
                (_CHOOSE, *_GOLDEN_PASS_STAGES),
            ),
        ),
    ),
    Component(
        slug="agent",
        node="agent",
        name="Agent",
        module="titlestd/agent.py",
        design=(("1.6", "step 6"), ("2.4", "The agent")),
        entries=("Q10", "Q19", "Q20"),
        job=(
            'For a disagreement between the LLM and the ML model, a family mismatch, or a "no '
            'match", one LLM agent reviews the case. Unlike the fallback it has two read-only '
            "searches: the whole catalog (generic family roles excluded), and similar "
            "human-approved titles.",
            "Picking the LLM's role confirms it; picking another corrects it. Its answer goes "
            "back through the rules.",
            "The agent is a plain loop: at most four search calls, then one structured "
            "answer. No agent framework.",
        ),
        facts=(
            *(
                Count(f"Reviews in {stage.value}", _ANSWERS, (stage, _REVIEW))
                for stage in (Stage.WAVE1, Stage.WAVE2)
            ),
            Count("Answered with a number", _ANSWERED, (_REVIEW, ReviewKind.ROLE)),
            Count("Escalated", _ANSWERED, (_REVIEW, ReviewKind.ESCALATE)),
            Count("No usable answer", _ANSWERED, (_REVIEW, ReviewKind.INVALID)),
        ),
    ),
    Component(
        slug="people",
        node="people",
        name="People",
        module="titlestd/people.py",
        design=(("1.6", "step 7"), ("1.8", ""), ("2.4", "People are simulated")),
        entries=("Q01", "Q03", "Q27"),
        job=(
            "HR SMEs and reviewers decide everything no machine may settle (1.8).",
            "People are simulated from the truth file, with a small seeded error rate, and "
            "every simulated decision is marked as simulated.",
        ),
        facts=(
            _WAVE1_APPROVALS,
            _GOLDEN_LABELS,
            _AUDIT_LABELS,
            _QUEUE_DECISIONS,
            _UNDECIDED,
            Count("No suitable role", _ANSWERED_BY_PEOPLE, (HumanAnswer.NO_SUITABLE_ROLE,)),
            Count("Marked as simulated", _SIMULATED),
        ),
    ),
    Component(
        slug="golden-set",
        node="golden",
        name="Golden set",
        module="titlestd/golden.py",
        design=(("1.2", "Golden set"), ("1.10", "")),
        entries=("Q07", "Q08"),
        job=(
            "A fixed random sample of long-tail titles that HR SMEs label without seeing the "
            "system's answer. It is re-scored at every wave and never becomes a look-up "
            "entry or an example.",
        ),
        facts=(
            _GOLDEN_TITLES,
            _GOLDEN_LABELS,
            Count("Times scored", _ROWS_IN["golden_scores"]),
            _GOLDEN_TITLES_LEARNED,
        ),
    ),
    Component(
        slug="random-audit",
        node="audit",
        name="Random audit",
        module="titlestd/audit.py",
        design=(("1.9", ""), ("2.4", "Audit at mock scale")),
        entries=("Q07", "Q16", "Q21", "Q23"),
        job=(
            "The ML margin and the example count only decide who looks next. Trust comes "
            "from people's labels, never from a model's own score.",
            "Sample 60 titles per path. Use the same statistics as 14 in 200: a path is "
            "confirmed if the number wrong is at most the largest count that a path at "
            "exactly 90% right would pass at most one time in ten, computed from the "
            "binomial for the sample size (2 of 60). It is stopped above 20% wrong (more "
            "than 12 of 60, the same line as 41 or more in 200); otherwise it is "
            "provisional. If a path holds 60 titles or fewer, every title in it is labeled: "
            "each takes its label with status Approved, and the path gets no status.",
        ),
        facts=_AUDITS,
    ),
    Component(
        slug="generic-role",
        node="generic",
        name="Generic family role",
        module="titlestd/close.py",
        design=(("1.2", "Generic family role"), ("1.8", "")),
        entries=("Q01", "Q03", "Q25", "Q26"),
        job=(
            "At the close, a title nobody could decide takes its family's generic role, "
            "marked provisional and capped at 2% of employees. A widely held or "
            "compliance-flagged title never does.",
            'A title marked "no suitable role" goes to the catalog owner, who adds a role or '
            "names one.",
        ),
        facts=(_UNDECIDED, _GENERIC, _LEGACY),
    ),
    Component(
        slug="release",
        node="release",
        name="Release",
        module="titlestd/release.py",
        design=(("1.6", "step 9"), ("2.6", "")),
        entries=("Q05", "Q16", "Q27", "Q40"),
        job=(
            "After the HR lead signs, the system writes a load file (one row per employee: "
            "the new titles and their status) and a reverse file (how to restore each "
            "employee's legacy title). The employer's HR-system team loads them; the system "
            "never writes to the HR system.",
        ),
        facts=(
            *(Count(f"Status {status.value}", _WITH_STATUS, (status,)) for status in Status),
            _LEGACY,
            _QUEUE_FILE,
        ),
    ),
    Component(
        slug="learning",
        node="learning",
        name="Learning",
        module="titlestd/lookup.py",
        design=(("1.6", "step 10"),),
        entries=("Q07", "Q13", "Q14"),
        job=(
            "Nothing is trained. Titles a person approved or labeled join the look-up table "
            "and the ML model's examples. A title the agent settled does not, until a person "
            "approves it.",
        ),
        facts=(
            *(fact for stage in _LATER_STAGES for fact in _learned(stage, f" in {stage.value}")),
            _GOLDEN_TITLES_LEARNED,
        ),
    ),
    Component(
        slug="model-switch",
        node="switch",
        name="Model switch",
        module="titlestd/switch.py",
        design=(("2.5", ""),),
        entries=("Q30", "Q31", "Q32", "Q40", "S09"),
        job=(
            "Three calls sit behind a switch: embed(texts), choose(record, ten roles) and "
            "review(case, tools). Masking, numbering, the quote check, the search cap and the "
            "number-to-role mapping stay in shared code, so no model output can become a "
            "role by itself.",
        ),
        facts=(_LLM_SETTING, _EMBEDDER_SETTING, _ANSWER_LABELS),
    ),
    Component(
        slug="state-file",
        node="statefile",
        name="State file",
        module="titlestd/store.py",
        design=(("2.1", ""),),
        entries=("Q06", "Q38"),
        job=(
            "Keep it small and plain: one package, the standard library by default, a local "
            "SQLite file or plain files for state.",
        ),
        facts=tuple(Count(f"Rows in {table}", _ROWS_IN[table]) for table in TABLES),
    ),
    Component(
        slug="report",
        node="report",
        name="Report",
        module="titlestd/report.py",
        design=(("2.6", ""),),
        entries=("Q05", "Q08", "Q16", "Q40"),
        job=(
            RUN_TEXT,
            "Rows: look-up hit, approved in Wave 1, models agree, agent-settled, review "
            "queue, generic family role, left on legacy title.",
        ),
        facts=tuple(Count(f"Titles in the row {row.value}", _IN_ROW, (row,)) for row in TableRow),
    ),
)


# --- The stages --------------------------------------------------------------------

# Each flowchart is fixed text, the same as its fence in docs/architecture.md. Its
# last two lines style the nodes of the components acting in its stage.
ACTING_STYLE = "classDef acting fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#1c1917"


def _styled(acting: tuple[str, ...]) -> str:
    return f"{ACTING_STYLE}\n    class {','.join(acting)} acting"


SETUP_ACTING = ("prepare", "lookup", "golden", "switch", "statefile")
WAVE1_ACTING = (
    "lookup",
    "ml_match",
    "fallback",
    "rules",
    "agent",
    "people",
    "golden",
    "learning",
    "statefile",
)
WAVE2_ACTING = (
    "ml_match",
    "fallback",
    "rules",
    "agent",
    "people",
    "audit",
    "golden",
    "learning",
    "statefile",
)
CLOSE_ACTING = (
    "people",
    "learning",
    "golden",
    "ml_match",
    "fallback",
    "rules",
    "generic",
    "release",
    "report",
    "statefile",
)

SETUP_FLOWCHART = f"""\
flowchart TD
    inputs([Catalog, employee records and seed look-up]) --> prepare[Prepare]
    prepare -->|one row per title and family| rows([Title rows])
    rows -->|most widely held| wave1([Wave 1 list])
    rows -->|drawn with the run seed| golden[Golden set]
    inputs --> lookup[Look-up]
    lookup -->|seed titles| table([Look-up table and examples])
    wave1 -.->|left out of the draw| golden
    table -.->|left out of the draw| golden
    switch[Model switch] -->|embed, choose, review| models([Embedder and LLM for the run])
    thresholds([Settings and thresholds]) -->|recorded before any scoring| statefile[(State file)]
    rows --> statefile
    wave1 --> statefile
    golden --> statefile
    {_styled(SETUP_ACTING)}
"""

WAVE1_FLOWCHART = f"""\
flowchart TD
    rows([Every title row]) --> lookup[Look-up]
    lookup -->|hit, no model| hit([Look-up hit, Approved])
    lookup -->|miss, on the Wave 1 list| ml_match[ML match]
    golden[Golden set] -->|one extra pass, no agent| ml_match
    ml_match -->|ten roles, a first choice and a margin| fallback[LLM fallback]
    fallback -->|a number and a quote| rules[Rules]
    rules -->|disagreement, family mismatch or no match| agent[Agent]
    rules -->|proposal| people[People]
    agent -->|recommends only| people
    rules -.->|golden picks scored| golden
    people -->|blind labels, never learned| golden
    people -->|an HR SME approves every one| learning[Learning]
    learning --> table([Look-up table and examples])
    rules -.->|answers and routes recorded| statefile[(State file)]
    {_styled(WAVE1_ACTING)}
"""

WAVE2_FLOWCHART = f"""\
flowchart TD
    tail([Long-tail titles]) -->|a name or no usable family code, no model| queue([Review queue])
    tail --> ml_match[ML match]
    ml_match --> fallback[LLM fallback]
    fallback --> rules[Rules]
    rules -->|every check passes| agree([Models agree path])
    rules -->|disagreement, family mismatch or no match| agent[Agent]
    agent -->|its answer goes back through the rules| rules
    rules -->|the agent answer passes| settled([Agent-settled path])
    rules -->|a failed check or an escalation| queue
    agree --> audit[Random audit]
    settled --> audit
    people[People] -->|blind labels| audit
    audit -->|every title audited| labeled([Approved, each with its label])
    audit -->|at most 2 wrong of 60| confirmed([The rest Confirmed])
    audit -->|3 to 12 wrong of 60| provisional([The rest Provisional])
    audit -->|13 or more wrong of 60| stopped([Stopped, the rest kept on legacy titles by default])
    people -->|audit labels| learning[Learning]
    rules -.->|golden titles scored again| golden[Golden set]
    rules -.->|answers and routes recorded| statefile[(State file)]
    {_styled(WAVE2_ACTING)}
"""

CLOSE_FLOWCHART = f"""\
flowchart TD
    queue([Review queue]) --> people[People]
    people -->|a role, Approved| decisions([One final decision per title])
    people -->|decisions join| learning[Learning]
    learning -.->|then| golden[Golden set]
    golden -->|one more pass, no agent| ml_match[ML match]
    ml_match --> fallback[LLM fallback]
    fallback --> rules[Rules]
    rules -.->|golden picks scored| golden
    people -->|could not decide| generic[Generic family role]
    people -->|no suitable role| catalogowner([Catalog owner, legacy title kept])
    generic -->|under the cap, Provisional| decisions
    generic -->|barred or over the cap| legacy([Left on legacy title])
    legacy --> decisions
    catalogowner --> decisions
    decisions --> release[Release]
    decisions --> report[Report]
    release --> files([Load file, reverse file, exceptions list, review queue])
    decisions -.->|recorded| statefile[(State file)]
    {_styled(CLOSE_ACTING)}
"""


@dataclass(frozen=True, slots=True)
class StagePage:
    """One stage of a run, as its page shows it.

    design is its row of the table in DESIGN.md 1.10 (the row design_wave),
    word for word; left_out names what DESIGN.md 2.2 leaves out of the mock
    from that row, if anything. acting are the nodes of the components that
    act in it, the nodes its flowchart styles; its page lists them in the
    order of COMPONENTS.
    """

    slug: str
    stage: Stage
    name: str
    design_wave: str
    design: str
    left_out: str
    acting: tuple[str, ...]
    flowchart: str
    facts: tuple[Fact, ...]


STAGES = (
    StagePage(
        slug="set-up",
        stage=Stage.SETUP,
        name="Set-up",
        design_wave="Set-up",
        design="Catalog approved; release thresholds signed; a small pilot timed; golden set drawn",
        left_out="the set-up pilot",
        acting=SETUP_ACTING,
        flowchart=SETUP_FLOWCHART,
        facts=(
            _TITLE_ROWS,
            Count("Titles on the Wave 1 list", _ON_WAVE1_LIST),
            _GOLDEN_TITLES,
            _SEED_ENTRIES,
            Count("Settings recorded", _ROWS_IN["settings"]),
            Count("Thresholds recorded", _ROWS_IN["thresholds"]),
            _LLM_SETTING,
            _EMBEDDER_SETTING,
        ),
    ),
    StagePage(
        slug="wave-1",
        stage=Stage.WAVE1,
        name="Wave 1",
        design_wave="Wave 1",
        design=(
            "The most widely held titles: the machine proposes, an HR SME approves every "
            "one. The golden set is labeled. The agent only recommends. A dry run scores "
            "every title once and releases nothing, to size the Wave 2 review queue."
        ),
        left_out="the dry run",
        acting=WAVE1_ACTING,
        flowchart=WAVE1_FLOWCHART,
        facts=(
            _LOOKUP_HITS,
            Count("Titles scored by the models", _SCORED, (Stage.WAVE1,)),
            Count("LLM answers", _ANSWERS, (Stage.WAVE1, _CHOOSE)),
            Count("Agent reviews", _ANSWERS, (Stage.WAVE1, _REVIEW)),
            _WAVE1_APPROVALS,
            _GOLDEN_LABELS,
            *_learned(Stage.WAVE1),
        ),
    ),
    StagePage(
        slug="wave-2",
        stage=Stage.WAVE2,
        name="Wave 2",
        design_wave="Wave 2",
        design=(
            "The long tail is scored for release. The two machine paths are audited and "
            "released. Everything else goes to a review queue. From here the agent's answer "
            "stands."
        ),
        left_out="",
        acting=WAVE2_ACTING,
        flowchart=WAVE2_FLOWCHART,
        facts=(
            Count("Titles scored by the models", _SCORED, (Stage.WAVE2,)),
            Count("On the models-agree path", _ROUTED, (Stage.WAVE2, MachinePath.MODELS_AGREE)),
            Count("On the agent-settled path", _ROUTED, (Stage.WAVE2, MachinePath.AGENT_SETTLED)),
            Count("Scored and sent to a person", _ROUTED, (Stage.WAVE2, Route.PERSON)),
            Count("Agent reviews", _ANSWERS, (Stage.WAVE2, _REVIEW)),
            *_AUDITS,
            *_learned(Stage.WAVE2),
        ),
    ),
    StagePage(
        slug="close",
        stage=Stage.CLOSE,
        name="Close",
        design_wave="Wave 3",
        design="People clear the queue; leftovers take a generic family role; final release",
        left_out="",
        acting=CLOSE_ACTING,
        flowchart=CLOSE_FLOWCHART,
        facts=(
            _QUEUE_DECISIONS,
            Count("Given a role", _DECIDED_AS, (pipeline.OCCASION_QUEUE, HumanAnswer.ROLE)),
            _UNDECIDED,
            Count(
                "No suitable role",
                _DECIDED_AS,
                (pipeline.OCCASION_QUEUE, HumanAnswer.NO_SUITABLE_ROLE),
            ),
            *_learned(Stage.CLOSE),
            Count("Golden-set titles scored again", _SCORED, (Stage.CLOSE,)),
            _GENERIC,
            _LEGACY,
            _QUEUE_FILE,
        ),
    ),
)

# The overview: DESIGN.md 1.1, its four lines, and what the page counts on its own.
IDEA = (
    "Machines pick, they never write. Every Short Title and Long Title comes from a catalog "
    "that HR approved. A model only chooses a catalog role by number, so it cannot invent a "
    "title.",
    "Cheapest check first. An exact look-up, then an ML match, then an LLM, then an agent, "
    "then a person.",
    "Two methods must agree. A title goes out without a person only if the LLM and the ML "
    "model agree, or an agent settles their disagreement, and only after HR has audited a "
    "random sample of that machine path.",
    "Every title carries a status, so anyone using it knows how far to rely on it.",
)
OVERVIEW_FACTS = (
    _TITLE_ROWS,
    _ANSWER_LABELS,
    _LLM_SETTING,
    _EMBEDDER_SETTING,
)


# --- Rendering ---------------------------------------------------------------------


class _Html(str):
    """Markup this module built from escaped values: _e passes it through."""

    __slots__ = ()


def _e(value: object) -> _Html:
    """The one escape: every value put into a page passes here."""
    if isinstance(value, _Html):
        return value
    return _Html(html.escape(str(value), quote=True))


def _join(parts: Iterable[object], separator: str = "") -> _Html:
    """The parts, each escaped unless it is markup already, one after another."""
    return _Html(_e(separator).join(_e(part) for part in parts))


def _link(href: str, text: str, *, nav_id: str = "", current: bool = False) -> _Html:
    """A real link that htmx follows into the panel (the body says where)."""
    marks = _Html(f' id="{_e(nav_id)}"') if nav_id else _Html("")
    if current:
        marks = _Html(f'{_e(marks)} aria-current="page"')
    return _Html(f'<a href="{_e(href)}" hx-get="{_e(href)}"{_e(marks)}>{_e(text)}</a>')


def _stage_href(stage: StagePage) -> str:
    return f"/stage/{stage.slug}"


def _component_href(component: Component) -> str:
    return f"/component/{component.slug}"


def _nav(current: str) -> _Html:
    # Each link has an id, so that htmx gives the focus back to the link followed
    # once the navigation is swapped in.
    def items(pages: Iterable[tuple[str, str, str]]) -> _Html:
        return _join(
            _Html(f"<li>{_e(_link(href, name, nav_id=nav_id, current=href == current))}</li>\n")
            for href, name, nav_id in pages
        )

    stages = items((_stage_href(s), s.name, f"nav-stage-{s.slug}") for s in STAGES)
    components = items((_component_href(c), c.name, f"nav-component-{c.slug}") for c in COMPONENTS)
    return _Html(
        '<nav id="nav" aria-label="Stages and components">\n'
        f"<h2>Stages</h2>\n<ul>\n{_e(stages)}</ul>\n"
        f"<h2>Components</h2>\n<ul>\n{_e(components)}</ul>\n"
        "</nav>\n"
    )


def _document(title: str, current: str, panel: _Html) -> bytes:
    page = _Html(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{_e(_Html(HTMX_CONFIG_META))}\n"
        f"<title>{_e(title)} - Architecture of the mock</title>\n"
        '<link rel="stylesheet" href="/static/app.css">\n'
        '<script src="/static/vendor/htmx.min.js" defer></script>\n'
        '<script src="/static/vendor/mermaid.min.js" defer></script>\n'
        '<script src="/static/app.js" defer></script>\n'
        "</head>\n"
        '<body hx-target="#panel" hx-select="#panel" hx-select-oob="#nav" '
        'hx-swap="outerHTML show:window:top" hx-push-url="true">\n'
        '<header class="masthead">\n'
        '<p class="kicker">Job-title standardization mock</p>\n'
        f'<p class="site">{_e(_link("/", "Architecture of a run"))}</p>\n'
        "</header>\n"
        '<div class="layout">\n'
        f"{_e(_nav(current))}"
        f'<main id="panel">\n{_e(panel)}</main>\n'
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
    return page.encode("utf-8")


def _paragraphs(texts: Iterable[str]) -> _Html:
    return _join(_Html(f"<p>{_e(text)}</p>\n") for text in texts)


def _quote(texts: Iterable[str], source: str) -> _Html:
    return _Html(
        f'<blockquote class="design">\n{_e(_paragraphs(texts))}'
        f'<p class="source">{_e(source)}</p>\n</blockquote>\n'
    )


def _value(connection: sqlite3.Connection, fact: Fact) -> str:
    if isinstance(fact, Count):
        return str(_count(connection, fact.sql, fact.params))
    found = [shown for sql, params, shown in fact.options if _count(connection, sql, params)]
    return ", ".join(found) or "none"


# The heading of a table of counts, by whether the run wrote its final decisions.
FINISHED_HEADING = "From this run's state file"
UNFINISHED_HEADING = (
    "From the state file of a run that has not finished, as it stood at its last completed step"
)


def _facts(connection: sqlite3.Connection, facts: Iterable[Fact]) -> _Html:
    rows = _join(
        _Html(
            f'<tr><th scope="row">{_e(fact.label)}</th>'
            f"<td>{_e(_value(connection, fact))}</td></tr>\n"
        )
        for fact in facts
    )
    # The final decisions are written in one step at the end of a run (store.save_run).
    finished = _count(connection, _ROWS_IN["decisions"], ())
    return _Html(
        f"<h2>{_e(FINISHED_HEADING if finished else UNFINISHED_HEADING)}</h2>\n"
        '<table class="counts">\n<tbody>\n'
        f"{_e(rows)}</tbody>\n</table>\n"
    )


def _design_section(number: str, detail: str = "") -> str:
    named = f"{number} {DESIGN_SECTIONS[number]}"
    return f"{named}, {detail}" if detail else named


def _overview(connection: sqlite3.Connection) -> _Html:
    stages = _join(
        _Html(f"<li>{_e(_link(_stage_href(s), s.name))}: {_e(s.design)}</li>\n") for s in STAGES
    )
    run, idea = (f"DESIGN.md {_design_section(number)}" for number in ("2.6", "1.1"))
    return _Html(
        '<p class="kicker">Overview</p>\n'
        "<h1>Architecture of a run</h1>\n"
        f"{_e(_quote([RUN_TEXT], run))}"
        f"<h2>{_e(DESIGN_SECTIONS['1.1'])}</h2>\n"
        f"{_e(_quote(IDEA, idea))}"
        "<h2>Stages</h2>\n"
        f'<ol class="stages">\n{_e(stages)}</ol>\n'
        f"{_e(_facts(connection, OVERVIEW_FACTS))}"
    )


def _stage_panel(connection: sqlite3.Connection, stage: StagePage) -> _Html:
    acting = [c for c in COMPONENTS if c.node in stage.acting]
    listed = _join(_Html(f"<li>{_e(_link(_component_href(c), c.name))}</li>\n") for c in acting)
    in_and_out, row = _design_section("2.2"), _design_section("1.10", f"row {stage.design_wave}")
    left_out = (
        _Html(
            f'<p class="left-out">Left out of the mock (DESIGN.md {_e(in_and_out)}): '
            f"{_e(stage.left_out)}.</p>\n"
        )
        if stage.left_out
        else _Html("")
    )
    return _Html(
        '<p class="kicker">Stage</p>\n'
        f"<h1>{_e(stage.name)}</h1>\n"
        f"{_e(_quote([stage.design], f'DESIGN.md {row}'))}"
        f"{_e(left_out)}"
        '<figure class="flowchart">\n'
        f'<pre class="mermaid">{_e(stage.flowchart)}</pre>\n'
        f"<figcaption>{_e(stage.name)} as a flowchart. The components acting in it are "
        "drawn with a heavy border.</figcaption>\n"
        "</figure>\n"
        f"<h2>Components acting in {_e(stage.name)}</h2>\n"
        f'<ul class="acting">\n{_e(listed)}</ul>\n'
        f"{_e(_facts(connection, stage.facts))}"
    )


def _component_panel(connection: sqlite3.Connection, component: Component) -> _Html:
    stages = [s for s in STAGES if component.node in s.acting]
    acts_in = _join((_link(_stage_href(s), s.name) for s in stages), ", ")
    sections = "; ".join(_design_section(*each) for each in component.design)
    return _Html(
        '<p class="kicker">Component</p>\n'
        f"<h1>{_e(component.name)}</h1>\n"
        "<h2>Its job</h2>\n"
        f"{_e(_quote(component.job, f'DESIGN.md {sections}'))}"
        '<dl class="facts">\n'
        f"<dt>Module</dt><dd><code>{_e(component.module)}</code></dd>\n"
        f"<dt>Record entries</dt><dd>docs/build-decisions.md "
        f"{_e(', '.join(component.entries))}</dd>\n"
        f"<dt>Acts in</dt><dd>{_e(acts_in)}</dd>\n"
        "</dl>\n"
        f"{_e(_facts(connection, component.facts))}"
    )


def _not_found() -> _Html:
    return _Html(
        '<p class="kicker">Not found</p>\n'
        "<h1>No page at this address</h1>\n"
        f"<p>{_e(_link('/', 'Back to the overview'))}</p>\n"
    )


# The pages, by address: (title, panel).
PAGES: dict[str, tuple[str, Callable[[sqlite3.Connection], _Html]]] = {
    "/": ("Overview", _overview),
    **{
        _stage_href(s): (s.name, lambda connection, s=s: _stage_panel(connection, s))
        for s in STAGES
    },
    **{
        _component_href(c): (c.name, lambda connection, c=c: _component_panel(connection, c))
        for c in COMPONENTS
    },
}


# --- The state file ---------------------------------------------------------------


def _count(connection: sqlite3.Connection, sql: str, params: tuple[str | int, ...]) -> int:
    (value,) = connection.execute(sql, params).fetchone()
    return int(value)  # COUNT(*) is a whole number


@contextlib.contextmanager
def _reading(path: Path) -> Iterator[sqlite3.Connection]:
    """A read-only connection to a free run's state file, checked first."""
    if not path.is_file():
        raise ExplorerError(
            f"there is no state file at {path}. Make a free run first: python3 -m titlestd run"
        )
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise ExplorerError(f"{path} cannot be opened: {error}") from error
    with contextlib.closing(connection):
        try:
            made = _count(connection, _TABLES_MADE, TABLES)
            free = _count(connection, _FREE_SETTING, ("llm", *_FREE_LLMS))
            live = _count(connection, _LABELLED, (LABEL_LIVE,))
        except sqlite3.Error as error:
            raise ExplorerError(f"{path} is not a state file of this program: {error}") from error
        if made != len(TABLES):
            raise ExplorerError(f"{path} is not a state file of this program")
        if not free or live:
            raise ExplorerError(
                f"{path} is not the state of a free run; the page shows only a run with "
                f"llm = {' or '.join(_FREE_LLMS)}, whose answers are stand-ins or recorded"
            )
        yield connection


# --- The application ---------------------------------------------------------------

_Response = tuple[str, list[tuple[str, str]], bytes]

_REFUSED_HOST = f"This page answers only at {ADDRESS}\n"
_REFUSED_METHOD = "This page answers only GET.\n"
_NO_FILE = "This file is missing from the static folder.\n"
_NO_STATE = (
    "The state file is missing, cannot be read or is not the state of a free run. "
    "Make a free run with python3 -m titlestd run, then load the page again.\n"
)


def _response(status: str, body: bytes, content_type: str, *extra: tuple[str, str]) -> _Response:
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
        ("Cache-Control", "no-store"),
        *extra,
    ]
    return status, headers, body


def _text(status: str, text: str, *extra: tuple[str, str]) -> _Response:
    return _response(status, text.encode("utf-8"), "text/plain; charset=utf-8", *extra)


def _answer(state_path: Path, environ: dict) -> _Response:
    if environ.get("HTTP_HOST") != f"{HOST}:{PORT}":
        return _text("403 Forbidden", _REFUSED_HOST)
    if environ.get("REQUEST_METHOD") != "GET":
        return _text("405 Method Not Allowed", _REFUSED_METHOD, ("Allow", "GET"))
    path = environ.get("PATH_INFO", "")
    if path in STATIC_FILES:
        name, content_type = STATIC_FILES[path]
        try:
            return _response("200 OK", (STATIC_DIR / name).read_bytes(), content_type)
        except OSError:
            return _text("404 Not Found", _NO_FILE)
    page = PAGES.get(path)
    try:
        with _reading(state_path) as connection:
            if page is None:
                status, title, panel = "404 Not Found", "Not found", _not_found()
            else:
                title, render = page
                status, panel = "200 OK", render(connection)
    except (ExplorerError, sqlite3.Error):
        return _text("503 Service Unavailable", _NO_STATE)
    return _response(status, _document(title, path, panel), "text/html; charset=utf-8")


def make_app(state_path: Path) -> Callable:
    """The WSGI application serving the state file at state_path.

    Raises ExplorerError at once when the file is not a free run's state.
    """
    state_path = Path(state_path)
    with _reading(state_path):
        pass

    def application(environ: dict, start_response: Callable) -> list[bytes]:
        status, headers, body = _answer(state_path, environ)
        start_response(status, headers)
        return [body]

    return application


class ThreadingServer(socketserver.ThreadingMixIn, WSGIServer):
    """wsgiref's server, taking each request in a thread of its own (Decided here)."""

    daemon_threads = True


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the state file of the output folder the settings give until Ctrl+C."""
    parser = argparse.ArgumentParser(
        prog="python3 -m titlestd.explorer",
        description=f"Serve the state file of a free run, read-only, at {ADDRESS}. It takes "
        "the settings of every command and uses only --out-dir, the folder of the run.",
    )
    add_setting_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        path = settings_from_arguments(arguments).out_dir / store.STATE_FILE
    except SettingsError as error:
        sys.stderr.write(f"titlestd.explorer: {error}\n")
        return 2
    try:
        application = make_app(path)
    except ExplorerError as error:
        sys.stderr.write(f"titlestd.explorer: {error}\n")
        return 1
    try:
        server = make_server(HOST, PORT, application, server_class=ThreadingServer)
    except OSError as error:
        sys.stderr.write(f"titlestd.explorer: cannot listen at {ADDRESS}: {error}\n")
        return 1
    with server:
        print(f"Serving {path}, read-only, at {ADDRESS} (stop with Ctrl+C)", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
