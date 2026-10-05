"""The state file: a local SQLite database, recreated on every run.

Owner: feat/pipeline-report-cli (stage 4).
Governed by: docs/design.md 2.1 ("a local SQLite file or plain files for
state"); docs/build-decisions.md Q06, Q38.

The state file is out/state.sqlite (settings.out_dir / STATE_FILE), made with
the standard library's sqlite3. It is deleted and recreated at the start of
every run and loaded with only the seed look-up, so a run never depends on an
earlier one.

Components do not read or write the database: they exchange the plain types of
titlestd.contracts in memory, and the pipeline writes the state here as the
run proceeds (title rows, the look-up table and examples, every model answer
with its label, people's decisions with their simulated mark, the golden set,
the audits and the final decisions). Tests of the eight section 2.7 rules read
it back. The schema belongs to this module and is not a contract for any
other.

Decided here, where the record is silent:
- One function per moment of a run. save_setup records the settings and the
  thresholds in force before any scoring (Q37), with the title rows and the
  two lists drawn in set-up. save_scoring records what the models said about
  one title at one stage and where the rules sent it. save_people records
  people's decisions, save_knowledge the look-up table and the examples after
  each learning step, save_run what RunResult holds at the end. So save_run
  takes no settings: they were recorded in set-up, before any result existed.
- A title is identified by the two fields of its key, family_code and
  cleaned_title, in every table. A stage is written as contracts.Stage has it.
- What a model was shown is recorded beside what it answered: shown_roles
  holds every role put in front of the LLM or the agent, in a numbered list
  or in a search result, and ml_shortlist every role the ML model listed
  with its score and the text that matched (1.5). No generic family role may
  ever be in either (1.2); a test reads both.
- The look-up table and the examples each say at which stage a title joined
  them. A title joins once: a later call adds only what is new.
- Every function commits what it wrote, so a run that stops leaves the state
  as it stood at the last completed step.
- What a model gave is recorded as it was given, and recording it never
  stops a run: shared code decides what an answer means (2.5), the state
  only keeps it. A number SQLite cannot hold as a whole number (one too
  large, or something that is no whole number at all) is kept as its text;
  the number column of model_answers has no declared type, so that SQLite
  does not turn such a text into a number of its own. A quote, detail or
  search query that is not valid Unicode is kept with the offending
  characters escaped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from titlestd.contracts import (
    AgentResult,
    ChooseAnswer,
    HumanDecision,
    MatchResult,
    NumberedRole,
    Reason,
    ReviewAnswer,
    RuleOutcome,
    RunResult,
    Stage,
    TitleKey,
    TitleRow,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings

STATE_FILE = "state.sqlite"

# The answer steps, as model_answers and shown_roles name them.
STEP_CHOOSE = "choose"
STEP_REVIEW = "review"

_SCHEMA = """
CREATE TABLE settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE thresholds (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE title_rows (
    family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL, headcount INTEGER NOT NULL,
    department TEXT NOT NULL, description TEXT NOT NULL,
    has_person_name INTEGER NOT NULL, family_usable INTEGER NOT NULL,
    on_wave1_list INTEGER NOT NULL, in_golden_set INTEGER NOT NULL,
    PRIMARY KEY (family_code, cleaned_title));
CREATE TABLE lookup (
    family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL, role_id TEXT NOT NULL,
    joined_at TEXT NOT NULL, PRIMARY KEY (family_code, cleaned_title));
CREATE TABLE examples (
    family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL, role_id TEXT NOT NULL,
    joined_at TEXT NOT NULL, PRIMARY KEY (family_code, cleaned_title));
CREATE TABLE scorings (
    stage TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    ml_first_choice TEXT, ml_margin REAL NOT NULL, ml_examples INTEGER NOT NULL,
    route TEXT NOT NULL, reasons TEXT NOT NULL,
    PRIMARY KEY (stage, family_code, cleaned_title));
CREATE TABLE ml_shortlist (
    stage TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    place INTEGER NOT NULL, role_id TEXT NOT NULL, score REAL NOT NULL,
    matched_text TEXT NOT NULL, PRIMARY KEY (stage, family_code, cleaned_title, place));
CREATE TABLE model_answers (
    stage TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    step TEXT NOT NULL, answer TEXT NOT NULL, number, quote TEXT NOT NULL,
    source TEXT NOT NULL, detail TEXT NOT NULL,
    role_id TEXT, route TEXT NOT NULL, reasons TEXT NOT NULL,
    PRIMARY KEY (stage, family_code, cleaned_title, step));
CREATE TABLE shown_roles (
    stage TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    step TEXT NOT NULL, shown_in TEXT NOT NULL, query TEXT NOT NULL,
    place INTEGER NOT NULL, number INTEGER, role_id TEXT NOT NULL);
CREATE TABLE human_decisions (
    occasion TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    answer TEXT NOT NULL, role_id TEXT, decided_by TEXT NOT NULL, simulated INTEGER NOT NULL,
    PRIMARY KEY (occasion, family_code, cleaned_title));
CREATE TABLE golden_scores (
    stage TEXT PRIMARY KEY, size INTEGER NOT NULL, ml_right INTEGER NOT NULL,
    llm_right INTEGER NOT NULL, llm_no_answer INTEGER NOT NULL);
CREATE TABLE audits (
    path TEXT PRIMARY KEY, path_size INTEGER NOT NULL, audited INTEGER NOT NULL,
    wrong INTEGER NOT NULL, outcome TEXT NOT NULL);
CREATE TABLE audit_samples (
    path TEXT NOT NULL, family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL,
    PRIMARY KEY (path, family_code, cleaned_title));
CREATE TABLE decisions (
    family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL, table_row TEXT NOT NULL,
    source TEXT NOT NULL, role_id TEXT, status TEXT, decided_by TEXT NOT NULL,
    simulated INTEGER NOT NULL, error_rate REAL, legacy_reason TEXT,
    PRIMARY KEY (family_code, cleaned_title));
CREATE TABLE review_queue (
    family_code TEXT NOT NULL, cleaned_title TEXT NOT NULL, reasons TEXT NOT NULL,
    answer_source TEXT NOT NULL, outcome TEXT NOT NULL,
    PRIMARY KEY (family_code, cleaned_title));
CREATE TABLE answer_labels (label TEXT PRIMARY KEY);
"""


def create_store(path: Path) -> sqlite3.Connection:
    """Delete any file at path and create the empty schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    # The file is rebuilt on every run, so a commit need not wait for the disk.
    connection.execute("PRAGMA synchronous = OFF")
    with connection:
        connection.executescript(_SCHEMA)
    return connection


def save_setup(
    connection: sqlite3.Connection,
    settings: Settings,
    thresholds: Mapping[str, object],
    rows: Sequence[TitleRow],
    wave1_keys: Iterable[TitleKey],
    golden_keys: Iterable[TitleKey],
) -> None:
    """Record what set-up fixes before any scoring (Q37): the settings and
    thresholds in force, the title rows, the Wave 1 list and the golden set."""
    wave1, golden = set(wave1_keys), set(golden_keys)
    with connection:
        connection.executemany(
            "INSERT INTO settings VALUES (?, ?)",
            [(name, _text(value)) for name, value in settings.as_dict().items()],
        )
        connection.executemany(
            "INSERT INTO thresholds VALUES (?, ?)",
            [(name, _text(value)) for name, value in thresholds.items()],
        )
        connection.executemany(
            "INSERT INTO title_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    *row.key,
                    row.headcount,
                    row.record.department,
                    row.record.description,
                    row.has_person_name,
                    row.family_usable,
                    row.key in wave1,
                    row.key in golden,
                )
                for row in rows
            ],
        )


def save_knowledge(connection: sqlite3.Connection, stage: Stage, knowledge: Knowledge) -> None:
    """Record the look-up entries and the examples that joined since the last call."""
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO lookup VALUES (?, ?, ?, ?)",
            [(*key, role_id, stage.value) for key, role_id in knowledge.entries()],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO examples VALUES (?, ?, ?, ?)",
            [(*each.key, each.role_id, stage.value) for each in knowledge.examples()],
        )


def save_scoring(
    connection: sqlite3.Connection,
    stage: Stage,
    key: TitleKey,
    *,
    match: MatchResult,
    route: str,
    reasons: Sequence[Reason],
    shown: Sequence[NumberedRole] = (),
    answer: ChooseAnswer | None = None,
    outcome: RuleOutcome | None = None,
    review: AgentResult | None = None,
    review_outcome: RuleOutcome | None = None,
) -> None:
    """Record what the models said about one title at one stage.

    match is the ML model's answer; margin and example count are kept for
    every title (Q04). shown, answer and outcome are the LLM's numbered list,
    its answer and what the rules made of it, and are left out on the ML-alone
    route. review and review_outcome are the agent's, when the title was
    referred to it. route says where the title went in the end.
    """
    at = (stage.value, *key)
    shown_roles = [(*at, STEP_CHOOSE, "numbered list", "", *each) for each in _places(shown)]
    answers = []
    if answer is not None and outcome is not None:
        answers.append((*at, STEP_CHOOSE, *_answer(answer), *_ruled(outcome)))
    if review is not None and review_outcome is not None:
        answers.append((*at, STEP_REVIEW, *_answer(review.answer), *_ruled(review_outcome)))
        case = review.transcript.case
        shown_roles += [(*at, STEP_REVIEW, "case", "", *each) for each in _places(case.roles)]
        for step in review.transcript.steps:
            asked = (step.request.tool.value, _storable(step.request.query))
            shown_roles += [
                (*at, STEP_REVIEW, *asked, *each) for each in _places(hit.role for hit in step.hits)
            ]
    with connection:
        connection.execute(
            "INSERT INTO scorings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *at,
                match.first_choice,
                match.margin,
                match.first_choice_examples,
                route,
                _joined(reasons),
            ),
        )
        connection.executemany(
            "INSERT INTO ml_shortlist VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (*at, place, scored.role_id, scored.score, scored.matched_text)
                for place, scored in enumerate(match.shortlist, start=1)
            ],
        )
        connection.executemany(
            "INSERT INTO model_answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", answers
        )
        connection.executemany(
            "INSERT INTO shown_roles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", shown_roles
        )


def save_people(
    connection: sqlite3.Connection, occasion: str, decisions: Iterable[HumanDecision]
) -> None:
    """Record people's decisions made on one occasion, each with its simulated mark (Q27)."""
    with connection:
        connection.executemany(
            "INSERT INTO human_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    occasion,
                    *each.key,
                    each.answer.value,
                    each.role_id,
                    each.decided_by,
                    each.simulated,
                )
                for each in decisions
            ],
        )


def save_run(connection: sqlite3.Connection, result: RunResult) -> None:
    """Write what a finished run holds: the final decisions, the audits with
    their samples, the golden-set scores, the review queue and the answer labels."""
    with connection:
        connection.executemany(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    *each.key,
                    each.row.value,
                    each.source.value,
                    each.role_id,
                    each.status.value if each.status else None,
                    each.decided_by,
                    each.simulated,
                    each.error_rate,
                    each.legacy_reason.value if each.legacy_reason else None,
                )
                for each in result.decisions
            ],
        )
        connection.executemany(
            "INSERT INTO audits VALUES (?, ?, ?, ?, ?)",
            [
                (each.path.value, each.path_size, each.audited, each.wrong, each.outcome.value)
                for each in result.audits
            ],
        )
        connection.executemany(
            "INSERT INTO audit_samples VALUES (?, ?, ?)",
            [(each.path.value, *key) for each in result.audits for key in each.audited_keys],
        )
        connection.executemany(
            "INSERT INTO golden_scores VALUES (?, ?, ?, ?, ?)",
            [
                (each.stage.value, each.size, each.ml_right, each.llm_right, each.llm_no_answer)
                for each in result.golden_scores
            ],
        )
        connection.executemany(
            "INSERT INTO review_queue VALUES (?, ?, ?, ?, ?)",
            [
                (*each.row.key, _joined(each.reasons), each.answer_source, each.outcome.value)
                for each in result.queue
            ],
        )
        connection.executemany(
            "INSERT INTO answer_labels VALUES (?)", [(label,) for label in result.answer_labels]
        )


def _places(roles: Iterable[NumberedRole]) -> list[tuple[int, int | None, str]]:
    """(place, number, role_id) of each role, in the order shown."""
    return [(place, each.number, each.role.role_id) for place, each in enumerate(roles, start=1)]


def _answer(answer: ChooseAnswer | ReviewAnswer) -> tuple[object, ...]:
    """The fields a ChooseAnswer and a ReviewAnswer share, before the role."""
    given = (_storable(answer.number), _storable(answer.quote))
    return (answer.answer.value, *given, answer.source, _storable(answer.detail))


def _storable(value: object) -> int | str | None:
    """What a model gave as a number, a quote or a query, in a form SQLite takes."""
    if value is None or (type(value) is int and -(2**63) <= value < 2**63):
        return value
    text = value if isinstance(value, str) else repr(value)
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _ruled(outcome: RuleOutcome) -> tuple[str | None, str, str]:
    return (outcome.role_id, outcome.route.value, _joined(outcome.reasons))


def _joined(reasons: Iterable[Reason]) -> str:
    return "; ".join(reason.value for reason in reasons)


def _text(value: object) -> str:
    """A setting or threshold as text: None is empty, a flag is 1 or 0."""
    if value is None:
        return ""
    return str(int(value)) if isinstance(value, bool) else str(value)
