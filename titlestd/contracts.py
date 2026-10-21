"""Shared types: the plain data every component exchanges.

This file is the contract between the modules of the package. It holds data
classes, enumerations, the three interfaces behind the model switch and a few
fixed numbers. It has no logic beyond small derived properties and imports
nothing from the rest of the package. A type that crosses a module boundary is
defined here; a type only one module uses stays in that module.

Most modules read it, so a change here reaches nearly all of them. Section numbers
refer to DESIGN.md, Q numbers to docs/build-decisions.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple, Protocol

# --- Numbers the design fixes (they are not settings) -------------------------

SHORTLIST_SIZE = 10  # 1.6 step 3: a shortlist of ten roles
ML_TOP_N = 3  # 1.7: "any of the ML model's top three"
DESCRIPTION_WORDS = 200  # 1.4: the first 200 words of the job description
AGENT_MAX_SEARCHES = 4  # 2.4: at most four search calls
AUDIT_SAMPLE_SIZE = 60  # 2.4: sample 60 titles per path
AUDIT_ACCURACY_BAR = 0.90  # 1.9: the accuracy bar is 90 in 100
AUDIT_PASS_CHANCE = 0.10  # 2.4: a path exactly at the bar passes at most one time in ten
AUDIT_STOP_SHARE = 0.20  # 2.4: stopped above 20% wrong
QUOTE_MIN_CHARS = 3  # Q15
QUOTE_MAX_WORDS = 25  # Q15
NAME_MASK = "[NAME]"  # Q14
EMAIL_MASK = "[EMAIL]"  # Q14

# --- Answer labels (2.5, Q40) -------------------------------------------------

LABEL_STAND_IN = "stand-in"
LABEL_LIVE = "live"


def recorded_label(date: str, model: str) -> str:
    """The label of a replayed answer: the UTC day of the call and the configured model."""
    return f"recorded {date}, {model}"


# --- Optional libraries (2.5, Q38, S09) ---------------------------------------


def run_with_extra(extra: str) -> str:
    """The command that installs an optional extra and runs the program with it.

    extra is "live" or "local". A provider whose library is missing stops with
    a message holding this text, followed by the command and settings to run.
    The package is not installed and a bare python3 never sees what uv puts
    in .venv, so the command has to install and run in one step (S09).
    """
    return f"uv run --extra {extra} python -m titlestd"


class TitlestdError(Exception):
    """An error whose message is meant for the person running the program.

    The command line prints the message and exits non-zero. Components raise a
    subclass (settings, data files, a replay miss, a live-mode stop).
    """


# --- Rows of the data files (2.3) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class CatalogRole:
    """One catalog role. A generic family role has is_generic set (1.2)."""

    role_id: str
    family_code: str
    role_name: str
    short_name: str
    definition: str
    compliance_flag: bool = False
    is_generic: bool = False


@dataclass(frozen=True, slots=True)
class LadderLevel:
    """One level of the ladder with its long and short level word (1.3)."""

    level: str
    long_word: str
    short_word: str


@dataclass(frozen=True, slots=True)
class GradeLevel:
    """One row of HR's grade-to-level table."""

    grade: int
    level: str


@dataclass(frozen=True, slots=True)
class Abbreviation:
    short_form: str
    long_form: str


@dataclass(frozen=True, slots=True)
class Employee:
    """One employee record as received. grade is None when the file has none."""

    employee_id: str
    job_title: str
    job_family_code: str
    department: str
    grade: int | None
    job_description: str


@dataclass(frozen=True, slots=True)
class SeedLookupRow:
    """A title a person has already decided, keyed as recorded (raw title, Q09)."""

    job_title: str
    job_family_code: str
    role_id: str


@dataclass(frozen=True, slots=True)
class Catalog:
    """The catalog as read: roles, the level ladder in file order, the grade table.

    Plain data. The functions that answer questions about it live in
    titlestd.catalog.
    """

    roles: tuple[CatalogRole, ...]
    ladder: tuple[LadderLevel, ...]
    grade_levels: tuple[GradeLevel, ...]


@dataclass(frozen=True, slots=True)
class InputData:
    """Everything the program reads, except the truth file (2.4, Q22)."""

    catalog: Catalog
    abbreviations: tuple[Abbreviation, ...]
    employees: tuple[Employee, ...]
    seed_lookup: tuple[SeedLookupRow, ...]


@dataclass(frozen=True, slots=True)
class StandardTitle:
    """An employee's standard titles. level is None when there is no usable grade."""

    level: str | None
    long_title: str
    short_title: str


# --- The unit of decision (1.4, Q09, Q11, Q14) --------------------------------


class TitleKey(NamedTuple):
    """One cleaned title within a family: the unit of decision.

    Sorting keys gives the canonical processing order, family code then cleaned
    title (Q18). The two fields, in this order, are also the item key of every
    seeded draw for a title (Q06).

    family_code is the recorded code with surrounding whitespace removed; a
    blank or unknown code is kept as it is (Q26). cleaned_title is the title
    after cleaning and name masking (Q14).
    """

    family_code: str
    cleaned_title: str


@dataclass(frozen=True, slots=True)
class Record:
    """What a model, a rule or a reviewer may read about a title row (Q11).

    Built once by the prepare step. department and description are masked;
    description holds at most DESCRIPTION_WORDS words and is empty when no
    employee on the row has one.
    """

    cleaned_title: str
    family_code: str
    department: str
    description: str


@dataclass(frozen=True, slots=True)
class TitleRow:
    """One decision row: a cleaned title within a family, with its employees."""

    record: Record
    employee_ids: tuple[str, ...]  # sorted
    raw_titles: tuple[str, ...]  # the distinct job_title values as received, sorted
    has_person_name: bool = False  # 1.4: to a person and to no model
    family_usable: bool = True  # 2.4: no usable family code goes to a person

    @property
    def key(self) -> TitleKey:
        return TitleKey(self.record.family_code, self.record.cleaned_title)

    @property
    def headcount(self) -> int:
        return len(self.employee_ids)

    @property
    def model_eligible(self) -> bool:
        """False when no model may see the row (Q14, Q26)."""
        return self.family_usable and not self.has_person_name


# --- The ML match (1.6 step 3, 2.4) -------------------------------------------

# A sparse embedding: index to weight, L2-normalised. A dense vector is the
# mapping of every index. Cosine similarity is then a dot product (Q12).
Vector = Mapping[int, float]


@dataclass(frozen=True, slots=True)
class ScoredRole:
    """A shortlisted role with its score and the text that produced it."""

    role_id: str
    score: float  # best cosine across the role's catalog text and approved titles
    matched_text: str = ""  # that catalog text or approved title, for the audit trail


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The ML model's answer for one title row.

    shortlist is ordered best first, ties by role_id, and holds SHORTLIST_SIZE
    roles (fewer only when the catalog has fewer). It never holds a generic
    family role.
    """

    shortlist: tuple[ScoredRole, ...]
    first_choice_examples: int = 0  # human-approved examples of the first choice

    @property
    def first_choice(self) -> str | None:
        return self.shortlist[0].role_id if self.shortlist else None

    @property
    def margin(self) -> float:
        """The gap between first and second; 0.0 with fewer than two roles."""
        if len(self.shortlist) < 2:
            return 0.0
        return self.shortlist[0].score - self.shortlist[1].score

    @property
    def top_three(self) -> tuple[str, ...]:
        return tuple(scored.role_id for scored in self.shortlist[:ML_TOP_N])


@dataclass(frozen=True, slots=True)
class Example:
    """A human-approved title that the ML model matches against (1.6 step 10).

    It keeps the title's record, so its vector is made from the same text the
    title was queried with (Q12).
    """

    record: Record
    role_id: str

    @property
    def key(self) -> TitleKey:
        return TitleKey(self.record.family_code, self.record.cleaned_title)


# --- What the LLM and the agent answer (1.6 steps 4 and 6, Q15, Q20) ----------


@dataclass(frozen=True, slots=True)
class NumberedRole:
    """A role as shown to a model.

    number is what the model may answer with. It is None for a role that is
    shown but cannot be picked (the agent sees out-of-family roles unnumbered).
    Prompts are rendered by shared code and never show a role_id.
    """

    number: int | None
    role: CatalogRole


class ChooseKind(StrEnum):
    ROLE = "role"
    NO_MATCH = "no_match"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID = "invalid"  # no usable answer: refusal, cut-off, failed validation (Q39)


@dataclass(frozen=True, slots=True)
class ChooseAnswer:
    """The LLM fallback's answer, before any check.

    number is set only with ROLE. quote is required with ROLE, NO_MATCH and
    INSUFFICIENT_DATA and is checked by shared code. source is one of the
    answer labels. detail says why an INVALID answer is invalid.
    """

    answer: ChooseKind
    number: int | None
    quote: str
    source: str
    detail: str = ""


class ReviewKind(StrEnum):
    ROLE = "role"
    ESCALATE = "escalate"
    INVALID = "invalid"  # no usable answer; shared code sends the title to a person (Q20)


@dataclass(frozen=True, slots=True)
class ReviewAnswer:
    """The agent's final answer, before any check. ESCALATE needs no quote."""

    answer: ReviewKind
    number: int | None
    quote: str
    source: str
    detail: str = ""


class SearchTool(StrEnum):
    """The agent's two read-only searches. The values are the tool names."""

    CATALOG = "search_catalog"
    APPROVED_TITLES = "search_approved_titles"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One search the agent asks for: a tool and one free-text query (Q10)."""

    tool: SearchTool
    query: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One search result as shown to the agent.

    role carries the agent's number, or None for an out-of-family role. For an
    approved-title hit, approved_title and title_family_code describe the
    similar title that maps to the role; they are empty for a catalog hit.
    """

    role: NumberedRole
    approved_title: str = ""
    title_family_code: str = ""


@dataclass(frozen=True, slots=True)
class SearchStep:
    request: SearchRequest
    hits: tuple[SearchHit, ...]


class Route(StrEnum):
    """Where the rules send a proposal (1.7)."""

    ACCEPT = "accept"  # every check passed
    AGENT = "agent"
    PERSON = "person"


class Reason(StrEnum):
    """Why a title goes to the agent or to a person. The values are shown in the
    review-queue file."""

    # 1.7, on failure a person
    ROLE_NOT_IN_CATALOG = "role not in catalog"
    QUOTE_NOT_IN_RECORD = "quote not in record"
    COMPLIANCE_FLAG = "compliance-flagged role"
    INSUFFICIENT_DATA = "insufficient data"
    # 1.7, on failure the agent (a person when the agent's own answer fails)
    MODELS_DISAGREE = "models disagree"
    FAMILY_MISMATCH = "role outside recorded family"
    NO_MATCH = "no match"
    # after the agent
    AGENT_ESCALATED = "agent escalated"
    INVALID_ANSWER = "invalid model answer"  # Q20, Q39
    # routed before any model
    PERSON_NAME = "person's name"  # Q14
    NO_USABLE_FAMILY = "no usable family code"  # Q26
    ML_ALONE = "ML alone"  # Q04
    # marked by a person
    NO_SUITABLE_ROLE = "no suitable role, waiting for catalog owner"  # Q03


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """The result of checking one proposal against the rules of 1.7.

    reasons lists every failed check in the order of the 1.7 table and is empty
    when route is ACCEPT. role_id is the role the answer resolved to, or None.
    """

    route: Route
    reasons: tuple[Reason, ...] = ()
    role_id: str | None = None


@dataclass(frozen=True, slots=True)
class AgentCase:
    """What the agent is given (Q10).

    roles are the shortlisted roles in the order the LLM saw them, carrying the
    agent's numbers: in-family roles numbered from 1 in that order, the others
    None. llm_role_id and ml_first_choice mark the two earlier opinions.
    """

    record: Record
    roles: tuple[NumberedRole, ...]
    llm_answer: ChooseAnswer
    llm_role_id: str | None
    ml_first_choice: str | None
    reasons: tuple[Reason, ...]


@dataclass(frozen=True, slots=True)
class AgentTranscript:
    """The case and every search run so far, oldest first.

    This is all an agent backend sees. Shared code owns the loop, the search
    cap and the numbering (2.5, Q20).
    """

    case: AgentCase
    steps: tuple[SearchStep, ...] = ()

    @property
    def searches_left(self) -> int:
        return max(0, AGENT_MAX_SEARCHES - len(self.steps))

    @property
    def numbered_roles(self) -> tuple[NumberedRole, ...]:
        """Every role the agent may pick, by number, each once."""
        shown = list(self.case.roles) + [hit.role for step in self.steps for hit in step.hits]
        by_number = {role.number: role for role in shown if role.number is not None}
        return tuple(by_number[number] for number in sorted(by_number))


@dataclass(frozen=True, slots=True)
class AgentResult:
    """What the agent loop returns for one case.

    answer is the backend's final answer, or an INVALID one made by the loop
    when the backend broke the protocol. role_id is the role its number mapped
    to, or None. transcript holds every search that ran.
    """

    answer: ReviewAnswer
    role_id: str | None
    transcript: AgentTranscript

    @property
    def confirms_llm(self) -> bool:
        """True when the agent picked the role the LLM had picked (1.6 step 6)."""
        return self.role_id is not None and self.role_id == self.transcript.case.llm_role_id


# --- The three calls behind the switch (2.5) ----------------------------------


class Embedder(Protocol):
    """embed(texts): one L2-normalised vector per text, in order.

    The same text always gives the same vector, in every process.
    """

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


class Chooser(Protocol):
    """choose(record, ten roles): the LLM fallback.

    roles are numbered from 1 in the order to show them. The answer is a
    number, "no match" or "insufficient data", plus a quote (Q15). Shared code
    maps the number to a role and checks the quote.
    """

    def choose(self, record: Record, roles: Sequence[NumberedRole]) -> ChooseAnswer: ...


class Reviewer(Protocol):
    """review(case, tools): the agent's backend, one turn at a time (Q20).

    The loop in titlestd.agent calls next_turn until it returns a final
    answer. A turn is either one search request or the final answer. After
    AGENT_MAX_SEARCHES searches the turn must be the final answer.
    """

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer: ...


def _nothing_to_finish() -> None:
    return None


@dataclass(frozen=True, slots=True)
class Llm:
    """The chooser and the reviewer of one llm setting.

    finish is called once by the pipeline after a run that completed; live
    mode completes its recording there (Q29).
    """

    chooser: Chooser
    reviewer: Reviewer
    finish: Callable[[], None] = _nothing_to_finish


@dataclass(frozen=True, slots=True)
class Models:
    """What the switch returns for the settings in force."""

    embedder: Embedder
    llm: Llm


# --- People (1.8, 2.4, Q01, Q03, Q27) -----------------------------------------


class HumanAnswer(StrEnum):
    ROLE = "role"
    UNDECIDED = "could not decide"  # Q01: review-queue decisions at the close only
    NO_SUITABLE_ROLE = "no suitable role"  # Q03


@dataclass(frozen=True, slots=True)
class HumanDecision:
    """A person's decision on one title. In the mock every one is simulated."""

    key: TitleKey
    answer: HumanAnswer
    role_id: str | None  # set only with ROLE; never a generic family role
    decided_by: str  # a persona, for example "SME 1 (simulated)"
    simulated: bool


# --- Outcomes (1.9, 2.4, 2.6) -------------------------------------------------


class Stage(StrEnum):
    """The stages of one run (2.6)."""

    SETUP = "set-up"
    WAVE1 = "Wave 1"
    WAVE2 = "Wave 2"
    CLOSE = "close"


class Status(StrEnum):
    APPROVED = "Approved"  # a named person decided it
    CONFIRMED = "Confirmed"  # no person read it; its path passed its audit
    PROVISIONAL = "Provisional"  # a path that missed the bar, or a generic role


class MachinePath(StrEnum):
    """How a title was decided without a person. There are two paths (1.2)."""

    MODELS_AGREE = "models agree"
    AGENT_SETTLED = "agent-settled"


class TableRow(StrEnum):
    """The rows of the printed table, in order. Every title is in exactly one (Q16)."""

    LOOKUP = "look-up hit"
    WAVE1 = "approved in Wave 1"
    MODELS_AGREE = "models agree"
    AGENT_SETTLED = "agent-settled"
    REVIEW_QUEUE = "review queue"
    GENERIC = "generic family role"
    LEGACY = "left on legacy title"


# Who decided a look-up hit, in place of a persona (S11). Within one run only
# the seed look-up can be hit, and that file names no person.
LOOKUP_DECIDED_BY = "Seed look-up (mock data)"


class DecisionSource(StrEnum):
    """Who or what chose the role of a decision."""

    LOOKUP = "look-up"  # an entry of the look-up table: a person's earlier decision
    PERSON = "person"  # Wave 1 approval, blind label or review-queue decision
    MACHINE = "machine"  # a machine path, released on that path's audit
    CODE = "code"  # the generic family role, assigned at the close
    NONE = "none"  # nobody: the title stays on its legacy title


class ExceptionReason(StrEnum):
    """Why an employee is on the exceptions list (Q05)."""

    STOPPED_PATH = "stopped path"
    OVER_GENERIC_CAP = "over the generic cap"
    UNDECIDED_BARRED = "undecided widely held or flagged title"
    UNDECIDED_UNSCORED = "undecided title with a person's name or no usable family"
    NO_SUITABLE_ROLE = "no suitable role"
    NO_GRADE = "no grade"  # the employee has a new title, with no level word


@dataclass(frozen=True, slots=True)
class Decision:
    """The final outcome for one title row. One per row, whatever happened.

    row says where the title is counted in the table. role_id and status are
    None exactly when row is LEGACY, and legacy_reason is set exactly then.
    decided_by names the persona when source is PERSON, is LOOKUP_DECIDED_BY
    when source is LOOKUP and is empty otherwise, so every Approved decision
    says who decided it (1.9, S11). simulated is True on every simulated
    person's decision and on nothing else. error_rate is the path's
    wrong/audited share and is set only when status is PROVISIONAL because an
    audited path missed the bar.
    """

    key: TitleKey
    row: TableRow
    source: DecisionSource
    role_id: str | None = None
    status: Status | None = None
    decided_by: str = ""
    simulated: bool = False
    error_rate: float | None = None
    legacy_reason: ExceptionReason | None = None


class AuditOutcome(StrEnum):
    """The result of auditing one machine path (1.9, 2.4)."""

    CONFIRMED = "Confirmed"
    PROVISIONAL = "Provisional"
    STOPPED = "Stopped"
    ALL_LABELED = "all labeled"  # AUDIT_SAMPLE_SIZE titles or fewer: no path status


@dataclass(frozen=True, slots=True)
class AuditResult:
    """One path's audit: who was labeled, how many were wrong, the outcome.

    wrong counts audited titles whose machine role differs from the SME's
    label (2.4).
    """

    path: MachinePath
    path_size: int
    audited_keys: tuple[TitleKey, ...]  # sorted
    wrong: int
    outcome: AuditOutcome

    @property
    def audited(self) -> int:
        return len(self.audited_keys)

    @property
    def error_rate(self) -> float | None:
        return self.wrong / self.audited if self.audited else None


@dataclass(frozen=True, slots=True)
class GoldenScore:
    """Golden-set accuracy at one stage, for the footer (Q08)."""

    stage: Stage
    size: int  # every golden title
    ml_right: int  # the ML model's first choice equals the label
    llm_right: int  # the LLM's pick equals the label
    llm_no_answer: int  # "no match" or "insufficient data": counted as wrong


@dataclass(frozen=True, slots=True)
class Leftover:
    """A review-queue title nobody decided, with what bars it from a generic role (Q25)."""

    row: TitleRow
    on_wave1_list: bool = False
    compliance_fired: bool = False  # the compliance check of 1.7 fired at any point
    no_suitable_role: bool = False  # marked by a person; waits for the catalog owner (Q03)


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """One title that entered the review queue, as it stood after Wave 2 (Q16)."""

    row: TitleRow
    reasons: tuple[Reason, ...]
    ml_first_choice: str | None  # role_id
    llm_pick: str | None  # role_id
    agent_recommendation: str | None  # role_id
    answer_source: str  # the answer label of the model answers shown
    decision: HumanDecision | None  # the person's decision at the close
    outcome: TableRow  # the table row the title ended in


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything a finished run produced, for the report, the state file and tests.

    The report is rendered from this alone. It reads no data file, so the two
    things it prints that the decisions do not hold are carried here: the
    Wave 1 list and the count of employees with no grade (Q05, Q16).
    """

    rows: tuple[TitleRow, ...]
    employee_count: int
    # Employees with no grade who received a new title, with no level word:
    # the released = 1 rows of the exceptions list (2.4, Q05).
    no_grade_employees: int
    wave1_keys: tuple[TitleKey, ...]  # the Wave 1 list, sorted (Q24)
    decisions: tuple[Decision, ...]  # exactly one per title row, sorted by key
    audits: tuple[AuditResult, ...]  # one per machine path
    golden_keys: tuple[TitleKey, ...]
    golden_scores: tuple[GoldenScore, ...]  # Wave 1, Wave 2 and the close (Q08)
    queue: tuple[QueueEntry, ...]
    answer_labels: tuple[str, ...]  # the distinct labels of the model answers, sorted (Q40)
