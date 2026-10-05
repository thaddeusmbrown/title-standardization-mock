"""The pipeline: set-up, Wave 1, Wave 2 and the close, in one run.

Owner: feat/pipeline-report-cli (stage 4).
Governed by: docs/design.md 1.6 (the flow), 1.10 (waves), 2.2 (what is in the
mock), 2.6 (what a run shows); docs/build-decisions.md Q04 (the ML-alone
route), Q06 (fresh state on every run), Q07 and Q08 (the golden set), Q13
(look-up first), Q16 (the table is a partition), Q18 (processing order; score
a whole wave, then learn), Q21 (audited titles), Q22 (Wave 1 scoring is one
batch before any approval), Q24 (the Wave 1 list), Q37 (what set-up does).
Section 2.7 tests, end to end: all eight, on the run's files and state.
Other tests this branch owns, listed with their entries in "How the build is
cut" of docs/build-decisions.md: the run shape and its tuning (Q23), the frozen
table (Q23, Q38), the invariants of Q05 and Q16, two runs with the same bytes
(Q06), a run at a high undecided rate (Q01), the default run under the import
guard (Q32 a), the 60-second limit (Q38), spies showing that no model is
called for a title with a person's name or no usable family (Q14, Q26), a seed
hit on the Wave 1 list (Q13) and the ML-alone route switched on (Q04).

Order of a run:
  set-up  read and check the data; fix the settings; prepare the title rows;
          load the seed look-up; pick the Wave 1 list (the top
          settings.wave1_size rows by headcount, then family code, then cleaned
          title); draw the golden set.
  Wave 1  look-up; the machine proposes for the Wave 1 list in one batch; an
          HR SME approves every one; the golden set is labeled and scored; the
          agent only recommends. Then the approvals are learned.
  Wave 2  the long tail is scored for release; the agent's answer stands; both
          machine paths are audited; everything else goes to the review queue.
  close   people clear the queue; the golden set is scored once more;
          leftovers take a generic family role under the cap; the files are
          written and the table printed.

Titles are processed sorted by key. This module and titlestd.mockdata are the
only ones that may import titlestd.people; a model call is handed only its
arguments (Q22).

run_pipeline(settings, models=...) is the seam for tests (S09): a test hands
in the models it wants, for example a misbehaving chooser, spying providers or
a live adapter built on a scripted client. Nothing else differs.

Decided here, where the record is silent:

- What a run leaves behind. A run first removes the four output files and
  the state file an earlier run left in settings.out_dir. The models are
  built before the state file is made, so a provider that refuses (a missing
  extra, a recording that is absent or incomplete) stops the run with
  nothing in the folder. A run that stops later, on any error, leaves none
  of the four output files (Q29: "no output files"); the state file stays
  and holds what was recorded up to the last completed step. The output
  files are written only after every decision is made, the review queue
  first and the load file last. Llm.finish is called last of all, after the
  files and the state are written; if it fails the output files are removed
  like on any other error.
- The data fingerprint is the SHA-256 of the fingerprint of the six input
  files, a line feed and the fingerprint of the truth file (S09).
- A Wave 1 title no model may see (a person's name, no usable family code)
  gets no machine proposal: the SME approves it like every other title of
  the list, and it is counted under "approved in Wave 1". The mock data has
  none, since such titles sit in the long tail.
- A title the SME marks "no suitable role" in Wave 1 or in an audit (Q03:
  only a title with no truth row) is listed in the review-queue file with
  the person's mark and stays on its legacy title. The Wave 1 title goes
  through the close like any leftover so marked; the audited one gets its
  decision from titlestd.audit.
- The golden-set passes of Wave 1 and of the close (Q08) put the golden
  titles through the same steps as a wave up to the rules, and never to the
  agent: the footer scores the ML first choice and the LLM's pick. With
  ml_alone on, a golden title whose ML first choice qualifies skips the LLM
  in these passes too and counts as "no answer". The LLM's pick is the role
  its number named among the roles it was shown, whether or not the quote
  check passed.
- The reasons of a review-queue title are those of the rules on the LLM's
  answer followed by those on the agent's answer, each once: "models
  disagree; agent escalated". A title no model may see carries "person's
  name", "no usable family code", or both in that order.
- In the review-queue file the agent's recommendation is the role its number
  named when the catalog check passed, whether or not another check then
  sent the title to a person. The answer label is the distinct labels of the
  title's LLM and agent answers in that order, joined by "; " (in replay the
  two may carry different dates); it is empty where no model answered.
- Learning (1.6 step 10) happens at three moments, each after the scoring it
  follows (Q18): the Wave 1 approvals after Wave 1 is scored and the golden
  set labeled; the audit labels after Wave 2 is scored and both paths are
  audited; the review-queue decisions before the golden-set pass of the
  close. Golden-set labels are never learned, and a decision that names no
  role is nothing to learn. After each of the three the run checks that no
  golden-set title is in the look-up table or the examples, and stops with
  an error if one is (Q08: "asserted after every stage").
- Which scorings bar a leftover from a generic role. Q25 makes a leftover
  eligible only if the compliance check of 1.7 "never fired on any proposal
  or on the ML top three", and contracts.Leftover says "at any point". So
  the check counts in every scoring of the title during the run. For most
  titles that is the Wave 2 scoring. A golden-set title is scored twice
  more, in the golden-set passes of Wave 1 and of the close (Q08), against
  other examples, and the check counts there as well: the fail-safe reading.
  It is the one way in which being in the golden set can change what a title
  is released with (Q07): such a leftover stays on its legacy title where
  another would take a generic role. The leftovers are therefore closed
  after the golden-set pass of the close. The review-queue file gives the
  reasons of Wave 2 as before; the state file has those of every scoring.
- The number of employees with no grade who received a new title is counted
  with titlestd.catalog.level_for_grade, as the release writer reads a grade.
- The state records the occasions of people's decisions as OCCASION_WAVE1,
  OCCASION_GOLDEN, OCCASION_AUDIT and OCCASION_QUEUE.
- After the table and footer of titlestd.report the run command prints one
  more line, naming the folder and the files it wrote.

The run shape (Q23), measured on the default seed with the data as
feat/mock-data committed it: the bar holds. All seven rows of the table hold
titles, the last two through the leftover mechanism of Q01; the models-agree
path holds more than 60 titles, so its audit is a sampled one, and it is not
stopped (it is confirmed). The agent-settled path holds fewer than 60 titles
and is labeled in full, which Q23 allows. So no tuning round was made: the
difficulty constants of titlestd.mockdata and the files in data/ are as that
branch left them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from titlestd import (
    agent,
    audit,
    close,
    datafiles,
    fallback,
    golden,
    lookup,
    prepare,
    release,
    report,
    rules,
    store,
    switch,
)
from titlestd.catalog import family_codes, get_role, level_for_grade
from titlestd.contracts import (
    AUDIT_ACCURACY_BAR,
    AUDIT_SAMPLE_SIZE,
    LOOKUP_DECIDED_BY,
    AgentResult,
    AuditResult,
    Catalog,
    ChooseAnswer,
    Decision,
    DecisionSource,
    GoldenScore,
    HumanAnswer,
    HumanDecision,
    InputData,
    Leftover,
    MachinePath,
    MatchResult,
    Models,
    NumberedRole,
    QueueEntry,
    Reason,
    Route,
    RuleOutcome,
    RunResult,
    Stage,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
    TitlestdError,
)
from titlestd.matcher import Matcher, ml_alone_eligible
from titlestd.people import SimulatedPeople, load_people, truth_fingerprint
from titlestd.settings import Settings

# The occasions on which a person decides, as the state file names them.
OCCASION_WAVE1 = "Wave 1 approval"
OCCASION_GOLDEN = "golden-set label"
OCCASION_AUDIT = "audit label"
OCCASION_QUEUE = "review-queue decision"

OUTPUT_FILES = (
    release.LOAD_FILE,
    release.REVERSE_FILE,
    release.EXCEPTIONS_FILE,
    release.REVIEW_QUEUE_FILE,
)


def data_fingerprint(data_dir: Path) -> str:
    """The fingerprint of the seven data files, as a recording's manifest holds it.

    datafiles.input_fingerprint combined with people.truth_fingerprint (S09).
    The pipeline hands it to the switch; a test that builds a provider itself
    calls this, so both use the same value.
    """
    both = f"{datafiles.input_fingerprint(data_dir)}\n{truth_fingerprint(data_dir)}"
    return hashlib.sha256(both.encode("utf-8")).hexdigest()


def run_pipeline(settings: Settings, models: Models | None = None) -> RunResult:
    """Run set-up, Wave 1, Wave 2 and the close; write the state and output files.

    models is None in a real run: the pipeline builds them with
    titlestd.switch.build_models for the settings in force. A test passes its
    own, and the switch is then not called. Either way Models.llm.finish is
    called once, after everything else succeeded (Q29).
    """
    out_dir = settings.out_dir
    _remove(out_dir, (*OUTPUT_FILES, store.STATE_FILE))
    data = datafiles.read_inputs(settings.data_dir)
    simulated = load_people(settings, data.catalog)
    if models is None:
        fingerprint = data_fingerprint(settings.data_dir)
        models = switch.build_models(settings, data.catalog.ladder, data.abbreviations, fingerprint)
    connection = store.create_store(out_dir / store.STATE_FILE)
    try:
        result = _Run(settings, data, simulated, models, connection).result()
        release.write_review_queue(out_dir, result.queue)
        release.write_release_files(
            out_dir, result.decisions, result.rows, data.employees, data.catalog
        )
        store.save_run(connection, result)
        models.llm.finish()
    except BaseException:
        _remove(out_dir, OUTPUT_FILES)
        raise
    finally:
        connection.close()
    return result


def run_command(settings: Settings) -> int:
    """The run and record commands: run_pipeline, then print the table rendered
    by titlestd.report. Returns the exit status."""
    print(report.render(run_pipeline(settings), settings))
    files = ", ".join((*OUTPUT_FILES, store.STATE_FILE))
    print(f"Files written to the folder {settings.out_dir}: {files}.")
    return 0


def _remove(folder: Path, names: Iterable[str]) -> None:
    for name in names:
        (Path(folder) / name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _Scored:
    """What the models said about one title at one stage, and where the rules send it."""

    row: TitleRow
    match: MatchResult
    outcome: RuleOutcome  # the rules on the LLM's answer, or on the ML first choice alone
    shown: tuple[NumberedRole, ...] = ()  # the roles the LLM was shown
    answer: ChooseAnswer | None = None  # None on the ML-alone route (Q04)
    review: AgentResult | None = None  # set when the title was referred to the agent
    review_outcome: RuleOutcome | None = None  # the rules on the agent's answer

    @property
    def path(self) -> MachinePath | None:
        """The machine path that may settle the title, or None: a person decides."""
        if self.outcome.route is Route.ACCEPT:
            return MachinePath.MODELS_AGREE
        if self.review_outcome is not None and self.review_outcome.route is Route.ACCEPT:
            return MachinePath.AGENT_SETTLED
        return None

    @property
    def machine_role(self) -> str | None:
        """The role the title's machine path chose."""
        if self.path is MachinePath.AGENT_SETTLED:
            return self.review_outcome.role_id
        return self.outcome.role_id if self.path else None

    @property
    def llm_pick(self) -> str | None:
        return self.outcome.role_id if self.answer is not None else None

    @property
    def agent_pick(self) -> str | None:
        return self.review_outcome.role_id if self.review_outcome is not None else None

    @property
    def reasons(self) -> tuple[Reason, ...]:
        """Every failed check, the LLM's first, each once."""
        after = self.review_outcome.reasons if self.review_outcome is not None else ()
        return tuple(dict.fromkeys((*self.outcome.reasons, *after)))

    @property
    def route(self) -> str:
        """Where the title went, for the state file."""
        if self.path is not None:
            return self.path.value
        return (self.review_outcome or self.outcome).route.value

    @property
    def labels(self) -> tuple[str, ...]:
        """The answer labels of the title's model answers, the LLM's first (Q40)."""
        answers = (self.answer, self.review.answer if self.review is not None else None)
        return tuple(dict.fromkeys(each.source for each in answers if each is not None))


class _Run:
    """One run. The stages are methods called in order by result()."""

    def __init__(
        self,
        settings: Settings,
        data: InputData,
        simulated: SimulatedPeople,
        models: Models,
        connection: sqlite3.Connection,
    ) -> None:
        self.settings = settings
        self.data = data
        self.catalog: Catalog = data.catalog
        self.people = simulated
        self.models = models
        self.connection = connection
        self.decisions: dict[TitleKey, Decision] = {}
        self.labels: set[str] = set()  # the answer labels seen (Q40)
        # Titles listed in the review-queue file: reasons, scoring, the person's decision.
        self.queue: dict[TitleKey, tuple[tuple[Reason, ...], _Scored | None]] = {}
        self.queue_decisions: dict[TitleKey, HumanDecision] = {}
        self.leftovers: list[Leftover] = []
        # Titles on which the compliance check of 1.7 fired, in any scoring (Q25).
        self.flagged: set[TitleKey] = set()

    def result(self) -> RunResult:
        self.set_up()
        scores = [self.wave1()]
        audits, wave2_score = self.wave2()
        scores += [wave2_score, self.close()]
        decisions = tuple(self.decisions[row.key] for row in self.rows)
        return RunResult(
            rows=self.rows,
            employee_count=len(self.data.employees),
            no_grade_employees=self._no_grade_employees(),
            wave1_keys=self.wave1_keys,
            decisions=decisions,
            audits=audits,
            golden_keys=self.golden_keys,
            golden_scores=tuple(scores),
            queue=tuple(self._queue_entry(key) for key in sorted(self.queue)),
            answer_labels=tuple(sorted(self.labels)),
        )

    # --- set-up (1.10, Q37) -------------------------------------------------------

    def set_up(self) -> None:
        data, settings = self.data, self.settings
        self.rows = prepare.prepare(data.employees, data.abbreviations, family_codes(self.catalog))
        self.by_key = {row.key: row for row in self.rows}
        # The Wave 1 list: the most widely held rows, ties by key (Q24).
        widest = sorted(self.rows, key=lambda row: (-row.headcount, row.key))
        self.wave1_keys = tuple(sorted(row.key for row in widest[: settings.wave1_size]))
        seed_keys = lookup.seed_keys(data.seed_lookup, data.abbreviations)
        self.golden_keys = golden.draw_golden(self.rows, self.wave1_keys, seed_keys, settings)
        self.golden_rows = [self.by_key[key] for key in self.golden_keys]
        self.knowledge = lookup.Knowledge(self.golden_keys)
        lookup.load_seed(self.knowledge, data.seed_lookup, self.rows, data.abbreviations)
        self.matcher = Matcher(self.catalog, self.models.embedder)
        self.tools = agent.SearchTools(self.catalog, self.knowledge, self.models.embedder, settings)
        thresholds = {
            "audit_sample_size": AUDIT_SAMPLE_SIZE,
            "audit_accuracy_bar": AUDIT_ACCURACY_BAR,
            "audit_confirm_limit": audit.confirm_limit(AUDIT_SAMPLE_SIZE),
            "audit_stop_limit": audit.stop_limit(AUDIT_SAMPLE_SIZE),
            "generic_cap_employees": close.generic_cap(len(data.employees), settings),
            "release_stopped_paths": settings.release_stopped_paths,
        }
        store.save_setup(
            self.connection, settings, thresholds, self.rows, self.wave1_keys, self.golden_keys
        )
        store.save_knowledge(self.connection, Stage.SETUP, self.knowledge)

    # --- Wave 1 (1.10, Q13, Q22) --------------------------------------------------

    def wave1(self) -> GoldenScore:
        # The look-up runs first for every row; a hit takes its role with no model (Q13).
        for row in self.rows:
            role_id = self.knowledge.lookup(row.key)
            if role_id is not None:
                self.decisions[row.key] = Decision(
                    key=row.key,
                    row=TableRow.LOOKUP,
                    source=DecisionSource.LOOKUP,
                    role_id=role_id,
                    status=Status.APPROVED,
                    decided_by=LOOKUP_DECIDED_BY,
                )
        listed = [self.by_key[key] for key in self.wave1_keys if key not in self.decisions]
        # Every machine answer of Wave 1 comes before any approval (Q22).
        proposals = self._score(Stage.WAVE1, [row for row in listed if row.model_eligible])
        golden_pass = self._score(Stage.WAVE1, self.golden_rows, with_agent=False)
        labels = [self.people.blind_label(row) for row in self.golden_rows]
        self.golden_labels = {label.key: label.role_id for label in labels}
        store.save_people(self.connection, OCCASION_GOLDEN, labels)

        approvals = [self.people.wave1_approval(row) for row in listed]
        store.save_people(self.connection, OCCASION_WAVE1, approvals)
        for row, approval in zip(listed, approvals, strict=True):
            if approval.answer is HumanAnswer.ROLE:
                self.decisions[row.key] = _approved(approval, TableRow.WAVE1)
            else:  # "no suitable role": it waits for the catalog owner (Q03)
                self.queue[row.key] = ((), proposals.get(row.key))
                self.queue_decisions[row.key] = approval
                self.leftovers.append(Leftover(row, on_wave1_list=True, no_suitable_role=True))
        self._learn(Stage.WAVE1, approvals)
        return self._golden_score(Stage.WAVE1, golden_pass)

    # --- Wave 2 (1.10, Q16, Q21) --------------------------------------------------

    def wave2(self) -> tuple[tuple[AuditResult, ...], GoldenScore]:
        wave1 = set(self.wave1_keys)
        long_tail = [
            row for row in self.rows if row.key not in self.decisions and row.key not in wave1
        ]
        scored = self._score(Stage.WAVE2, [row for row in long_tail if row.model_eligible])
        for row in long_tail:
            each = scored.get(row.key)
            if each is None:  # to a person and to no model (Q14, Q26)
                self.queue[row.key] = (_unscored_reasons(row), None)
            elif each.path is None:
                self.queue[row.key] = (each.reasons, each)

        audits, labels = [], []
        for path in MachinePath:
            roles = {key: each.machine_role for key, each in scored.items() if each.path is path}
            sample = audit.draw_sample(path, roles, self.settings.seed)
            blind = {key: self.people.blind_label(self.by_key[key]) for key in sample}
            audited = audit.audit_path(path, roles, blind)
            audits.append(audited)
            labels += blind.values()
            for decision in audit.path_decisions(audited, roles, blind, self.settings):
                self.decisions[decision.key] = decision
            for key, label in blind.items():
                if label.answer is HumanAnswer.NO_SUITABLE_ROLE:  # listed, as Q03 asks
                    self.queue[key] = ((), scored[key])
                    self.queue_decisions[key] = label
        store.save_people(self.connection, OCCASION_AUDIT, labels)
        self._learn(Stage.WAVE2, labels)
        return tuple(audits), self._golden_score(Stage.WAVE2, scored)

    # --- the close (1.8, 1.10, Q01, Q25) ------------------------------------------

    def close(self) -> GoldenScore:
        waiting = [key for key in sorted(self.queue) if key not in self.queue_decisions]
        decided = [self.people.queue_decision(self.by_key[key]) for key in waiting]
        store.save_people(self.connection, OCCASION_QUEUE, decided)
        for decision in decided:
            self.queue_decisions[decision.key] = decision
            if decision.answer is HumanAnswer.ROLE:
                self.decisions[decision.key] = _approved(decision, TableRow.REVIEW_QUEUE)
        # The golden set is scored once more, after the queue decisions joined (Q08).
        self._learn(Stage.CLOSE, decided)
        golden_pass = self._score(Stage.CLOSE, self.golden_rows, with_agent=False)
        # Only now are the leftovers closed: the compliance check bars a title
        # from a generic role wherever in the run it fired, this pass included (Q25).
        self.leftovers += [
            Leftover(
                self.by_key[decision.key],
                compliance_fired=decision.key in self.flagged,
                no_suitable_role=decision.answer is HumanAnswer.NO_SUITABLE_ROLE,
            )
            for decision in decided
            if decision.answer is not HumanAnswer.ROLE
        ]
        for decision in close.close_leftovers(
            self.leftovers, len(self.data.employees), self.catalog, self.settings
        ):
            self.decisions[decision.key] = decision
        return self._golden_score(Stage.CLOSE, golden_pass)

    # --- shared steps -------------------------------------------------------------

    def _score(
        self, stage: Stage, rows: Sequence[TitleRow], *, with_agent: bool = True
    ) -> dict[TitleKey, _Scored]:
        """Put the rows through the ML match, the LLM, the rules and the agent
        (1.6 steps 3 to 6), against the examples as they stand now.

        The whole batch is matched first and nothing is learned here (Q18). The
        rows are sorted by key and every one must be model_eligible.
        """
        catalog, settings, llm = self.catalog, self.settings, self.models.llm
        matches = self.matcher.match(rows, self.knowledge)
        scored: dict[TitleKey, _Scored] = {}
        for row in rows:
            match, family = matches[row.key], row.record.family_code
            if ml_alone_eligible(match, family, catalog, settings):
                # The ML first choice skips the LLM; a person still decides (Q04).
                each = _Scored(row, match, rules.check_ml_alone(match, family, catalog))
            else:
                shortlist = [get_role(catalog, listed.role_id) for listed in match.shortlist]
                shown = fallback.number_roles(shortlist, row.key, settings.seed)
                block = fallback.render_record(row.record)
                answer = llm.chooser.choose(row.record, shown)
                outcome = rules.check_llm(answer, shown, block, match, family, catalog)
                each = _Scored(row, match, outcome, shown, answer)
                if with_agent and outcome.route is Route.AGENT:
                    case = agent.build_case(
                        row, shown, answer, outcome.role_id, match, outcome.reasons
                    )
                    review = agent.review(case, self.tools, llm.reviewer)
                    numbered = review.transcript.numbered_roles
                    review_outcome = rules.check_agent(
                        review.answer, numbered, block, match, family, catalog
                    )
                    each = dataclasses.replace(each, review=review, review_outcome=review_outcome)
            scored[row.key] = each
            self.labels.update(each.labels)
            if Reason.COMPLIANCE_FLAG in each.reasons:
                self.flagged.add(row.key)
            store.save_scoring(
                self.connection,
                stage,
                row.key,
                match=match,
                route=each.route,
                reasons=each.reasons,
                shown=each.shown,
                answer=each.answer,
                outcome=each.outcome,
                review=each.review,
                review_outcome=each.review_outcome,
            )
        return scored

    def _learn(self, stage: Stage, decisions: Iterable[HumanDecision]) -> None:
        """A wave's decisions join the look-up table and the examples, after the
        wave is scored (1.6 step 10, Q18). The guard in titlestd.lookup refuses
        a golden-set title, whatever the decision was (Q07)."""
        for decision in decisions:
            if decision.answer is HumanAnswer.ROLE:
                self.knowledge.learn(self.by_key[decision.key], decision.role_id)
        # Asserted after every stage (Q08): no golden title was learned.
        learned = {key for key, _ in self.knowledge.entries()}
        learned.update(example.key for example in self.knowledge.examples())
        golden_learned = sorted(learned.intersection(self.golden_keys))
        if golden_learned:
            raise TitlestdError(
                f"golden-set title {golden_learned[0].cleaned_title!r} in family "
                f"{golden_learned[0].family_code!r} entered the look-up table or the examples "
                f"at {stage.value}; the run is stopped (design 1.2)"
            )
        store.save_knowledge(self.connection, stage, self.knowledge)

    def _golden_score(self, stage: Stage, scored: Mapping[TitleKey, _Scored]) -> GoldenScore:
        """Score the golden set on the answers of one stage (Q08)."""
        return golden.score_golden(
            stage,
            self.golden_labels,
            {key: each.match.first_choice for key, each in scored.items()},
            {key: each.llm_pick for key, each in scored.items()},
        )

    def _queue_entry(self, key: TitleKey) -> QueueEntry:
        reasons, scored = self.queue[key]
        return QueueEntry(
            row=self.by_key[key],
            reasons=reasons,
            ml_first_choice=scored.match.first_choice if scored else None,
            llm_pick=scored.llm_pick if scored else None,
            agent_recommendation=scored.agent_pick if scored else None,
            answer_source="; ".join(scored.labels) if scored else "",
            decision=self.queue_decisions[key],
            outcome=self.decisions[key].row,
        )

    def _no_grade_employees(self) -> int:
        """Employees with no usable grade who received a new title (2.4, Q05)."""
        released = set()
        for row in self.rows:
            if self.decisions[row.key].row is not TableRow.LEGACY:
                released.update(row.employee_ids)
        return sum(
            employee.employee_id in released
            and level_for_grade(self.catalog, employee.grade) is None
            for employee in self.data.employees
        )


def _approved(decision: HumanDecision, table_row: TableRow) -> Decision:
    """The final decision on a title a person gave a role (status Approved, 1.9)."""
    return Decision(
        key=decision.key,
        row=table_row,
        source=DecisionSource.PERSON,
        role_id=decision.role_id,
        status=Status.APPROVED,
        decided_by=decision.decided_by,
        simulated=decision.simulated,
    )


def _unscored_reasons(row: TitleRow) -> tuple[Reason, ...]:
    """Why a title goes to a person and to no model (Q14, Q26)."""
    reasons = []
    if row.has_person_name:
        reasons.append(Reason.PERSON_NAME)
    if not row.family_usable:
        reasons.append(Reason.NO_USABLE_FAMILY)
    return tuple(reasons)
