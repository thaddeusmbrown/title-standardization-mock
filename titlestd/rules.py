"""The rules: code checks every proposal.

Owner: feat/llm-fallback-rules (stage 2).
Governed by: DESIGN.md 1.7; docs/build-decisions.md Q04 (the ML-alone
route keeps the catalog and compliance checks), Q15 (the quote is checked on
every choose answer; an out-of-range number fails the catalog check), Q20 and
Q39 (an invalid answer goes to a person).
Section 2.7 tests: a quote that is not in the record sends the title to a
person; a compliance-flagged role is never settled by a machine.

The table of 1.7, in order:
  the role exists in the catalog                         a person
  the quote appears word for word in the masked record   a person
  the proposed role, or any of the ML model's top three,
    is compliance-flagged                                a person
  the LLM answered "insufficient data"                   a person
  the LLM and the ML model's first choice agree          the agent
  the proposed role is in the recorded family            the agent
  the LLM answered "no match"                            the agent

If several fail, a person beats the agent. Everything the agent returns is
checked by the same rules except agreement, and any failure sends the title to
a person, as does an escalation.

The result lists every failed check as a contracts.Reason, in table order.

Decided here, where the record leaves the detail open:

- The proposed role is the one the answer's number names among the roles
  shown (titlestd.fallback.role_for_number), read again from the catalog, so
  the family and the compliance flag are the catalog's own. Only an answer
  of the kind "role" proposes one: a number on any other answer is not read.
- The catalog check fails when the number names no shown role, when that
  role is not in the catalog, and when it is a generic family role, which
  only code may assign (1.2). The outcome then carries no role.
- The agreement and family checks are about the proposed role. With none
  (a failed catalog check, "no match", "insufficient data") they neither
  pass nor fail and are not listed.
- The compliance check reads the ML model's top three whatever the answer
  was, an invalid answer and an escalation included, so the outcome always
  says whether the title touches a flagged role (Q25 needs that at the
  close). A role id of the top three that the catalog does not hold cannot
  carry a flag and is passed over.
- An invalid answer (S10) is not checked for its quote: there is no answer
  to check. An answer of a kind this module does not know is invalid too.
- Reasons that are not in the table follow those that are: "agent
  escalated" and "invalid model answer" come last. check_ml_alone is the
  exception the scaffold fixed: "ML alone" comes first.
- The recorded family is read as titlestd.catalog reads a family code:
  trimmed, then compared exactly (S05). A blank or unknown code holds no
  role, so every proposal is outside it.
- check_ml_alone also lists "role outside recorded family" when the ML first
  choice is not in the family it is given. Q04 makes that a condition of the
  route (titlestd.matcher.ml_alone_eligible), so it only shows for a caller
  that skipped the condition; the title goes to a person either way.
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.catalog import get_role, roles_in_family
from titlestd.contracts import (
    Catalog,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    MatchResult,
    NumberedRole,
    Reason,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
)
from titlestd.fallback import quote_in_record, role_for_number

# 1.7: the checks whose failure the agent may settle. Every other reason needs a person.
_TO_THE_AGENT = frozenset({Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH, Reason.NO_MATCH})


def check_llm(
    answer: ChooseAnswer,
    roles: Sequence[NumberedRole],
    record_block: str,
    match: MatchResult,
    family_code: str,
    catalog: Catalog,
) -> RuleOutcome:
    """Check the LLM fallback's answer for one title.

    roles are the numbered roles the LLM was shown. Route.ACCEPT means the two
    models agree and every check passed.
    """
    kind = answer.answer
    if kind not in (ChooseKind.ROLE, ChooseKind.NO_MATCH, ChooseKind.INSUFFICIENT_DATA):
        flagged = _compliance(None, match, catalog)
        return RuleOutcome(Route.PERSON, (*flagged, Reason.INVALID_ANSWER))
    role = _catalog_role(answer.number, roles, catalog) if kind == ChooseKind.ROLE else None
    reasons: list[Reason] = []
    if kind == ChooseKind.ROLE and role is None:
        reasons.append(Reason.ROLE_NOT_IN_CATALOG)
    if not quote_in_record(answer.quote, record_block):
        reasons.append(Reason.QUOTE_NOT_IN_RECORD)
    reasons += _compliance(role, match, catalog)
    if kind == ChooseKind.INSUFFICIENT_DATA:
        reasons.append(Reason.INSUFFICIENT_DATA)
    if role is not None and role.role_id != match.first_choice:
        reasons.append(Reason.MODELS_DISAGREE)
    if role is not None and role not in roles_in_family(catalog, family_code):
        reasons.append(Reason.FAMILY_MISMATCH)
    if kind == ChooseKind.NO_MATCH:
        reasons.append(Reason.NO_MATCH)
    if not reasons:
        route = Route.ACCEPT
    elif _TO_THE_AGENT.issuperset(reasons):
        route = Route.AGENT
    else:
        route = Route.PERSON
    return RuleOutcome(route, tuple(reasons), role.role_id if role else None)


def check_agent(
    answer: ReviewAnswer,
    roles: Sequence[NumberedRole],
    record_block: str,
    match: MatchResult,
    family_code: str,
    catalog: Catalog,
) -> RuleOutcome:
    """Check the agent's final answer for one title.

    roles are every numbered role the agent was shown. The result is
    Route.ACCEPT or Route.PERSON, never Route.AGENT.
    """
    kind = answer.answer
    role = None
    reasons: list[Reason] = []
    if kind == ReviewKind.ROLE:
        role = _catalog_role(answer.number, roles, catalog)
        if role is None:
            reasons.append(Reason.ROLE_NOT_IN_CATALOG)
        if not quote_in_record(answer.quote, record_block):
            reasons.append(Reason.QUOTE_NOT_IN_RECORD)
    reasons += _compliance(role, match, catalog)
    if role is not None and role not in roles_in_family(catalog, family_code):
        reasons.append(Reason.FAMILY_MISMATCH)
    if kind == ReviewKind.ESCALATE:
        reasons.append(Reason.AGENT_ESCALATED)
    elif kind != ReviewKind.ROLE:
        reasons.append(Reason.INVALID_ANSWER)
    route = Route.PERSON if reasons else Route.ACCEPT
    return RuleOutcome(route, tuple(reasons), role.role_id if role else None)


def check_ml_alone(match: MatchResult, family_code: str, catalog: Catalog) -> RuleOutcome:
    """The checks that still run when the ML first choice skips the LLM (Q04):
    catalog and compliance. The result is always Route.PERSON, with
    Reason.ML_ALONE first."""
    role = _model_role(match.first_choice, catalog)
    reasons = [Reason.ML_ALONE]
    if role is None:
        reasons.append(Reason.ROLE_NOT_IN_CATALOG)
    reasons += _compliance(role, match, catalog)
    if role is not None and role not in roles_in_family(catalog, family_code):
        reasons.append(Reason.FAMILY_MISMATCH)
    return RuleOutcome(Route.PERSON, tuple(reasons), role.role_id if role else None)


def _model_role(role_id: str | None, catalog: Catalog) -> CatalogRole | None:
    """The catalog's role of this id when a model may propose it: never a
    generic family role (1.2)."""
    role = get_role(catalog, role_id) if role_id is not None else None
    return None if role is None or role.is_generic else role


def _catalog_role(
    number: int | None, roles: Sequence[NumberedRole], catalog: Catalog
) -> CatalogRole | None:
    """The role an answer's number proposes, or None when the catalog check fails."""
    shown = role_for_number(number, roles)
    return _model_role(shown.role_id, catalog) if shown else None


def _compliance(role: CatalogRole | None, match: MatchResult, catalog: Catalog) -> list[Reason]:
    """The compliance check: the proposed role, or any of the ML model's top three."""
    touched = [role, *(get_role(catalog, role_id) for role_id in match.top_three)]
    flagged = any(each is not None and each.compliance_flag for each in touched)
    return [Reason.COMPLIANCE_FLAG] if flagged else []
