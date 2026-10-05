"""The rules: code checks every proposal.

Owner: feat/llm-fallback-rules (stage 2).
Governed by: docs/design.md 1.7; docs/build-decisions.md Q04 (the ML-alone
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
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import (
    Catalog,
    ChooseAnswer,
    MatchResult,
    NumberedRole,
    ReviewAnswer,
    RuleOutcome,
)

_NOT_BUILT = "titlestd.rules is built by feat/llm-fallback-rules"


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
    raise NotImplementedError(_NOT_BUILT)


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
    raise NotImplementedError(_NOT_BUILT)


def check_ml_alone(match: MatchResult, family_code: str, catalog: Catalog) -> RuleOutcome:
    """The checks that still run when the ML first choice skips the LLM (Q04):
    catalog and compliance. The result is always Route.PERSON, with
    Reason.ML_ALONE first."""
    raise NotImplementedError(_NOT_BUILT)
