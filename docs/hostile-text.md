# Hostile text in the record: what holds and what does not

PROMPT.md "Known limits" said the record's text had never been tested with hostile text: words
written into a title, a family code, a department or a description to steer a model, or to slip
a name or an address past the masks. Five test files now try it, offline, on records written
for them, on the mini data set and on copies of it with hostile employees added. The findings
below are as of 2026-10-20, and the first command under "Rerun the tests" checks them again.

No hosted model read any of this. Where a test needs a model that obeys the hostile text, its
answer is scripted: a chooser that answers with the number or the role a payload names, and a
reviewer that does what its description says. The offline stand-ins run too. The stand-in LLM
scores the words of the title and of the description against each role's name and definition
(Q17), so a description that names a role steers it. So the tests show what the code makes of a
steered answer, not how often a model would give one.

Every check that code makes held as written. Four things do not hold. The text a model is shown
is not fenced off from the record's own lines. A steered pick of an ordinary role of the right
family passes every rule, with only the audit behind it. The masks know only their own shapes,
and an address in a title is not masked at all. And a name or an address in a family code is
never masked and reaches an agent once the row is learned before that agent searches: a defect
against DESIGN.md 1.4, left to the owner.

| Hostile text | Where it ends |
|---|---|
| A name or an address in a description or a department, in a shape the masks know | Masked before any model sees it |
| A title with a listed name, or a family code that is not a catalog code | A person, and no model |
| A number that is on no list | The catalog check fails: a person |
| A pick of the compliance-flagged role | A person |
| A pick of a role of another family | The agent |
| A quote from the real roles list | A person |
| A quote copied from forged text in the record | Passes: the role the number names decides |
| A pick of an unflagged role of the family that the ML match is drawn to as well | Models agree, with no reason: only the audit |
| An agent that obeys a forged role list | Agent-settled: only the audit |
| A name split by an invisible character, or in fullwidth letters | Not masked: it reaches a model |
| An address in a title | Not masked: broken into tokens, it reaches a model |
| A name or an address in a family code | Not masked: shown to every agent that searches after the row is learned (a Wave 1 approval, or a seed look-up row) |

## What holds

- **Masking, in its own shapes.** In a description and a department, an email address and a
  listed name are masked wherever a payload puts them: inside an instruction, after `mailto:`,
  in angle brackets, across a line break, in capitals, on a forged label line, at a host under
  the reserved domain. In a whole run no model is handed a name or an address of the hostile
  records, and a title with a listed name or a family code that is not a catalog code goes to
  a person and to no model (DESIGN.md 1.4, Q14, Q26). `test_prepare_hostile.py`:
  `test_an_address_or_a_name_in_a_hostile_text_is_masked`,
  `test_a_hostile_title_with_a_name_goes_to_no_model`,
  `test_a_family_code_carrying_a_payload_is_not_usable`; `test_pipeline_hostile.py`:
  `test_no_model_is_handed_a_name_an_email_or_a_row_no_model_may_see`.
- **The cut and the title.** A description ends at its 200th word, counted after masking, so a
  payload after it never reaches the record. A title cleans to lower-case tokens joined by
  single spaces: its words stay, but no line break, colon or label line does.
  `test_prepare_hostile.py`: `test_a_payload_after_the_200th_word_is_not_in_the_record`,
  `test_a_hostile_title_cleans_to_tokens_joined_by_single_spaces`.
- **A number names only a role that was shown.** The numbering never reads the description
  (Q18). A forged number out of range names no role and fails the catalog check, so the title
  goes to a person (Q15, S11). `test_fallback_hostile.py`:
  `test_a_hostile_record_is_numbered_as_a_plain_record_of_its_title`,
  `test_a_forged_number_names_no_role`; `test_rules_hostile.py`:
  `test_a_forged_number_out_of_range_fails_the_catalog_check`.
- **The rules of 1.7.** A steered pick of the compliance-flagged role goes to a person, even
  when the ML match agrees and even when the ML top three hold no flagged role (T3). A steered
  pick of another family's role goes to the agent, whose answer in the family the rules accept.
  A quote from the real roles list, which the record does not hold, sends the title to a person
  (T2). `test_rules_hostile.py`: `test_a_steered_pick_of_the_flagged_role_goes_to_a_person`,
  `test_a_steered_pick_outside_the_family_goes_to_the_agent`,
  `test_a_hostile_row_quoting_the_real_list_goes_to_a_person`; `test_pipeline_hostile.py`:
  `test_t3_a_steered_pick_of_the_flagged_role_sends_the_title_to_a_person`,
  `test_a_steer_out_of_the_family_is_held_by_the_family_rule_and_goes_to_the_agent`.
- **The agent's leash.** A reviewer that obeys the record cannot settle a role shown with "-",
  a number it was never shown or the generic role: each fails the catalog check (T4, T5). A
  steer to the flagged role goes to a person. A fifth search, two searches in one turn and a
  tool that does not exist are an invalid answer, refused before anything runs. After four
  searches the request still lists the two tools but sets `tool_choice` to none, so the turn
  can only be the final answer (Q20). No search returns a generic family role, whatever the
  query, and none costs a result. `test_agent_hostile.py`:
  `test_a_record_that_steers_the_pick_to_a_role_shown_with_a_dash_settles_nothing`,
  `test_a_record_that_steers_the_pick_to_a_number_never_shown_settles_nothing`,
  `test_a_record_that_steers_the_pick_to_the_generic_role_settles_nothing`,
  `test_a_record_that_says_search_again_gets_four_searches_and_a_person`,
  `test_two_searches_in_one_turn_are_a_broken_protocol_and_neither_is_run`,
  `test_two_tool_requests_in_one_response_are_an_invalid_turn`,
  `test_a_record_that_says_call_update_catalog_runs_nothing_and_goes_to_a_person`,
  `test_no_hostile_query_brings_a_generic_role_into_a_search_result`.
- **A whole run.** T1, T3, T5, T7 and T8 hold on the run's files and state. T1 holds by
  construction there, since no hostile text reaches a title builder. `test_pipeline_hostile.py`:
  the tests `test_t1_` to `test_t8_`.

## What does not hold

**The record's text is not fenced off.** A description or a department keeps its own line
breaks and label-shaped lines, and the record block shows it after its label with nothing to
mark where it ends. In the choose request the block runs straight on to the real list, so a
forged "Roles:" list stands above the real one in the very form of its lines. In the agent's
case, forged marks and a forged "Referred because:" line are lines of the prompt too, told from
the real ones only by where they stand. A department reaches the LLM and the agent the same
way, though the ML match does not embed it. And since the quote check accepts any passage of
the record block, a quote copied from forged text passes, so the role the number names, or the
kind of the answer, decides the route. A description that copies a line of the real list makes
that role's definition a passing quote. `test_prepare_hostile.py`:
`test_a_descriptions_line_breaks_and_label_lines_reach_the_record_as_recorded`;
`test_fallback_hostile.py`:
`test_the_choose_request_holds_the_hostile_block_with_nothing_around_it`,
`test_a_quote_copied_from_forged_text_passes_the_check`; `test_rules_hostile.py`:
`test_the_same_forged_quote_is_routed_by_the_role_its_number_names`; `test_agent_hostile.py`:
`test_forged_marks_and_a_forged_referral_stay_inside_the_description`;
`test_pipeline_hostile.py`:
`test_an_order_and_forged_lines_in_a_department_reach_the_llm_and_the_agent`.

**Within the family, only the audit stands behind a steered pick.** PROMPT.md says that what
still stands behind a steered pick is the rules and the audit, and both parts hold: the rules
keep a steered pick to a catalog role of the recorded family that is not flagged, or send the
title to a person. Which of those roles the pick names, no rule checks, and the two models
agreeing is no check of it either. The description feeds the ML match as well as the prompt
(DESIGN.md 1.4), so text that uses a role's words draws the match there too: the same title
with the hostile lines taken out is matched first to another role. A steered pick of an
unflagged role of the recorded family that the match is drawn to passes every check of 1.7
with no reason, and in a run the title goes on the models-agree path. An agent that obeys a
forged numbered list answers with a number of its own case, which shared code maps to the
case's own role: a role of the family that the truth does not hold. Every rule passes and the
title goes on the agent-settled path. Only the audit stands behind either path (1.9). On the
mini data set a path is labeled in full (2.4), so the audit catches both and the person's role
is released. At scale the audit is a sample: on a models-agree path larger than the sample, a
steered title the seeded sample does not draw is released by the machine with the steered role
and the path's status, and no person reads it. `test_rules_hostile.py`:
`test_the_hostile_text_draws_the_ml_match_as_well`,
`test_a_steered_pick_in_the_family_that_the_ml_match_agrees_with_is_accepted`;
`test_agent_hostile.py`:
`test_a_record_that_steers_the_pick_to_a_numbered_role_of_the_family_is_accepted`;
`test_pipeline_hostile.py`:
`test_hostile_text_that_steers_both_models_to_one_wrong_role_is_caught_only_by_the_audit`,
`test_an_agent_that_obeys_a_forged_list_passes_every_rule_and_only_the_audit_catches_it`,
`test_a_steered_title_the_sampled_audit_does_not_draw_is_released_with_the_steered_role`.

**The masks know only the shapes prepare.py defines:** a name is a whole token of the invented
list, which Q14 lets stand in for a real masker, and an email is an unbroken run around its "@".
A name glued to a letter, split by a soft hyphen, a zero-width space or a combining accent, or
written in fullwidth letters passes unmasked in a description or a department. In a title the
split and fullwidth ones are no name as prepare.py defines one, so the row goes to the models.
An address with white space around its "@", or a fullwidth at sign (U+FF20) in its place, is
not masked as an address, and after a zero-width space the rest of a domain stays. An email
address in a title is not masked either: cleaning breaks it into tokens, and the row goes to
the models unless one of them is a listed name. The department has no 200-word cut, since 1.4
and Q11 cut only the description, so a payload after its 200th word stays.
`test_prepare_hostile.py`:
`test_a_name_or_an_address_outside_the_shapes_of_the_masks_stays`,
`test_a_name_the_mask_cannot_read_in_a_title_reaches_a_model`,
`test_an_address_in_a_title_is_broken_into_tokens_and_reaches_a_model`,
`test_a_department_is_not_cut_at_its_200th_word`.

## Left to the owner

On 2026-10-20 the owner agreed a plan for the two points below and for three limits of "What
does not hold": the forged layout, the masks' blindness to invisible and fullwidth characters,
and the uncut department. It accepts the steered pick within the family and leaves the fence
for a second recording. It is in `docs/hostile-fixes-plan.md`, with the two withheld tests, and
is not yet built.

**A name or an address in a family code reaches agents.** This is a defect against DESIGN.md
1.4 ("Names are masked before any text reaches a model"). prepare masks descriptions and
departments, and titles for names, but keeps a family code as received, trimmed (Q26). Such a
row has no usable family code, so no model is handed the row itself. Once it is learned,
though, it joins the examples under its key (Q26), and `agent.render_step` shows an
approved-title hit with the title's own family code (Q10). Every agent search after that which
returns the example shows the code verbatim, name and address included. Two paths learn such a
row before an agent searches:

- **A Wave 1 approval.** A row widely held enough to be on the Wave 1 list is approved and
  learned after Wave 1 is scored, before the agents of Wave 2 search. In
  `test_pipeline_hostile.py`, `test_a_family_code_learned_before_wave2_reaches_its_agents_as_written`
  shows a payload with no name reaching them that way.
- **A seed look-up row.** `datafiles.check_decided_roles` holds a seed row's role to its family
  only where the family code is one of the catalog's, so a seed row with an unusable family
  code is allowed, and `lookup.load_seed` learns it at set-up as a look-up entry and an example
  (Q13). Then the agents of Wave 1 are shown it too.

A row decided only in the review queue is learned at the close, before the golden-set pass,
which never runs the agent (`titlestd/pipeline.py`), so no agent is shown it.

No committed test takes a name and an address in a family code as far as an agent.
`test_a_family_code_carrying_a_payload_is_not_usable` in `test_prepare_hostile.py` has such a
code, but stops at prepare, where it shows the code kept as received. A probe on 2026-10-20
took the seed path on the default data: it added an employee and a seed row with a name and an
address in the family code, learned the row and ran an approved-title search, and render_step's
line showed both. From the repository root this runs it again:

```sh
uv run python - <<'EOF'
from titlestd import agent, catalog, contracts as c, datafiles, lookup, offline_embedder, prepare, settings

s = settings.Settings()
data = datafiles.read_inputs(s.data_dir)
code = "FIN Oswin Thistlewood tavi@example.com"
extra = c.Employee("E9999", "Expense Analyst", code, "Accounting", 5, "Checks expense claims.")
rows = prepare.prepare((*data.employees, extra), data.abbreviations, catalog.family_codes(data.catalog))
seed = (*data.seed_lookup, c.SeedLookupRow("Expense Analyst", code, "FIN-05"))
datafiles.check_decided_roles("seed", data.catalog, [(r.job_title, r.job_family_code, r.role_id) for r in seed])
knowledge = lookup.Knowledge()
lookup.load_seed(knowledge, seed, rows, data.abbreviations)
tools = agent.SearchTools(data.catalog, knowledge, offline_embedder.build_embedder(s), s)
case = c.AgentCase(rows[0].record, (), c.ChooseAnswer(c.ChooseKind.NO_MATCH, None, "x", "offline"), None, None, ())
step = tools.search(c.SearchRequest(c.SearchTool.APPROVED_TITLES, "expense analyst"), c.AgentTranscript(case))
print([line for line in agent.render_step(step).splitlines() if "Oswin" in line])
EOF
```

On 2026-10-20 it printed:

```text
['- Accounts Payable Clerk: approved for the title "expense analyst" (family FIN Oswin Thistlewood tavi@example.com)']
```

The default data has no such row. Its unusable family codes are blank, ADM and TBD, none of
them on the Wave 1 list (in the order the set-up of `titlestd/pipeline.py` picks it) and none
in the seed look-up, and no family code of a title row holds anything `prepare.mask_text`
changes (measured 2026-10-20). From the repository root this prints the four facts again, in
that order, as `['', 'ADM', 'TBD']` and then `[]` three times:

```sh
uv run python - <<'EOF'
from titlestd import catalog, datafiles, prepare, settings

s = settings.Settings()
data = datafiles.read_inputs(s.data_dir)
codes = catalog.family_codes(data.catalog)
rows = prepare.prepare(data.employees, data.abbreviations, codes)
wave1 = sorted(rows, key=lambda row: (-row.headcount, row.key))[: s.wave1_size]
print(sorted({row.key.family_code for row in rows if not row.family_usable}))
print([row.key for row in wave1 if not row.family_usable])
print([seed for seed in data.seed_lookup if seed.job_family_code.strip() not in codes])
print(sorted({row.key.family_code for row in rows if prepare.mask_text(row.key.family_code) != row.key.family_code}))
EOF
```

The fix is to show the "none recorded" family (`agent.NO_FAMILY`) for an unusable family code
in render_step, or to mask family codes in prepare. The first re-decides Q10, under which an
approved-title result shows the title's family code, and the "Decided here" of
`titlestd/agent.py`, which shows NO_FAMILY only for a blank code. The second re-decides Q26,
which keys the row on the family value as received, and Q14, whose masks cover titles,
descriptions and departments and no family code. Those entries are the owner's to re-decide.
Either fix also changes what a request can hold for such a row, in code CLAUDE.md freezes
because it shapes requests, and each turns a pinned test of these files red by design.

Whether the committed recording still replays is for `uv run pytest -m replay` to say. On
2026-10-20 each sketch below was made in a checkout, judged by that command, and taken out
again:

- **NO_FAMILY.** In `agent._approved_title_line`, `family = hit.title_family_code if
  hit.title_family_code == role.family_code else NO_FAMILY`; a usable code always names its
  role's family, since `check_decided_roles` holds a person's decision to it. `uv run pytest -m
  replay` stayed green, and
  `test_a_family_code_learned_before_wave2_reaches_its_agents_as_written` turned red.
- **Masking.** In `prepare.prepare` and `prepare.title_key`,
  `mask_text(job_family_code.strip())` in place of `job_family_code.strip()`. `uv run pytest -m
  replay` stayed green, and the `SAL Oswin Thistlewood tavi@example.com` case of
  `test_a_family_code_carrying_a_payload_is_not_usable` turned red.
- **The positive control.** `LLM_PICK_MARK = "[the LLM's choice]"` in `titlestd/agent.py`
  turned `uv run pytest -m replay` red.

That fits the default data, inferred from the four facts above: no row with an unusable family
code is learned before an agent search, so no request of the recorded run shows one, and
masking changes no family code of the data. So a new paid recording may not be needed, but
that is measured on the sketches, not on the fix, and the replay tests decide for the fix
itself.

**Folding characters before masking.** Whether prepare should fold format and fullwidth
characters before it matches names and addresses: Unicode NFKC, and dropping invisible format
characters (category Cf). That re-decides Q14, the cleaning and masking, and so is the owner's
to re-decide; CLAUDE.md lists prepare's cleaning and masking among what shapes every request.
A sketch on 2026-10-20 folded each text and kept the folded text in the record: in
`titlestd/prepare.py`, with `import unicodedata`, `"".join(ch for ch in
unicodedata.normalize("NFKC", t) if unicodedata.category(ch) != "Cf")` applied to `text` first
thing in `mask_text` and to `job_title` first thing in `_clean`. It masked the soft-hyphen,
zero-width-space and fullwidth cases above but not the combining accents, which NFKC composes
into other letters ("w" and a combining acute become U+1E83), so those names still pass.
`uv run pytest -m replay` stayed green, since the default data is plain ASCII (the `python3 -c`
command below prints `True` when it is). In
`test_a_name_or_an_address_outside_the_shapes_of_the_masks_stays` and
`test_a_name_the_mask_cannot_read_in_a_title_reaches_a_model` the cases of the soft hyphen, the
zero-width space, the combining accents and the fullwidth forms turned red. The two
combining-accent cases turned red although their names are still not masked: the fold changed
the text the record stores.

## Rerun the tests

From the repository root, the five files:

```sh
uv run pytest tests/unit/test_prepare_hostile.py tests/unit/test_fallback_hostile.py \
  tests/integration/test_rules_hostile.py tests/integration/test_agent_hostile.py \
  tests/integration/test_pipeline_hostile.py
```

pytest prints how many ran. The gate runs them with the rest of the suite, in both
environments:

```sh
make check
```

A test that pins a limit asserts the weakness as it stands, so a change that ends the limit
turns it red and changes it. Whether such a change leaves the committed recording replaying,
and whether the default data is plain ASCII:

```sh
uv run pytest -m replay
python3 -c "import pathlib; print(all(p.read_bytes().isascii() for p in pathlib.Path('data').glob('*.csv')))"
```
