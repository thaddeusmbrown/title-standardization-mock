# How the mock prompts a model

Two steps of the mock prompt a model: the LLM fallback (`choose`), which maps a job title onto
one of ten shortlisted catalog roles, and the agent (`review`), which settles the cases the
first step could not. The default run answers both with offline stand-ins that never read a
prompt. A hosted model, `claude-haiku-4-5`, read the prompts below in one recorded live run,
which `python3 -m titlestd run --llm replay` repeats with no key and nothing installed.

## The strategy in brief

- **A model picks, it never writes.** It answers with the number of a role it was shown and a
  quote from the record. Code maps the number to a catalog role and checks the quote, so no
  model output becomes a title by itself (sections 1.1 and 2.5 of the design).
- **Code enforces the limits; the prompt only states them.** The quote check, the family rule,
  the search cap, the kinds of answer and the mapping from number to role are all applied
  outside the model. An answer that fails one sends its title to a person.
- **Little goes in.** Names and email addresses are masked. A title that holds a person's name,
  or has no usable family code, goes to a person and to no model. A description is cut to 200
  words.
- **The shortlist comes from the ML match, in an order that says nothing about it.** The ten
  roles are the ML match's nearest ten across all families, shuffled with a draw seeded from
  the run's seed and the title. A generic family role is never shown.
- **Two methods must agree.** A title is released without a person only when every check
  passes and the LLM picks the ML match's first choice. A disagreement goes to the agent.
- **The agent gets a narrower mandate and a short leash.** It may pick only a role of the
  record's own family, has two read-only searches and at most four of them, and then picks or
  passes the title to a person. Its answer is checked like the LLM's.
- **Structured answers, no sampling settings.** Each request carries a JSON schema for the
  answer and a limit of 1,024 output tokens, and nothing else: no temperature, no thinking
  setting, never a forced tool choice. Repeatability comes from the recording, not from
  temperature 0.
- **Plain prompts, held by tests.** Both system prompts are plain text in
  `titlestd/recording.py`, with no worked examples. Tests assert that they state the values
  code enforces, and pin the recording keys of sample requests, so a reworded prompt fails a
  test.

## 1. The LLM fallback (`choose`)

A request is a system prompt and one user message, with no tools and no earlier turns. The
system prompt, `CHOOSE_SYSTEM`:

```text
You map one job title of a large employer onto a standard role from a catalog that HR approved.

The user message holds:
- the record of the job title: the cleaned title, the code of the job family the employer
  recorded for it, the department and the job description. Names and email addresses are masked
  as [NAME] and [EMAIL]. A field may be empty.
- a numbered list of roles, each with its name, the code of its job family and a one-line
  definition.

Decide which listed role is the same job as the record describes. Any listed role may be chosen,
whichever job family it belongs to. The level of a job (junior, senior, lead, a numeral such as
II) is set elsewhere from the pay grade: ignore it.

Answer with one JSON object:
- "answer": "role" when one listed role fits the record; "no_match" when the record makes clear
  what the job is and no listed role fits; "insufficient_data" when the record says too little
  to tell what the job is.
- "number": the number of the role you chose, exactly as listed; null unless "answer" is "role".
- "quote": the words of the record that your answer rests on, copied character for character
  from the title or the description: whole words, at least 3 characters and at most 25 words.
  A quote is required with every answer; when you choose no role, quote the title.

You can only choose a role by its number. Never write a role name or a title of your own.
```

The user message is the record, then the ten roles. One from the live run, with six of the
ten roles left out here:

```text
Title: senior account manager
Family: SAL
Department: Enterprise Sales
Description: Full-time position based at the main office.

Roles:
1. Payroll Specialist (family FIN): Runs payroll and resolves pay queries.
   (numbers 2 to 7)
8. Account Manager (family SAL): Manages a portfolio of customer accounts and renews customer contracts.
9. Operations Manager (family OPS): Runs daily site operations and improves processes.
10. Marketing Manager (family MKT): Plans campaigns and manages the marketing budget.
```

The model answered number 8, the true role, with the quote "account manager manages a
portfolio of customer accounts and renews customer contracts": the role's name joined to its
definition, not words of the record. So the quote check failed, which is the live run's main
finding in one example.

One sentence is there for the rules. "Any listed role may be chosen, whichever job family it
belongs to" leaves the family check to code and the agent, so the LLM does not rule a role out
itself. The agent is told the opposite.

**What code does with the answer.** A refusal, a cut-off answer, text that is not the JSON
object, or `role` with no number is an invalid answer, which goes to a person. The number is
mapped to the role shown under it. The quote must appear in the record on word boundaries,
after folding case and white space, with at least 3 characters, a letter among them, and at
most 25 words. Then the rules of section 1.7 of the design apply: "models disagree", "role
outside recorded family" and "no match" go to the agent when nothing else failed, and
anything else goes to a person.

## 2. The agent (`review`)

A title reaches the agent only when every check its LLM answer failed is one of those three.
The agent is a plain loop in `titlestd/agent.py`, with no framework: one request per turn, and
each turn is one search or the final answer. The system prompt, `REVIEW_SYSTEM`:

```text
You review one job title that an earlier automatic step could not settle. Either map it onto a
standard role from a catalog that HR approved, or pass it to a person.

The first user message holds:
- the record of the job title: the cleaned title, the code of the job family the employer
  recorded for it, the department and the job description. Names and email addresses are masked
  as [NAME] and [EMAIL]. A field may be empty.
- the roles shortlisted for it, each with its name, the code of its job family and a one-line
  definition. "[the LLM's pick]" marks the role an earlier model chose, and "[the ML model's
  first choice]" the role a similarity match ranked first.
- why the title was referred to you: "models disagree" (the two earlier choices differ), "role
  outside recorded family" (the earlier model chose a role of another family) or "no match" (the
  earlier model found no listed role that fits).

You may only pick a role of the record's own job family. Such a role is listed with a number; a
role of another family is listed with "-" and cannot be picked.

Two searches are available. You may run at most 4 searches in all, one at a time:
- search_catalog finds roles in the whole catalog that are similar to a query.
- search_approved_titles finds job titles that a person has already mapped to a role.
A search can show roles that were not on the shortlist. A new role of the record's own family
gets the next free number, and you may pick it. Search when the shortlist does not settle the
case; you may also answer at once.

When you have decided, call no tool and answer with one JSON object:
- "answer": "role" when one numbered role is the same job as the record describes; "escalate"
  when none is, or the record says too little, so that a person decides.
- "number": the number of the role you picked; null when you escalate.
- "quote": with "role", the words of the record that your pick rests on, copied character for
  character from the title or the description: whole words, at least 3 characters and at most
  25 words. With "escalate", an empty string.

The level of a job (junior, senior, lead, a numeral such as II) is set elsewhere from the pay
grade: ignore it. Never write a role name or a title of your own.
```

A case from the live run, with eight of the ten roles left out here:

```text
Title: junior paralegal
Family: LEG
Department: Legal Affairs
Description: Researches precedents and keeps case files in order.

Roles:
- Accounts Payable Clerk (family FIN): Processes supplier invoices and schedules payments.
   (eight more roles of other families, each listed with "-")
1. Paralegal (family LEG): Prepares legal documents and supports counsel on filings. [the ML model's first choice]

Referred because: no match
```

The model searched the catalog for "paralegal legal research case files", got five lines back
with Paralegal first, and answered number 1 with the quote "Researches precedents and keeps
case files in order", which is in the record. The rules accepted it.

| Limit | What enforces it |
|---|---|
| Two read-only searches and nothing else | Only two tools are defined, each taking one text query. Any other tool is refused. |
| At most four searches, one a turn | The request allows one tool call a turn at most, and none once four searches are used. A fifth is refused, and a turn that asks for two is an invalid answer. |
| Only a role of the record's own family | Only such a role is given a number, the answer is mapped against the numbered roles alone, and the rules check the family again. |
| The answer is `role` or `escalate` | The schema allows nothing else. |
| A quote with every pick | The same quote check as for the LLM. |
| No generic family role | Never searched for, numbered or shown. |
| The answer is checked like the LLM's | Every rule applies again except agreement. Any failure sends the title to a person. |

**How much the agent decides** is set by the wave, not by the prompt. In Wave 1 it only
recommends: an HR SME approves every title on the list. In Wave 2 an answer that passes the
rules puts the title on the agent-settled path, which is audited before release.

## 3. Record and replay

Live mode and replay build each request with the same code. Its key is the SHA-256 of the
step, the model name, a key version and the whole request: system prompt, tools, tool choice,
schema, token limit and every turn. Replay builds the request again and looks the key up. A
miss stops the run, and no other model answers in its place. An entry of a recording holds
the key and its version, the step, the model and the model ID the API reported, the date, the
parsed answer or tool request, the stop reason and the token counts, with the cleaned title
and family code as a hint for a reader. It never holds the system prompts, the rest of the
request or a credential. The model setting has no default, so no paid model runs unless it is named.

## 4. What the live run showed

- **The picks were sound:** the LLM's pick was right for 36 of the 40 golden-set titles.
- **The quote instruction did not hold.** About a third of the quotes in Wave 2 were not word
  for word from the record. Most copied a listed role's definition, or joined the title or the
  description to it. Code sent every one to a person, which roughly doubled the titles sent to
  a person.
- **A clearer wording helped only partly** when tried outside the program on a small sample.
  It was not adopted: adopting it means recording again, and the owner chose to keep the one
  planned live run and report this as a finding.
- **The agent seldom searched,** and answered most of its cases at once.

`docs/live-run-observation.md` has the figures.

## 5. Known limits

- **The quote check is looser than the instruction.** The prompt asks for words of the title
  or the description; the check accepts any passage of the record block, its labels and its
  family and department lines included. It shows that a quote was copied, not that it is good
  evidence.
- **The record's text is not fenced off.** A description goes into the prompt masked and cut,
  otherwise as it is, and neither prompt tells the model to treat it as data. It also feeds the
  ML match, so the two models agreeing is no independent check against hostile text. What still
  stands behind a steered pick is the rules (a catalog role of the recorded family, never a
  compliance-flagged one, or a person) and the audit. This is now tested offline, with scripted
  answers and the stand-ins in place of a hosted model: `docs/hostile-text.md` says what holds
  and what does not.
- **The evidence is one run on mock data,** whose descriptions are written from the same duties
  as the role definitions, which makes a copied definition likely. The first change would be
  the quote wording, and its failure rate is a number to measure in a pilot.

## Check it

```sh
python3 -c "from titlestd import recording; print(recording.CHOOSE_SYSTEM, recording.REVIEW_SYSTEM, sep='\n\n')"
python3 -m titlestd run --llm replay --out-dir /tmp/titlestd-replay
```
