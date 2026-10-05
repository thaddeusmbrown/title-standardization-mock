# Build decisions

`docs/design.md` is the design. It is firm, and where anything here seems to differ from it,
the design wins. This file records what the design leaves open and how each point was settled,
so that every part of the build makes the same choice.

How to read it:

- **Owner decisions** are choices the repository owner made. They are applied throughout.
- **Q01 to Q40** are the open points, one entry each: the point, the decision, and its basis.
  Code and commit messages cite them by number.
- **S01 to S16** are the contracts settled while the build was cut into parts: names, formats
  and interfaces that several parts share.
- **Diagram rules not to copy**, **Arithmetic checked** and **Library facts** are reference
  notes.
- **How the build is cut** says which branch builds what, which files it owns and which tests
  it owns: the eight of section 2.7 and every test or deliverable that an entry decides. It
  also says what crosses a branch boundary and what the README must say.

Section numbers such as 1.8 or 2.4 refer to `docs/design.md`. "The mini data set" is the small
hand-written data set in `tests/fixtures/mini`.

---

## Owner decisions

1. **Local only.** Nothing is pushed and there are no hosted pull requests. Each component is
   built on its own feature branch and merged into main locally with a merge commit (`--no-ff`)
   after the check command passes on the branch and again on the merge result. A CI workflow
   file is committed and takes effect only when the owner pushes.
2. **One live recording.** The build makes one live run on `claude-haiku-4-5` and commits the
   recording. The program reads the key only from the `ANTHROPIC_API_KEY` environment variable,
   as the design says, and never from a file; on the build machine the variable is set from a
   file outside the repository, for the recording step only. Before the full run there is a
   smoke run on about 10 titles. The request cap is a setting with default 3000. At most two
   full live runs are made.
3. **Leftovers come from a seeded "could not decide" rate** for simulated reviewers (setting
   `undecided_rate`, default 0.10). The truth file keeps one true role per title.
4. **`embedder = local` uses model2vec** with `minishlab/potion-base-8M`. No PyTorch anywhere.
5. **The invented name list and location list are module constants** inside the package, not
   CSV files, so the data files are exactly the seven listed in section 2.3.
6. **The optional extras are named `live`** (the anthropic SDK) **and `local`** (model2vec).
7. **Commit messages end with the co-author and session trailer lines** of the coding tool.
   Section 2.8 covers real companies, clients and people, not technology
   vendors or the coding tool.
8. **The configured model string is the one that counts.** Recording keys and the
   "recorded <date>, <model>" label use the configured `llm_model` string exactly as written.
   Each recorded entry also stores the model ID the API reported in its response (the alias
   `claude-haiku-4-5` currently reports `claude-haiku-4-5-20251001`), as information only.
9. **Python 3.12 or later.** The package is the directory `titlestd` at the repository root,
   run as `python3 -m titlestd run` from a fresh clone with nothing installed.

---

## Open points and decisions

### Q01. Where the leftovers at the close come from

**Open point.** The close gives a generic family role to "a title nobody could decide" (1.8,
1.10), and the mock includes the generic role and its cap (2.2). But people are simulated from
a truth file that holds one true role per title (2.3, 2.4), so a simulated reviewer always has
an answer, right or wrong. The design names no way for a queue title to stay undecided, and
does not say whether simulated people clear the queue at the close at all.

**Decision.** Simulated people do clear the queue at the close. A setting `undecided_rate`,
default 0.10, makes the simulated reviewer return "could not decide" for a queue title when a
keyed hash of (seed, family code, cleaned title) falls under the rate, so the result does not
depend on call order. It applies only to review-queue decisions at the close, never to Wave 1
approvals, audit labels or golden-set labels. The truth file stays exactly as 2.3 defines it.
Undecided titles are the leftovers the close works on (order and cap in Q25). Setting the rate
to 0 turns the mechanism off. The rate is not tuned to force an outcome: the default run needs
only a non-empty generic row for the fixed seed; cap overflow and the two bars are proven by
the unit tests for the cap (2.7) plus one integration run at a high rate (for example 0.5).
The README says this is a simulation device. "No suitable role" is never the source of
leftovers.

**Basis.** 1.8: "At the close, a title nobody could decide takes its family's generic role,
marked provisional and capped at 2% of employees." 1.10: "People clear the queue; leftovers
take a generic family role; final release". 2.2 puts simulated people, the generic family role
and its cap, and the close in the mock. 2.4: "People are simulated from the truth file, with a
small seeded error rate". 2.3: "one true role per title within a family". 1.8 sends a "no
suitable role" title to the catalog owner, and the preamble calls giving it a generic role a
diagram error. The design fixes what happens to an undecided title but names no mechanism and
no rate; a seeded rate beside the seeded error rate is the smallest addition, and it is a
setting. Owner decision 3.

### Q02. Whether a real recording ships, and how the live run is made

**Open point.** 2.5 offers `llm = replay` and says recordings come only from live runs and
are never written by hand. It does not say whether the repository ships a recording so that
`llm = replay` works on a fresh clone, which model, embedder and seed a shipped recording is
made with, or what the spend limit is.

**Decision.** The build makes one live run on `claude-haiku-4-5` and commits the recording.
The program reads the key only from the environment variable `ANTHROPIC_API_KEY` and never
from a file; on the build machine the variable is set from a file outside the repository, for
the recording step only. A dedicated key with a spend limit of about USD 20, revoked after the
recording, is the suggested way to supply it. First a smoke run on about 10 titles, costing
cents, proves the adapter; then one full recording is made at the very end on frozen data and
prompts (default data and seed, `embedder = offline`, `llm_model = claude-haiku-4-5`),
committed, and replayed with zero misses in the final gate. The request cap is the setting
`request_cap`, default 3,000: a full run is estimated at 1,000 to 1,500 requests, and a cap of
1,500 would abort a legitimate run after the money is spent. At most two full live runs are
made. The key is never written to `.env` or any file in the repository and never echoed; a
test scans tracked files and recordings for `sk-ant-`. One documented command,
`python3 -m titlestd record`, lets the owner record again later.

**Basis.** 2.1: "Show the whole idea in under a minute, offline, on made-up data." 2.5: the
`llm` values and the empty default of `llm_model`; "The only secret is `ANTHROPIC_API_KEY`,
read by the SDK from the environment and only when `llm = anthropic`. It is never written to
a file in the repository."; "In live mode, record each answer in a file ... In replay mode a
miss is an error; never fall back silently. Recordings are never written by hand." 2.8: "No
secrets. `.env` is ignored. No keys in recordings." Only a live run can create a recording.
`claude-haiku-4-5` is a valid model ID at USD 1 input and USD 5 output per million tokens, so
a full recording costs in the order of USD 1 to 5. Owner decision 2.

### Q03. "No suitable role" in the mock

**Open point.** "A title marked 'no suitable role' goes to the catalog owner, who adds a role
or names one" (1.8), and the preamble says the diagrams are wrong to give such a title a
generic role. Part 2 lists it neither in nor out of the mock. The truth file gives every title
a true role, so a simulated reviewer has no ground for the mark. Nothing says who marks a
title, how often, what a simulated catalog owner does, or where the title ends if the owner
has not acted by the close.

**Decision.** The route is built, with a concrete trigger. The simulated reviewer returns the
decision "no suitable role" only when the truth file has no row for the title; a data test
asserts that every title in the default data has a truth row, so the default run never
produces the mark. A marked title is assigned to "Catalog owner (simulated)", written to the
review-queue file with the reason "no suitable role, waiting for catalog owner", skipped by
the generic-role step, not counted against the 2% cap, and counted in the "left on legacy
title" row. The catalog is read-only during a run: no role is added. Unit tests on a small
fixture cover the route, the never-generic rule and the cap exclusion. There is no seeded
share of such titles, and the mark is never the source of the leftovers that take the generic
role (Q01).

**Basis.** Preamble: the diagrams "give a 'no suitable role' title a generic role", which the
design rejects. 1.8: "A title marked 'no suitable role' goes to the catalog owner, who adds a
role or names one." 2.3 with 2.4: one true role per title, and people simulated from it, so no
simulated person has grounds for the mark on the default data. 2.2 and the table rows of 2.6
give it no place of its own. The rule is settled (to the catalog owner, never generic); a
guarded, tested route is cheap, and showing the case later is a data change (remove a few
truth rows).

### Q04. The ML-alone route

**Open point.** "The ML model's choice may stand alone only if the margin clears a cut-off and
the chosen role has at least three human-approved examples" (1.6 step 4), and "The ML-alone
route is built and off by default" (2.4). But 1.1 says a title goes out without a person only
if the LLM and the ML model agree or an agent settles their disagreement; the vocabulary and
the table of 2.6 allow only two machine paths; and the preamble says three audited paths is a
diagram error. The design does not say whether a title decided by the ML model alone is
released, audited or sent to a person, which path and row it belongs to, or what the default
margin cut-off is.

**Decision.** "Stand alone" means without the LLM, not without a person. With the switch on
(`ml_alone`, default off), a look-up miss whose ML first choice clears `ml_margin_cutoff`
(placeholder default 0.10, documented as untuned), has at least `ml_min_examples` (3)
human-approved examples and is in the recorded family skips the LLM. The catalog and
compliance checks still run. The title goes to the review queue with reason "ML alone" and the
ML role as the proposal; a simulated person decides it, status Approved, counted in the
"review queue" row (in Wave 1 the SME approves it like any Wave 1 title). There is no third
path, no eighth row and no path audit, and the title is never counted under "models agree".
Margin and example count are stored for every title either way. Unit tests switch the route
on.

**Basis.** 1.1: "A title goes out without a person only if the LLM and the ML model agree, or
an agent settles their disagreement, and only after HR has audited a random sample of that
machine path." 1.2: a machine path is "models agree" or "agent-settled". Preamble: the
diagrams "list three audited paths where there are two". 1.9: "The ML margin and the example
count only decide who looks next. Trust comes from people's labels, never from a model's own
score." 1.6 step 4 and 1.10 (after handoff "A person approves every new title"). 2.4: "The
margin cut-off and the three-example rule are settings. The ML-alone route is built and off by
default." Since the title cannot go out without a person, it goes to one.

### Q05. The load file, the reverse file and the exceptions list

**Open point.** The load file is "one row per employee: the new titles and their status" and
the reverse file is "how to restore each employee's legacy title" (1.6 step 9). No file type,
names, folder or columns are given. A stopped path is left "on legacy titles" (2.4) and "every
title carries a status" (1.1), yet only three statuses exist and none fits an employee left on
a legacy title. It is unstated whether such an employee appears in the load file and whether
the reverse file covers them; where a no-grade employee "is listed" (2.4); whether the
provisional mark, its error rate and the simulated mark are in the load file; and what
"applying the reverse file" means in code.

**Decision.** UTF-8 CSV with a header row, written to `out/`, rows sorted by `employee_id`.

- `load.csv` holds one row per employee who receives a new title: `employee_id`, `role_id`,
  `long_title`, `short_title`, `level` (blank when there is no grade), `status` (Approved,
  Confirmed or Provisional only), `path` (the 2.6 row that released it), `error_rate` (the
  path's wrong/audited, filled only when Provisional comes from an audited path), `no_grade`
  (0/1), `decided_by` (on every Approved row: the persona who decided it during the run, or
  the fixed label of S11 on a look-up hit), `simulated` (1 on every decision a simulated
  person made during the run). A no-grade employee is in `load.csv` with the catalog's
  no-level title.
- `reverse.csv` holds `employee_id` and `legacy_job_title` for exactly the same employees:
  the `job_title` as received, byte for byte.
- `exceptions.csv` (`employee_id`, `legacy_job_title`, `job_family_code`, `reason`,
  `released`) lists every employee left on a legacy title (`released` = 0; reason: stopped
  path, over the generic cap, undecided widely held or flagged title, no suitable role) and
  every no-grade employee who received a new title (`released` = 1, reason "no grade"); an
  employee is listed once, so one with no grade who stays on a legacy title has the
  `released` = 0 row only. Both counts are printed under the table.

The footer's coverage by status prints the three statuses plus "left on legacy title" (the
existing table row), which together sum to 100%. One function
`apply_reverse(load_rows, reverse_rows)` returns `employee_id` to restored title; the test
asserts it equals the employees file's `job_title` for every load row, that both files hold
the same ids, and that `load.csv` plus the `released` = 0 exceptions cover every employee
exactly once.

**Basis.** 1.6 step 9; 2.7: "No title outside the catalog can reach the load file" and
"Applying the reverse file to the load file restores every legacy title"; 2.4: "A stopped path
is left on legacy titles unless a setting releases it" and "No grade: the title with no level
word, and the employee is listed"; 1.9: three statuses only, and "Provisional: released,
marked, with its measured error rate"; 2.4: "every simulated decision is marked as simulated";
2.6 has the row "left on legacy title". A load file with every employee and a fourth status is
ruled out by the three statuses and by the catalog-only rule, and employees left on legacy
titles already have their own table row. Only file type, names and columns were open.

### Q06. Seeds, fresh state and where randomness comes from

**Open point.** The design says "seeded" (2.4), "fixed random sample" (1.2), "Sample 60 titles
per path" (2.4) and "shuffled" (1.6), but the settings table has no seed. Not stated: one seed
or several, its default, whether the mock data is generated on every run or committed, which
state store is used, and whether a run starts from empty state.

**Decision.** One setting `seed`, default 20260101. Every random use builds its own
`random.Random` from a SHA-256 of (seed, stream name, item key): golden-set draw, audit sample
per path, role shuffle per title, simulated-person error per title. Never the global random
module, the built-in `hash()`, or set iteration order. State is `out/state.sqlite`
(standard-library `sqlite3`), deleted and recreated at the start of every run and loaded with
only the seed look-up; all outputs go under `out/`. Mock data is CSV in `data/`, written by a
generator with its own fixed seed (separate from the run seed, so changing `seed` does not
change the data) and committed; one test regenerates into a temporary folder and compares
byte for byte. An integration test runs the command twice and compares the printed table and
the three output files byte for byte, so no output may contain the wall-clock time.

**Basis.** The settings table of 2.5 has no seed, although the design requires seeded
behaviour (1.2 "A fixed random sample"; 2.4 "with a small seeded error rate"). 2.6: "One
command runs set-up, Wave 1, Wave 2 and the close on the mock data and prints one table", so
each run starts at set-up, from fresh state, on data files that already exist. 2.1: "a local
SQLite file or plain files for state". The repository's `.gitignore` ignores `out/` and
`*.sqlite` and does not ignore CSV files. This is the conventional reproducibility set-up;
the seed number itself is arbitrary.

### Q07. Golden-set titles: learning and release

**Open point.** 1.2: the golden set "never becomes a look-up entry or an example". 1.6 step
10: "Titles a person approved or labeled join the look-up table and the ML model's examples",
and golden-set titles are labeled by HR SMEs and can be labeled again in an audit, a fully
labeled small path or the review queue. Beyond that contradiction, the design does not say
whether golden titles go through Wave 2 like any long-tail title, whether the audit may draw
them, or what their employees receive. No table row is named for them.

**Decision.** The specific rule (1.2 and 2.7, both "never") overrides the general sentence in
step 10. For release, golden titles are ordinary long-tail titles: scored in Wave 2, routed,
audited and released like any other, landing in the ordinary rows. The single function that
adds look-up entries and examples refuses golden-set keys unconditionally, whatever produced
the human decision (golden label, audit label, small-path labeling, queue decision); the
person's decision is still used for that title's release. Golden titles stay in the audit
sampling frame. The 2.7 test asserts the guard after a full run.

**Basis.** 1.2: "A fixed random sample of long-tail titles that HR SMEs label without seeing
the system's answer. It is re-scored at every wave and never becomes a look-up entry or an
example." 2.7: "Golden-set titles never enter the look-up table or the examples." 1.10: "The
long tail is scored for release." The system "maps every employee onto a standard role"
(opening paragraph). 2.6 has seven rows and none for the golden set. 1.9: the audit is "a
random sample of 200 titles from it" (the path), with no exclusion stated. Holding golden
titles out of release would need a row and a status the design does not have.

### Q08. Golden set: size, frame and scoring

**Open point.** 2.1 sizes everything except the golden set, and 2.4 rescales the audit but not
the golden set. "Long-tail titles" (1.2) does not say whether the frame includes seed look-up
hits, titles with a person's name, or titles with no family code. "It is re-scored at every
wave", but the footer asks for one "golden-set accuracy of the ML model's first choice and of
the LLM's pick" without saying which wave, or whether Wave 1 scoring happens before or after
Wave 1 approvals join the examples; the dry run, the only Wave 1 scoring of long-tail titles,
is out of the mock. How a "no match" or "insufficient data" answer counts is not stated.

**Decision.** Setting `golden_size` = 40. The set is drawn at set-up with the run seed from
title rows that are not on the Wave 1 list, miss the seed look-up, have a usable family code
and carry no person's name. It is labeled blind by the simulated SME in Wave 1. Golden titles
still go through Wave 2 and are released like any other long-tail title; the only special
handling is that no label or decision for them is ever written to the look-up or the examples
(one guard in the learning step, asserted after every stage). The set is scored three times:
Wave 1 (with the examples the Wave 1 proposals used, before Wave 1 approvals join; one extra
embed-and-choose pass over the 40 because the dry run is out), Wave 2 (the answers Wave 2
actually produced, no extra calls) and the close (after queue decisions join, one extra pass).
The footer prints, per stage, ML first choice n/40 and LLM pick n/40. The denominator is all
golden titles; "no match" and "insufficient data" count as wrong and are also shown as their
own count.

**Basis.** 1.2 (definition, "re-scored at every wave"); 1.10: "golden set drawn" in set-up
and "The golden set is labeled" in Wave 1; 2.2: the mock contains Wave 1, Wave 2 and the
close, and the dry run is out; 2.4: "Wrong ... means the machine's role differs from the
simulated SME's label for that title"; 2.6 footer; 2.7. The preamble forbids copying the
diagram's 300. No size is given, so it is a setting. Scoring at every stage follows "every
wave"; the size does not shift path sizes, because golden titles are still decided and
released.

### Q09. What the truth and seed files are keyed on, and what "about 400 titles" counts

**Open point.** The unit of decision is "one row per cleaned title within a family" (1.4) and
the look-up is an "Exact match on cleaned title plus family" (1.6), but the seed and truth
files are keyed on `job_title, job_family_code` (2.3), the same column name as the raw
employees field. Not stated: whether those files hold raw or cleaned titles; what applies when
several raw spellings that clean to one row carry different true roles; whether "about 400
titles" and "the 40 most widely held titles" (2.1) count raw strings or cleaned decision rows;
whether a true role may be generic or outside the title's recorded family; and how a title
with no usable family code is keyed (Q26).

**Decision.** Seed and truth files hold the raw `job_title` and `job_family_code` exactly as
they appear in the employees file, one row per distinct raw pair; the shared prepare function
cleans them on load. The generator guarantees, and a data test using the real prepare function
asserts: every cleaned (title, family) row has exactly one true role across its raw spellings;
every seed row cleans to a row present in the employees file and its role equals the true
role; true and seed roles are never generic and, where the family code is valid, are in that
family. "About 400" counts cleaned decision rows (test: 360 to 440), with more raw spellings
than that; the Wave 1 list, the audit samples and every "titles" count in the table count
decision rows. For a blank or invalid family code the truth row is keyed on the code as
recorded and may name any non-generic role (the rest is Q26).

**Basis.** 1.2: "Title | What the employer has today: free text on an employee record." 2.3
uses the same column name `job_title` in Employees, Seed look-up and Truth, so the two files
hold titles as recorded. 1.6 step 2: the look-up matches on the cleaned title, so the seed is
cleaned on load. 1.4: "one row per cleaned title within a family, with its headcount." 2.3:
"one true role per title within a family". 1.2: only code assigns a generic role, so a seed or
true role, being a person's decision, is never generic. Only the cleaned-row reading of "about
400" keeps both machine paths above the 60-title audit line that 2.4 assumes. The band is wide
because the count depends on both the generator and the cleaner.

### Q10. The agent's case, its two searches and the numbering

**Open point.** The agent has "two read-only searches: the whole catalog (generic family roles
excluded), and similar human-approved titles" and "code numbers every in-family role it has
been shown" (1.6 step 6). `review(case, tools)` has no defined case or tool shape. Open: the
search arguments; the scoring and ranking; how many results; what each result shows; whether
out-of-family results are returned without a number or withheld; whether the roles already in
the case count as shown and are numbered before any search; and whether numbers stay fixed
across searches.

**Decision.** The case is the masked record as the LLM saw it (cleaned title, family code,
department, description text), the ten shortlisted roles in the order the LLM saw them with
the LLM's answer and the ML first choice marked, and the referral reasons. The tools are
`search_catalog(query)` and `search_approved_titles(query)`: one free-text argument, no family
filter, ranked by cosine similarity using the active `embed` call, ties by `role_id` or title,
top 5 (setting `agent_search_k`). Catalog results show role name, family code and definition;
approved-title results show title, family code and role name. No similarity scores appear in
the prompt. Shared code drops generic roles and keeps one running list: in-family shortlist
roles are numbered first, each new in-family role gets the next number on first appearance,
numbers never change and are never reused, and out-of-family roles are shown without a number.
At most four search calls in total, enforced by shared code.

**Basis.** 1.6 step 6: "it has two read-only searches ... It answers the same way as the LLM:
code numbers every in-family role it has been shown, and the agent returns a quote and a
number, or escalates. Picking the LLM's role confirms it". 1.4: "the agent may only pick
in-family roles". 2.4: "at most four search calls, then one structured answer". 2.5:
"Masking, numbering, the quote check, the search cap and the number-to-role mapping stay in
shared code". These settle that the catalog search is not family-filtered ("the whole
catalog"), that out-of-family roles are shown but not numbered, that roles in the case count
as shown, and that generic roles never appear. Argument shape, result count, ranking and the
fields shown were open. This is fixed before any recording exists, because changing it later
changes prompts and so recording keys.

### Q11. Which description and department form "the record" of a title row

**Open point.** Employees each have their own `job_description` and `department`, with "some
empty descriptions" (2.3), but the ML embedding, the LLM and the quote check all work on "one
row per cleaned title within a family" (1.4). The design never says which employee's
description or department represents the row when they differ, what fields "the masked
record" contains, or what "No description: use the title alone" (2.4) means when only some of
a row's employees have one.

**Decision.** The prepare step builds one record per title row, once, and every later step
(embedding, LLM, agent, quote check, reviewers) reads only that record: cleaned masked title,
family code, department, description. Description is the most common non-empty masked
description among the row's employees, ties going to the one held by the lowest `employee_id`,
cut to its first 200 words after masking. Department is the most common non-empty department
with the same tie-break. The row uses the title alone only when no employee in it has a
description. The result must not depend on file order (test: shuffle the employees file and
get identical records). The generator gives one description per raw title as the norm, plus a
handful of rows with differing or partly empty descriptions so the rule is exercised.

**Basis.** 1.4: "Cleaned title plus the first 200 words of the job description | The ML
model's embedding"; "Department | Context for the LLM, the agent and reviewers"; "The unit of
decision is one row per cleaned title within a family, with its headcount." 1.6 step 4: "The
LLM sees the masked record and the ten roles". 1.7: "The LLM's quote appears word for word in
the masked record". 2.3: department and description are per employee. 2.4: "No description:
use the title alone." Nothing says which employee's text represents a row; a deterministic
most-common pick is the conventional way to represent a grouped row. The point that matters is
that the record is built once and shared, so the text embedded, shown and quote-checked is the
same.

### Q12. The offline embedder

**Open point.** The default embedder is only "`offline` (built-in, no install)" (2.5). 2.4
describes the stand-in LLM and agent (word overlap) but not the embedder. Not stated: its
features, vector size, weighting, hashing, and whether an approved title's example vector is
the title alone or the title plus description. It must use the standard library only, give
meaningful cosine similarity between a query and a role's "name and one-line definition", and
give the same vectors in every process and on every machine.

**Decision.** Hashed character n-grams, standard library only. Case-fold; split into
`[a-z0-9]+` tokens; wrap each as `<token>`; features are all 3-, 4- and 5-character n-grams
of the wrapped token; weight 1 + ln(count); bucket = an 8-byte `hashlib.blake2b` digest taken
modulo `embed_dim`; L2-normalise; keep vectors as sparse dicts and compute cosine as a sparse
dot product. No fitted weights (no IDF), so vectors never change as approvals are added. An
approved title's example vector is the same vector it was queried with (cleaned title plus the
first 200 description words; the title alone when the description is empty). `embed_dim`
defaults to 32,768, not 2,048.

**Basis.** 2.5: "`offline` (built-in, no install)". 2.1: "the standard library by default".
1.5: "The embedding model is off the shelf and is not trained" and "The ML model gives the
same answer every time on the key fields". 2.4: the stand-in LLM and agent "decide by word
overlap after abbreviation expansion", so the embedder must be a different method. 1.4 and 2.4
fix the texts and the score. With sparse vectors a small dimension saves nothing: in a probe,
2,048 buckets doubled the similarity of unrelated texts (about 0.20 against 0.10 at 32,768)
through hash collisions, which eats into the margin; 400 by 350 sparse cosines took about 4
seconds in pure Python, so the run stays well under a minute.

### Q13. A seed look-up hit that is also widely held, and the status of look-up hits

**Open point.** "A hit takes that role with no model" (1.6), but "The most widely held titles
always go to a person" (1.4) and each is "approved by an HR SME" (1.8). Nothing says which
applies when a seed title is in the top-headcount group, which row it lands in, or whether the
Wave 1 list is drawn before or after removing hits. The status of a look-up hit is also never
stated.

**Decision.** The look-up runs first for every row; a hit takes its role with no model, has
status Approved, is counted under "look-up hit" and is not approved again. The rule that
widely held titles go to a person is met because a look-up entry is already a person's
decision. Seed entries are human-decided titles, so they are ML examples from the start. The
Wave 1 list is the top N titles by headcount over all title rows (ties by family code, then
cleaned title); a seed hit inside it stays a look-up hit, so the Wave 1 row may show fewer
than N. The data generator puts the seed titles in the long tail, so the overlap never occurs
in the default data; one unit test covers the overlap case.

**Basis.** 1.1: "Cheapest check first. An exact look-up, then an ML match, then an LLM, then
an agent, then a person." 1.6 step 2: "A hit takes that role with no model." 1.2: "Look-up
table | Titles a person has already decided, each mapped to a role." 1.9: "Approved | A named
person decided it". 1.10: after handoff "Known titles hit the look-up. A person approves every
new title", so known titles are not approved again. 2.4: "The seed look-up exists so that the
look-up step has hits to show."

### Q14. Cleaning and name masking without a masking service

**Open point.** Step 1 must "strip punctuation, locations and requisition numbers" and "mask
names"; "A title that contains a person's name goes to a person and to no model" (1.4), and
masking stays in shared code (2.5). Production uses an open-source name masker (1.11), but the
mock is "the standard library by default" and the files of 2.3 supply one reference list,
abbreviations: there is no names list, no location list and no requisition format. Not stated:
how a name or a location is recognised offline; in which fields; what replaces it; whether
emails are masked; whether masking runs before or after cleaning; whether the row is keyed on
the masked or unmasked title; and whether, once a person approves it, it joins the ML model's
examples, which would mean embedding it.

**Decision.** Small invented reference lists, held as module constants in one shared module
of the package (`titlestd/reference.py`) and read by both the generator and the prepare step:
given names and surnames, about 10 locations, and two fixed requisition patterns (`REQ-` plus
4 to 6 digits, `#` plus 4 to 6 digits; at least four digits so level numerals such as "II" or
"2" are never removed). They are not data files: the data files are exactly the seven of 2.3.
A test asserts the lists share no word with role names, abbreviations, level words,
departments or each other. Titles follow the design's order: clean (lower-case, strip
punctuation, trailing location tokens and requisition patterns, expand abbreviations), then
mask, then group. Matching is on lower-cased whole tokens, so it works after cleaning; an
adjacent given name and surname become one `[NAME]`, and a lone listed name token is masked
too. Descriptions and departments are not cleaned, only masked: emails by pattern to `[EMAIL]`
first, then names. A row whose title held a name is keyed on the masked cleaned title plus
family, goes straight to the review queue with reason "person's name", and `embed`, `choose`
and `review` are never called for it (test with spies). It is out of the golden-set frame. A
person's decision on it joins the look-up under the masked key but never the ML examples,
because joining the examples means embedding it. The README says the list stands in for a real
name masker.

**Basis.** 1.4: "Names are masked before any text reaches a model. A title that contains a
person's name goes to a person and to no model." 1.6 step 1: "Clean each title (lower-case,
strip punctuation, locations and requisition numbers, expand a small abbreviation list such as
sr to senior), mask names, group to one row per title and family." 1.6 step 10 (learning).
1.11: the name masker is a production component. 2.1: "the standard library by default". 2.3:
"location and requisition suffixes ... a few with a person's name ... All names, emails and
companies are invented; use the reserved `example.com` domain." 2.5: "Masking ... stay in
shared code". A shared invented list is the only approach that is exact, standard-library only
and testable. The rule of 1.4 is met in the fail-safe direction and overrides the general
learning rule for the embedded examples. Owner decision 5 makes the lists module constants.

### Q15. The structured answer and the quote check

**Open point.** "It must answer with a number from 1 to 10, or 'no match', or 'insufficient
data', plus a short quote from the record as evidence" (1.6) and "The LLM's quote appears word
for word in the masked record" (1.7). Open: the answer's fields; whether a quote is required
with "no match", "insufficient data" or an agent escalation; how "word for word" is compared
and the minimum length; which text is "the masked record"; and what quote the offline stand-in
gives, since it matches on abbreviation-expanded words that may not occur literally in the
record.

**Decision.** `choose` returns {answer: `role` | `no_match` | `insufficient_data`, number:
integer 1 to 10 or null, quote: string}; `review` returns {answer: `role` | `escalate`,
number: integer or null, quote: string}. The quote is required and checked on every `choose`
answer, including "no match" and "insufficient data"; for the agent it is required and checked
with a number, and `escalate` needs none. The check, in shared code: case-fold and collapse
whitespace on both sides, then the quote must be a substring of the record block exactly as
rendered for the model, starting and ending on a word boundary, at least 3 characters
including a letter, at most 25 words; an empty quote fails. The record block is built once by
shared code (masked title, family, department, masked description) and the same string serves
the prompt and the check. The stand-in quotes a verbatim slice of that block (the title as
shown, otherwise the description words that matched), never an expanded form absent from the
block. In the live schema 1 to 10 is expressed as an enum, and range and quote length are
checked again in code; an out-of-range number fails the catalog check and goes to a person.

**Basis.** 1.6 step 4 (quoted above). 1.5: "Structured output and a required quote keep it
from inventing." 1.7: "The LLM's quote appears word for word in the masked record | A person",
with no condition on the answer, so the quote is checked always rather than only with a
number. 1.6 step 6: "the agent returns a quote and a number, or escalates." 1.7: "If several
fail, a person beats the agent." 2.5 keeps the quote check and the number-to-role mapping in
shared code. The rule that "no match" goes to the agent stays reachable because the stand-in
always quotes literally; a "no match" with a missing or invented quote goes to a person, which
is the safe direction. Structured output on the hosted model supports `enum` but not
`minimum`, `maximum` or `minLength`.

### Q16. The printed table, its footer and the review-queue file

**Open point.** 2.6 names seven rows, the columns and a footer, but not their rules. Rows mix
how a title was decided with how it ended, so a title can fit two rows. Not stated: whether
each title sits in exactly one row and shares add to 100%; what the "review queue" row means
after the close; what "audited", "wrong" and "status" show on rows that are not machine paths,
on a fully labeled small path, or on a path row holding both audited and unaudited titles; the
numerator of "the share of titles sent to a person"; whether "coverage by status" is by
employee and includes employees left on legacy titles; and what the review-queue file holds.

**Decision.** Rows partition the decision units (cleaned title within family): each unit is
in exactly one row, titles and employees sum to the totals and shares to 100%, and a test
asserts it. First match wins:

1. look-up hit;
2. approved in Wave 1;
3. models agree, and
4. agent-settled: titles released through that path, audited and unaudited together;
5. review queue: sent to the queue and decided there by a person;
6. generic family role: leftovers given the generic role at the close;
7. left on legacy title: every unit with no new title, that is leftovers barred or over the
   cap, plus the unaudited titles of a stopped path that the setting did not release.

Audited titles take the SME's label with status Approved in every path, so a stopped path's
row keeps its audited titles. "Audited" and "wrong" are filled only on the two path rows.
"Status" on a path row is the audit outcome: Confirmed; Provisional with the measured error
rate; Stopped with the number of titles moved to legacy (or "Stopped, released as provisional"
when the setting is on); "all labeled" for a path of 60 or fewer. On other rows it is the
status of the row's titles (Approved; Provisional for generic; none for legacy).

Footer: coverage by status is by employee (Approved, Confirmed, Provisional, and "left on
legacy title" as in Q05) and sums to 100%. "Sent to a person" = Wave 1 list titles plus titles
routed to the review queue, over all titles; blind audit and golden-set labels are given as a
separate count on the same line.

Review-queue file: one row per title that entered the queue (the queue as it stood after Wave
2) with cleaned title, family, headcount, reasons, ML first choice, LLM pick, agent
recommendation and, since people are simulated, the simulated decision marked as simulated and
the final outcome.

**Basis.** 2.6: rows "look-up hit, approved in Wave 1, models agree, agent-settled, review
queue, generic family role, left on legacy title"; columns "titles, employees, share of
employees, audited, wrong, status"; the footer; "It also writes the load file, the reverse
file and the review queue." 2.4: in a small path "each takes its label with status Approved,
and the path gets no status. A stopped path is left on legacy titles unless a setting releases
it." 1.9: Approved is "A named person decided it"; Confirmed is "No person read it". A "share
of employees" column only makes sense as a partition. A stopped path "is left on legacy
titles", the same phrase as the row name, so its unreleased titles belong in that row.

### Q17. The stand-in LLM

**Open point.** "The offline LLM and agent decide by word overlap after abbreviation
expansion" (2.4) is the whole specification. Not stated: which record fields are compared
against which role text; the score; the tie-break; when the stand-in answers "no match" and
when "insufficient data"; and how it builds the quote. A stand-in that copies its quote from
the record can never fail the quote check.

**Decision.** Content words are lower-cased tokens after abbreviation expansion, minus a fixed
stop-word list and the level words (ladder long and short words and level numerals). T =
content words of the cleaned title, D = those of the shown description, N_r and F_r = those of
role r's name and definition.

1. Answer "insufficient data" when the row has no description and T has fewer than 2 words
   (`standin_min_title_words` = 2).
2. For each of the ten listed roles, score_r = 2 x |T and N_r| + |(T or D) and (N_r or F_r)|;
   rank by score, then by |T and N_r|, then by lowest `role_id`, never by position in the
   shuffled list.
3. Answer "no match" when the best score is below `standin_no_match_below` (default 2: no
   role-name word in the title and at most one shared word anywhere). Otherwise answer the
   list number of the top role.

The quote is never empty and is copied character for character from the text shown: the
longest run of consecutive title words that matched the chosen role, else the same from the
description; for "no match" and "insufficient data" it is the whole title as shown. Thresholds
are named settings. The mock data includes about four titles built for each of "no match"
(wording shared with no role) and "insufficient data" (bare one-word titles with no
description). The failed quote and the out-of-range number are covered only by tests with an
injected answer; the stand-in has no fault injection.

**Basis.** 2.4: "Stand-in models never read the truth file. The offline LLM and agent decide
by word overlap after abbreviation expansion. The README says offline accuracy shows the flow,
not model quality." 1.6 step 4 requires both "no match" and "insufficient data". 1.7 and 2.7
(the quote rule and its test). Weights, ties and thresholds are constants that are cheap to
retune; what must be fixed is that ties never depend on the shuffle and that the quote is
copied from the exact text the quote check reads. Seeded fault injection would make the
stand-in dishonest.

### Q18. A reproducible shuffle and prompts that repeat

**Open point.** "The LLM sees the masked record and the ten roles, numbered and shuffled"
(1.6). Answers are "keyed by a hash of the step, model ID, prompt and request. In replay mode
a miss is an error" (2.5). No seed is given for the shuffle. The shuffled order is part of the
prompt, so part of the recording key. Also unstated: the order in which titles are processed,
and the moment approvals join the examples, both of which change later shortlists and agent
search results and therefore later prompts; and how a recording is tied to the data, seed and
embedder that shaped its prompts.

**Decision.** Sort the ten shortlist roles by `role_id`, then permute them with a
`random.Random` seeded from SHA-256 of (seed, "shuffle", family code, cleaned title), so the
order depends only on the title. Process titles sorted by family code then cleaned title.
Break every tie (nearest neighbours, shortlist cut, search results) on a stable key such as
`role_id`. Score a whole wave first, then apply that wave's approvals and labels to the
look-up table and the examples. The agent's numbering follows the order in which roles were
shown, which is deterministic under these rules. Each recording set carries a manifest (seed,
embedder, data hash, configured model, key version); replay compares it first and stops with a
message naming the difference.

**Basis.** 1.6 step 4 and 2.5 (quoted above). 1.5: "The ML model gives the same answer every
time on the key fields". 1.11: "One batch program", and the wave table of 1.10 describes batch
steps, not title-by-title updates. The replay rule requires prompts that repeat exactly, and a
per-title hash-derived permutation gives them without coupling one title's prompt to
another's. Changing the seed, the data or the embedder invalidates a recording, which the
README says.

### Q19. The stand-in agent

**Open point.** The agent is "a plain loop: at most four search calls, then one structured
answer" and decides "by word overlap" (2.4). Not stated: what the stand-in searches for and in
what order, how results become a pick, and what makes it escalate instead of picking. Because
it uses the same word overlap as the stand-in LLM, nothing in the design makes its answer
differ from the LLM's.

**Decision.** A fixed script, run through the same loop and shared code as the live agent.
Search 1: `search_catalog(cleaned title)`. Search 2: `search_approved_titles(cleaned title)`.
Search 3, only if fewer than two in-family roles have been shown so far and the record has a
description: `search_catalog(first 12 words of the masked description)`. Never a fourth call.
Score each numbered (in-family) role with the stand-in LLM's overlap function, plus a fixed
bonus worth one shared word when a similar approved title returned by search 2 maps to that
role. Answer with the top role if it shares at least one content word with the cleaned title
and strictly beats the runner-up; otherwise escalate. The quote is built the same way as the
stand-in LLM's: a verbatim span of the masked record.

**Basis.** 2.4 (word overlap; four search calls; the README statement on offline accuracy).
1.6 step 6: "the agent returns a quote and a number, or escalates. Picking the LLM's role
confirms it; picking another corrects it." 1.4: "the agent may only pick in-family roles".
The script differs from the stand-in LLM in three ways the design itself gives the agent
(in-family roles only, roles beyond the ten, evidence from approved titles), so it can
confirm, correct or escalate.

### Q20. The agent loop behind `review(case, tools)`

**Open point.** "The agent is a plain loop: at most four search calls, then one structured
answer" and "the search cap and the number-to-role mapping stay in shared code". Open: whether
`review` is one model turn driven by a shared loop, or the whole loop given capped tools; what
counts as a search call when the live model puts several tool calls in one turn, what happens
on a fifth, and whether zero searches is allowed; how the final structured answer is requested
when tools are present; whether the offline stand-in really issues searches through the capped
tools; and whether replay re-runs the recorded searches or replays only the final answer.

**Decision.** `review(case, tools)` stays the call behind the switch, its loop is written
once in shared code, and each backend supplies only `next_turn(transcript)`, which returns
either one search request (tool, query) or a final answer (number or escalate, plus quote).
Each executed search of either kind counts one toward four; zero searches is allowed. Shared
code runs the search, drops generic roles, and gives each newly shown in-family role the next
number, stable for the case; out-of-family roles get no number. After the fourth search the
next turn must be the final answer. A further search request, more than one tool request in a
turn, a refusal, a truncated or unparseable answer, or a number not in the case's map
escalates to a person.

- Live: one `client.messages.create` per turn, two tools with `strict: true`, `tool_choice`
  `{"type": "auto", "disable_parallel_tool_use": true}`, `output_config.format`
  (`json_schema`) for the final answer, and `tool_choice` `{"type": "none"}` once four
  searches are used; at most five requests per case. The schema uses a plain integer for the
  number (minimum and maximum are not supported in these schemas) and shared code checks the
  range.
- Offline: the stand-in issues its searches through the same capped tools, then answers: one
  catalog search, one approved-title search and, under the condition of Q19, one more catalog
  search.
- Recording: one entry per model turn; replay looks each turn up by key and re-runs the
  searches for real.

The cap, not-shown, out-of-family and zero-search tests run against the shared loop with a
scripted backend.

**Basis.** 1.6 step 6; 1.7: "If it fails any, or the agent escalates, the title goes to a
person."; 2.4: "The agent is a plain loop: at most four search calls, then one structured
answer. No agent framework."; 2.5: the three calls behind a switch, shared code, and "record
each answer in a file keyed by a hash of the step, model ID, prompt and request"; 2.7: "The
agent cannot pick a role it was not shown or one outside the family." The only reading under
which cap and numbering hold for all three backends is one shared loop with a per-turn backend
function, and replay must re-run the searches or the number map does not exist. On the hosted
API, tools and `output_config.format` can be combined, and a forced answer tool would be
rejected by the newest models.

### Q21. The role and status of the audited titles in a sampled path

**Open point.** For a path of 60 titles or fewer the design is explicit: "every title in it is
labeled: each takes its label with status Approved" (2.4). For a larger, sampled path it says
nothing about the 60 drawn titles: whether they are released with the SME's label or the
machine's role, with which status, and which role is released when the two differ. "Confirmed"
means "No person read it" (1.9), which is untrue of an audited title. It is equally unstated
whether audited titles in a stopped path are released.

**Decision.** Each of the 60 audited titles takes the SME's label as its role with status
Approved, whether or not the label matches the machine's role, and joins the look-up table and
the examples; the path outcome (Confirmed, Provisional or Stopped) applies only to the
unaudited remainder. Wrong is still counted against the machine's original role. In a stopped
path the 60 are released as Approved and the rest stay on legacy titles unless the release
setting is on. Table: the path row counts the titles released through the path, audited and
unaudited together, shows audited and wrong, and its status cell shows the path outcome; the
unaudited titles of a stopped path that the setting did not release are counted in the legacy
row instead (Q16). The footer's coverage by status counts the 60 under Approved. A golden-set
title must not reach the look-up table or the examples through the audit: golden titles stay
in the audit sampling frame (Q07), an audited golden title takes its label and status Approved
for release like any other, and the learning guard of Q07 refuses to record it.

**Basis.** 1.6 step 10: "Titles a person approved or labeled join the look-up table and the ML
model's examples." 1.2: "Look-up table | Titles a person has already decided, each mapped to a
role." 2.4: the small-path rule, and "A stopped path is left on legacy titles unless a setting
releases it"; wrong "means the machine's role differs from the simulated SME's label for that
title". 1.9: the meanings of Approved and Confirmed. A labeled title joins the look-up table
with the person's role, and a title a person read cannot be "Confirmed". Releasing the 60 in a
stopped path rests on analogy: the small-path rule releases labeled titles however wrong the
machine was. For golden titles, 1.2 and 2.7 forbid learning, not auditing, and 1.9 states no
exclusion from the sample; one guard in the learning function covers every way a person's
decision can arise.

### Q22. Testing "stand-in models never read the truth file"

**Open point.** "Stand-in models never read the truth file" and "People are simulated from the
truth file" (2.4) are both settled, but 2.7 lists no test for the first and the design does
not say what "never read" must mean in a test. Truth reaches the models indirectly and
legitimately: simulated approvals join the look-up table and the examples, the seed look-up is
a person's decisions, and the agent searches approved titles.

**Decision.** Both a structural and a behavioural test.

- Structural: the truth file is opened by one loader in the simulated-people module, and the
  pipeline hands the model calls only their arguments. A test parses the package source and
  asserts that no module other than the people module (and the data generator) refers to the
  truth loader or the truth file name, and that the model modules do not import the people
  module.
- Behavioural: (a) the stand-in `embed`, `choose` and `review` return identical output for
  identical arguments, including with the truth file absent; (b) with `true_role_id` permuted
  and every other file unchanged, the Wave 1 machine proposals and the Wave 1 golden-set
  scoring are identical.

For (b) to hold, all Wave 1 machine scoring is computed as one batch before any Wave 1
approval is applied, and the golden-set draw depends only on the seed and the title keys.
Nothing is claimed or tested after the first human decision.

**Basis.** 2.4: "Stand-in models never read the truth file." and "People are simulated from
the truth file". 1.6 step 10 (learning). 2.7 is headed "Tests worth having" and lists no test
for this rule. The invariant is settled; how to test it was not. The behavioural test is only
valid if Wave 1 scoring happens in a batch before approvals feed the examples, so the build is
ordered that way. The seed look-up is a separate file and stays fixed in the permutation.

### Q23. What the default run must show

**Open point.** 2.2 says the mock contains "the random audit and its three outcomes" and 2.6
says one command "prints one table" of seven rows, but nothing says which rows must be
non-empty in the default run, which outcome each path should get, whether the default data
must put each path above 60, or how hard the made-up titles and how informative the made-up
descriptions should be. A run has two machine paths, so it can show at most two of confirmed,
provisional and stopped, and a path of 60 titles or fewer gets no status at all. The only
expectation given is "provisional is the usual result in the mock" (2.4). A throwaway probe
(48 roles, 400 cleaned titles, 40 to Wave 1, a word-overlap chooser, two hashing embedders,
several seeds) showed how much depends on the data: titles reaching the agent ranged from 22
to 147 of 360 long-tail titles, depending mainly on how informative the generated descriptions
were; the models-agree path was 94% to 100% right, straddling the confirmed/provisional line
of 2 wrong in 60; the compliance rule alone sent 30 to 122 of 360 titles to people, depending
on which three roles were flagged. By arithmetic, of about 360 long-tail rows about 280 are
scored; the agent-settled path passes 60 only if the stand-ins disagree, mismatch family or
return "no match" on about 29% or more with a quarter escalating, and at 80 to 90% agreement
it holds 20 to 45 titles and is labeled in full.

**Decision.** A minimum bar. On the default seed: all seven rows are non-empty (the last two
through the leftover mechanism of Q01), the models-agree path holds more than 60 titles so one
sampled audit really runs, and that path is not stopped. The agent-settled path is not
required to exceed 60; if it holds 60 or fewer it is fully labeled as 2.4 provides, and the
README says so. The bar is reached by shaping the generator only (its difficulty knobs are
generator settings), never by tuning stand-ins or thresholds against the truth and never by
aiming at a status. At most three tuning rounds, then ship what the seed yields and state the
shape in the README and the final report. The bar can first be measured when the pipeline
exists, so the rounds are made in stage 4 by the branch that builds it ("Work that crosses a
branch boundary"). The three outcomes, the 60-or-fewer case and the stopped-path release
setting are covered by unit tests of the audit function on constructed counts (0 to 2, 3 to
12, 13 or more wrong of 60). The integration test asserts invariants (rows partition all
titles and employees, shares sum to 100%, the status follows from the audited and wrong
counts, the bar above) and that two runs in separate processes give byte-identical output.
The printed table is frozen as a committed expected-output file only once the shape is final,
that is after the last tuning round and not before (the file, its test and its owner: "Work
that crosses a branch boundary"). The three compliance-flagged roles sit in three different
families.

**Basis.** 2.2: "the random audit and its three outcomes". 2.4: the stand-ins and the README
statement; "The seed look-up exists so that the look-up step has hits to show"; the small-path
rule; "At 60 per path the audit has little power: a path that is truly 95% right is confirmed
less than half the time, so provisional is the usual result in the mock. The README says so."
2.6: one command prints one table of seven rows. The design anticipated small paths and gave
"provisional is the usual result" as a consequence of audit power, not a target; three
outcomes from two paths can only live in tests. The seed look-up shows the intent that data is
shaped so each step has something to show. A stronger bar would push the build to manufacture
stand-in disagreement, and no bar risks empty rows.

### Q24. The size of the Wave 1 list and its tie-break

**Open point.** Wave 1 is "the most widely held titles (the Wave 1 list)" (1.8). The only
number is in the data description: "the 40 most widely held titles cover roughly 40% of
employees" (2.1), which describes the skew of the data, not the list. The design never says
the Wave 1 list is those 40, nor how ties at the cut are broken. On any smooth curve that hits
40%, the titles around rank 40 hold about 6 employees each and many tie.

**Decision.** The Wave 1 list is the top `wave1_size` decision rows by headcount, default 40,
ordered by headcount descending, then family code, then cleaned title (the same canonical
order used for processing). In addition the generator makes the 40th and 41st rows differ in
headcount, and a data test asserts that and that the top 40 cover 35% to 45% of employees, so
membership never rests on the tie-break. "Widely held" in 1.8 and "Wave 1 title" in 2.7 mean
this same list.

**Basis.** 1.8: "The most widely held titles (the Wave 1 list), each approved by an HR SME."
2.1: "Headcount is skewed: the 40 most widely held titles cover roughly 40% of employees." The
same phrase names the Wave 1 list and the 40, and the design gives no other size. 1.4: "one
row per cleaned title within a family, with its headcount" is what is ranked. Only the
tie-break was unstated; any fixed rule works, and this one matches the processing order.

### Q25. The generic-role cap

**Open point.** The generic role is "capped at 2% of employees. A widely held or
compliance-flagged title never does" (1.8). Not stated: what an undecided title gets once the
cap is reached or when it is barred; the order in which undecided titles take the generic role
when not all fit; whether a title whose headcount would cross the cap is taken whole, split or
skipped; whether the cap is 2% of all employees or per family, and how 2% is rounded; and what
a "compliance-flagged title" is, since only roles carry a flag and an undecided title has no
role.

**Decision.** Cap = floor(`generic_cap_share` x employees in the employees file), default
0.02, one global cap (30 at 1,500). Whole titles only. Eligible leftovers are undecided queue
titles that have a usable family, are not on the Wave 1 list, and for which the compliance
check of 1.7 never fired on any proposal or on the ML top three. Titles no model scored (those
containing a person's name) and titles with no usable family are not eligible. Fill in
ascending headcount, ties by family code then cleaned title, assigning while the running total
plus the title's headcount stays at or under the cap (in ascending order the first title that
does not fit ends the fill). Every leftover not assigned, whether barred or over the cap,
stays on its legacy title and is counted in that row.

**Basis.** 1.8 (quoted above) and 2.7: "The generic family role is never assigned past its
cap, or to a Wave 1 or compliance-flagged title." 1.4: the unit of decision is a title row, so
a title is never split. 1.8 and 1.7: "Any title that touches a compliance-flagged role"; the
check is "The proposed role, or any of the ML model's top three, is compliance-flagged". 1.9
and 2.6: the only unreleased outcome named is to keep employees on legacy titles. "Never
assigned past its cap" forces rounding down and no partial titles. Only the fill order was a
free choice; ascending headcount fits the intent that widely held titles do not get the
generic role, and makes "skip" and "stop" the same thing.

### Q26. A missing or unknown family code

**Open point.** 2.4 says "No usable family code: a person", but the list of deliberate mess in
2.3 has no missing family codes, so it is unclear whether the data should contain any. It is
unstated whether such a row goes to no model or still gets a machine proposal, how it is keyed
in the look-up and the truth file, and what happens if it is a leftover, since it has no
"family's generic role".

**Decision.** The default data includes 3 title rows on 5 employees with a blank or unknown
family code. "Unusable" means blank after trimming, or not one of the catalog's family codes.
Such a row goes straight to the review queue with reason "no usable family code"; `embed`,
`choose` and `review` are not called for it, and it is out of the golden-set frame. It is
keyed on the cleaned title plus the family value as received (blank stays blank); the truth
file uses the same key and may name a role in any family. A person's decision makes it
Approved and joins the look-up under that key and the chosen role's examples. If it is still
undecided at the close it stays on its legacy title, because with no family there is no
generic role to take. One unit test for each of these.

**Basis.** 2.4: "No usable family code: a person." 2.3: the list of deliberate mess names
abbreviations, case, suffixes, level words, names, empty descriptions and missing grades, but
no missing family codes. 1.8: "a title nobody could decide takes its family's generic role".
1.6 step 2 and 1.4 (the key). 2.3: the truth file holds "one true role per title within a
family". The routing rule is in the design; a handful of rows makes it live at no cost. "A
person" with no model call is the plainest reading.

### Q27. Simulated people

**Open point.** People are simulated "with a small seeded error rate, and every simulated
decision is marked as simulated" (2.4). Not stated: the rate; whether it applies to every
human act; which role a mistaken person picks; whether a title labeled on more than one
occasion always gets the same simulated answer; who the "named person" of status Approved is;
and where the mark is stored and shown. Wrong is "the machine's role differs from the
simulated SME's label", so SME errors count against the machine.

**Decision.** Setting `sme_error_rate`, default 0.02, applied to every simulated human
decision. The label for a title is a pure function of (seed, family code, cleaned title): a
keyed SHA-256 draw decides whether it is wrong, and if so picks a different non-generic role
in the true role's family; the same label is returned on every occasion in the run (golden
set, audit, Wave 1, review queue). Personas are role labels, not invented human names: "SME 1
(simulated)" to "SME 3 (simulated)", "Reviewer (simulated)", "Catalog owner (simulated)", "HR
lead (simulated)", "CIO (simulated)". Every decision a simulated person makes carries
simulated = true and the persona in the state; the load file has a `decided_by` column
carrying the persona on the rows a person decided during the run (a look-up hit carries the
fixed label of S11 instead); the review-queue file and the table footer say that people were
simulated.

**Basis.** 2.4: "People are simulated from the truth file, with a small seeded error rate, and
every simulated decision is marked as simulated." and "the machine's role differs from the
simulated SME's label for that title" (one label per title). 1.2: only code assigns a generic
role, so a person never picks one. 1.9: "Approved | A named person decided it". 2.3 and 2.8:
names are invented, and no real person is named anywhere. At 2% a perfect path is confirmed
88% of the time and a 95%-right path 22%, which fits "provisional is the usual result". Role
labels as personas remove any chance of naming a real person.

### Q28. "No real company is named anywhere" and the vendors the design itself names

**Open point.** 2.8: "No real company, client or person is named anywhere, including commit
messages." Section 1.11 names Microsoft and Azure products, and 2.5 requires a setting value
`anthropic`, the variable `ANTHROPIC_API_KEY` and a model name. The rule does not say whether
it covers technology vendors or only real companies, clients and people.
Commits made with the coding tool end with a co-author line and a session link.

**Decision.** The rule covers real companies, clients and people, in data,
code, docs, branch names and commit text. Technology vendors and products may be named where
the design names them or the code needs them. The repository's git identity stays as
configured. Commits keep the coding tool's co-author and session trailer lines. No hygiene
test searches for vendor names; the tests check instead that mock emails use `example.com` and
that no key-shaped strings or `.env` files are committed.

**Basis.** 2.8 (quoted above). Opening paragraph: "mock data, a mock workflow, no real
employer and no real people". 2.3: "All names, emails and companies are invented; use the
reserved `example.com` domain." 1.11 names Microsoft and Azure products, and 2.5 fixes the
setting value `anthropic`, the example model and the variable name. A design that names these
vendors cannot mean its own rule to forbid them. Owner decision 7.

### Q29. Where recordings live and what a recording is valid for

**Open point.** "Record each answer in a file" can mean one file per answer or one file
holding all answers. No path or setting is given, and `.gitignore` ignores `out/`, so a
recording written there can never be committed. Nothing says what is stored beside the answer,
what live mode does when a recording already exists, or which combination of settings a
recording belongs to: the prompt contains the ten-role shortlist, which comes from the
embedder, so a recording made with `embedder = offline` cannot serve `embedder = local`,
another seed, other data, another model or reworded prompts.

**Decision.** A `recording_dir` setting, default `recordings/` (tracked, outside `out/`).
Layout `recordings/<llm_model>/<embedder>/` with `choose.jsonl` and `review.jsonl` (one line
per model request, sorted by key when the run ends) and `manifest.json` (model, embedder,
seed, data fingerprint, prompt version, UTC start and end, request count, complete flag)
written last. Each entry stores key, step, a title-row hint, the configured model, the model
ID the API reported (as information only), UTC date, the parsed answer or tool request, stop
reason and token counts; never headers or the key. Live mode always calls the API and never
reads a recording; it writes to a fresh sibling directory and replaces the previous recording
only when the run completes. Replay refuses a missing or incomplete recording before doing any
work. Any miss stops the run at once with a non-zero exit and no output files, and the message
names the step and title and prints the differences between the manifest and the current
settings. The manifest explains a miss; it is not a second validity rule. `embedder = local`
with `llm = replay` works only if that combination was recorded; only the offline-embedder
combination is ever committed. A test scans recordings for `sk-ant-`.

**Basis.** 2.5: "In live mode, record each answer in a file keyed by a hash of the step, model
ID, prompt and request. In replay mode a miss is an error; never fall back silently.
Recordings are never written by hand." and the three answer labels. 2.8: "No secrets. `.env`
is ignored. No keys in recordings.", which treats recordings as repository content, so they
cannot live under the ignored `out/`. Which settings a recording is valid for is already
decided by the key: another embedder, seed, data set or prompt wording gives a different hash,
and a miss is an error. Owner decision 8 for the two model strings.

### Q30. The library and model behind `embedder = local`

**Open point.** "`local` (a small local embedding model)" names no library, model or revision,
and does not say how the weights arrive (every realistic choice downloads them on first use,
so the first `local` run is not offline), whether the build must actually run it, or whether
it is tested.

**Decision.** model2vec with `minishlab/potion-base-8M` at revision
`bf8b056651a2c21b8d2565580b8569da283cab23`, declared as the optional extra `local` and
imported only inside the `local` embedder. Weights download on the first `embedder = local`
run into the user's Hugging Face cache, never into the repository; a missing extra or a failed
download stops with a clear message (for a missing extra, the command of S09) and never falls
back to `offline`. Unit tests cover the
switch and the lazy import with a stub. One smoke test, marked `local_model` and excluded from
the merge gate, may install the extra and embed a few titles once during the build, time-boxed
to 15 minutes; if it fails or times out, that is noted in the README and the build moves on.
sentence-transformers and PyTorch are not installed anywhere.

**Basis.** 2.5: "`local` (a small local embedding model)" and "Import the SDK and the
embedding library lazily, so the default run installs nothing." 1.5: "The embedding model is
off the shelf and is not trained." 2.8: "Nothing from outside this folder goes into the
repository." No library or model is named. model2vec is the lightest candidate (see Library
facts), needs no account or credential, and avoids a PyTorch install. Owner decisions 4 and 6.

### Q31. `llm_model` has no default

**Open point.** In the settings table of 2.5 the default cell of `llm_model` is empty. Open:
with `llm = anthropic` and no model set, does the run fail or fall back to the example? With
`llm = replay`, the model ID is part of the key, so a replay with the model unset would hash a
blank and miss every answer unless replay takes the model from the recording. Also unstated:
whether any hosted model must work or only the example.

**Decision.** No default. `llm = offline` ignores `llm_model`. `llm = anthropic` with
`llm_model` unset is a start-up error that names `claude-haiku-4-5` as the example. `llm =
replay` reads the model from the recording's manifest; if `llm_model` is also set and differs,
that is a start-up error. The layout of Q29 has the model in the path, so the recording is
found first: with `llm_model` set, replay opens `<recording_dir>/<llm_model>/<embedder>/`;
with it unset, it uses the only folder `<recording_dir>/<model>/<embedder>` that exists for
the embedder in force, and stops at start-up, listing the models found and naming
`--llm-model`, if there is none or more than one. Docs and tests use the alias
`claude-haiku-4-5` exactly as written, with no date suffix. Any model string is passed through
unchanged; to stay valid across current models the request sends no `temperature`, `top_p` or
`top_k`, no thinking parameter, no effort parameter and no forced tool choice (tool choice
auto with a strict tool, or structured output).

**Basis.** 2.5: "`llm_model` | a small hosted model, for example `claude-haiku-4-5` |" with an
empty default, while `embedder` and `llm` both show `offline`; the key is read "only when `llm
= anthropic`"; recordings are "keyed by a hash of the step, model ID, prompt and request. In
replay mode a miss is an error; never fall back silently." An empty cell beside two filled
ones reads as "no default", and an explicit start-up error fits "never fall back silently"; it
also means no paid model is used without the owner naming it. On the hosted API, sampling
parameters and forced tool choice are rejected by the newest models and the effort parameter
errors on the example model, so all are left out.

### Q32. How replay, live mode and the local embedder are tested

**Open point.** The test list in 2.7 has nothing on the switch, recording, replay misses,
labels or lazy imports. "Recordings are never written by hand" leaves open whether a test may
create a recording by running the real record path against a scripted fake model. It is also
unstated whether the suite must pass where the SDK is not installed.

**Decision.** Four tiers.

- (a) The core suite needs only the standard library and pytest; this is the merge gate,
  whose second run has no optional extra installed (S14). It includes a subprocess test that
  runs the default mode and replay mode with an import blocker on `anthropic`, `httpx2`,
  `model2vec` and `numpy` and expects success. The blocker is the import guard of S15, which
  refuses those four and everything else from outside the standard library.
- (b) Record-then-replay round trip: live mode with an injected scripted client under the
  model ID `test-double` writes through the real recorder into a temporary directory; replay
  gives identical outputs; changing one input gives a miss error, non-zero exit and no output
  files; the three labels are asserted. No recording made by a test is ever committed.
- (c) Request-shape tests guarded by `pytest.importorskip("anthropic")`: a real SDK client
  with `http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))`,
  asserting model, tools, `tool_choice` and `output_config` and that no key reaches a recorded
  entry. They run in the gate, where uv installs the `live` extra, and skip wherever the SDK
  is absent.
- (d) Tests marked `live` and `local_model`, skipped unless a key or a cached model is
  present, and never part of the gate. The build may run `local_model` once at the end if the
  model download works; the result is reported but never gates a merge.

The committed recording gets its own replay test, marked `replay`: a full run with `llm =
replay` has zero misses.

**Basis.** 2.5: "Import the SDK and the embedding library lazily, so the default run installs
nothing."; "In replay mode a miss is an error; never fall back silently. Recordings are never
written by hand."; the three answer labels. 2.7 is headed "Tests worth having": a list of
eight rule tests, not a full test plan. 2.1: "the standard library by default". A recording
produced by the program's own recorder in a temporary directory under an obviously fake model
name is not hand-written and is never committed. The SDK's HTTP layer is `httpx2`, so
transport fakes must be `httpx2` objects or they silently see no traffic.

### Q33. The recording key

**Open point.** "Record each answer in a file keyed by a hash of the step, model ID, prompt
and request." Open: what "prompt" and "request" each mean, and whether "request" is the
program's own structure or the body the SDK serialises; the hash function and the
serialisation it runs over; whether "model ID" is the configured string or the ID the API
reports back; the step names and, for the agent, whether a step is one model turn or the whole
case; and what must be left out.

**Decision.** Key = SHA-256 hex of UTF-8 JSON (sorted keys, compact separators, non-ASCII
kept, strings NFC-normalised) of {key_version, step (`choose` or `review`), model as
configured in `llm_model`, system prompt text, tool definitions, output schema, `max_tokens`
and any other request parameter the program sets, normalised turns: role and text, tool name
and input, tool result text}. Tool-use IDs, dates, request IDs and the SDK version are left
out. One key per model turn, so an agent case has up to five. The structure is built by the
program's own code before the SDK is imported, and the output schema is a hand-written dict,
not derived from pydantic. Each recording stores the key, key version, step, configured model,
the model ID the API reported, the date and the answer, and no headers or credentials.

**Basis.** 2.5 (quoted above) and "recorded <date>, <model>": date and model are stored with
the answer, so they are data, not key. 2.8: "No keys in recordings." The design names what
goes into the key; hashing the program's own canonical structure rather than the SDK's
serialised body keeps keys stable across SDK upgrades. `claude-haiku-4-5` is an alias of
`claude-haiku-4-5-20251001`, so the configured string must be the keyed one, because the
reported ID is not known before the call. Owner decision 8.

### Q34. How branches reach main

**Open point.** The design has no delivery section. It says nothing about pushing, pull
requests or merge style.

**Decision.** A local gate; nothing is pushed and the hosting service's command-line tool is
never invoked. One feature branch per component; rebase on main; run the single check command
(lint, unit and integration tests) on the branch and again on the merge result in a separate
integration worktree; merge with `--no-ff`; merges are serialized; never force-push and never
rewrite main. A CI workflow that runs the same command is committed; it takes effect only once
the owner pushes.

**Basis.** The design is silent. The build was asked for feature branches with unit and
integration tests when merging components to main, which local merges behind tests meet;
pushing publishes the repository and is the owner's step. Owner decision 1.

### Q35. Counts and reference data in the mock data set

**Open point.** Several quantities are words, not numbers: "a few titles" in the seed look-up,
"a few with a person's name", "some empty descriptions", "a few missing grades" (2.3), "a
small abbreviation list" (1.6). The Wave 1 list size is only implied (Q24), missing family
codes are not in the list of mess (Q26), the cleaner needs location and requisition references
(Q14), the grade-to-level table is not given (Q36), and file type and folder are not stated
(Q38).

**Decision.** The values are fixed once in the generator's constants: UTF-8 CSV with headers
in `data/`, produced by the seeded generator and committed.

- 1,500 employees; 400 title rows counted after cleaning and per family (the unit every count
  in 2.6 uses; raw spellings are more numerous); 8 families; 48 roles plus 8 generic; 3
  flagged roles in three different families.
- Wave 1 list: the 40 title rows with the highest headcount, ties by family code then cleaned
  title (Q24).
- Seed look-up: 8 long-tail title rows, correct against the truth, none in the golden set.
- 5 title rows with a person's name.
- About 15% of title rows with no description on any employee, plus a few rows where only
  some employees have one.
- About 20 employees with no grade.
- 5 employees on 3 title rows with a blank or unknown family code.
- About 30 abbreviations, about 10 locations, 2 requisition patterns; the locations and
  patterns are the module constants of Q14, shared by generator and cleaner.
- About 4 titles each built to draw "no match" and "insufficient data" from the stand-in.
- Grades 1-3 Junior, 4-6 Associate, 7-8 Senior, 9 Lead, 10 Principal.

Size tests allow plus or minus 10%, and 35% to 45% for the top-40 share.

**Basis.** 2.1: "Suggested size: about 1,500 employees on about 400 titles, 8 job families,
about 48 roles plus one generic role per family, 5 levels, grades 1 to 10, 3
compliance-flagged roles. Headcount is skewed: the 40 most widely held titles cover roughly
40% of employees." 2.3: "a few titles a person has already decided", "a few with a person's
name, some empty descriptions, a few missing grades". 1.6: "a small abbreviation list". 1.3:
grades 7 and 8 are both Senior. The design calls its sizes "suggested" and uses "a few" and
"some" on purpose; any consistent values work. They are constants in the generator and only
need to be shared by the generator, the cleaner and the tests.

### Q36. The grade-to-level table and the worked example

**Open point.** 1.3: "Grades 7 and 8 both read as Senior in HR's table". 2.1 gives "5 levels,
grades 1 to 10" and leaves the table to the build. The obvious split of two grades per level
puts grades 7 and 8 at Lead, the fourth level, while Senior is the third.

**Decision.** Grade-to-level file: grades 1-3 Junior, 4-6 Associate, 7-8 Senior, 9 Lead, 10
Principal. Level ladder short words: Jr, Assoc, Sr, Lead, Prin. The mock catalog includes
Sales x Account Manager with short name "Acct Mgr", and a unit test reproduces the 1.3
example: grades 7 and 8 both give "Senior Account Manager" and "Sr Acct Mgr".

**Basis.** 1.3: "'ACCT MGR II' (Sales, grade 8) and 'Senior Account Manager' (Sales, grade
7)"; "Grades 7 and 8 both read as Senior in HR's table"; "Long Title | Senior Account Manager"
and "Short Title | Sr Acct Mgr". 1.2: "Level | Junior, Associate, Senior, Lead, Principal."
2.1: "5 levels, grades 1 to 10". The example rules out the even split; the other eight grades
have one natural monotonic completion.

### Q37. What set-up does in the mock

**Open point.** 2.6: one command "runs set-up, Wave 1, Wave 2 and the close". "In set-up,
three HR SMEs label the same 30 titles, which gives an estimate of how often people agree with
each other" (1.9). 2.2 lists this exercise neither in nor out of the mock. "The set-up pilot"
is out, but 1.10 describes that as "a small pilot timed", a different thing. The footer has no
agreement figure.

**Decision.** The three-SME exercise is left out, and the README says so in one line. Set-up
in the mock is the Set-up row of 1.10 without the pilot: load and validate the catalog and
data files, fix and record the thresholds and settings in force before any scoring, and draw
the golden set (seeded, from long-tail titles).

**Basis.** 2.2: the "In the mock" list names no agreement exercise, and "Not in the mock: a
review application (people are simulated), a real HR system, cloud deployment, sign-in, the
set-up pilot, the dry run ...". 1.10: "Set-up | Catalog approved; release thresholds signed; a
small pilot timed; golden set drawn". 1.2: thresholds are signed "before any result exists".
The "in the mock" list is an enumeration and the exercise is not on it, and the output has no
place for its result.

### Q38. Settings mechanism, optional libraries, Python version, names and formats

**Open point.** The settings table gives names, values and defaults but not the mechanism.
"Import the SDK and the embedding library lazily, so the default run installs nothing" does
not say how the two libraries are declared or what happens when one is missing. The design
fixes no language version, package name, command name, file format or location for the files
of 2.3 and the outputs, or state store. It does not say whether the mock data is committed or
generated at run time, or whether "under a minute" is a tested limit.

**Decision.** Package `titlestd` as a directory at the repository root (flat layout), so
`python3 -m titlestd run` works from a fresh clone with nothing installed; further commands
`generate-data` and `record`. `pyproject.toml` with no runtime dependencies and two optional
extras, `live` = `anthropic>=1.11,<2` and `local` = `model2vec>=0.9,<0.10`; pytest and ruff
are dev-only, fetched by uv. A missing extra gives an error containing the exact install
command, whose text S09 fixes. `requires-python` is `>=3.12` with no upper bound; the merge
gate runs on 3.12 (`.python-version`), and the suite also runs on 3.13 and 3.14, with
`make check-all` (S14). Exact-output comparison only on 3.12 (the frozen table of Q23).
Settings are one immutable dataclass with defaults, overridden by an optional TOML file given
with `--config` and then by command-line flags generated from its fields (`--embedder`,
`--llm`, `--llm-model`, `--seed`, `--recording-dir` and the rest). There are no
environment-variable settings, so the environment carries only `ANTHROPIC_API_KEY`. Inputs are
committed CSV in `data/`. Outputs and state go to `out/`: `load.csv`, `reverse.csv`,
`review_queue.csv`, `exceptions.csv` and `state.sqlite`, recreated on every run. The
end-to-end test fails if the default run takes more than 60 seconds.

**Basis.** 2.1: "Show the whole idea in under a minute ... one package, the standard library
by default, a local SQLite file or plain files for state." 2.5: the settings table; the key is
"read by the SDK from the environment and only when `llm = anthropic`"; lazy imports. 2.6:
"One command runs set-up, Wave 1, Wave 2 and the close on the mock data and prints one table."
The repository's `.gitignore` already ignores `out/` and `*.sqlite`. These are packaging
conventions the design leaves to the build. A default run that "installs nothing" has to work
on whatever `python3` a reader has, which argues for no upper bound and for the flat layout.
Owner decisions 6 and 9.

### Q39. Live-mode failures and where the credential comes from

**Open point.** "The only secret is `ANTHROPIC_API_KEY`, read by the SDK from the environment
and only when `llm = anthropic`." Open: what live mode does when the key is missing, when the
API still fails after the SDK's own retries, when the model refuses or is cut off, or when the
answer fails validation; whether a partly recorded run is kept; and how a key kept in a file
reaches the environment, since the program does not read one.

**Decision.** When `llm = anthropic`, first check that `ANTHROPIC_API_KEY` is set and
non-empty and stop with a clear message if it is not; only then import the SDK and build
`anthropic.Anthropic()` with no arguments. A rejected key stops the run on the first call and
nothing is written (a free `client.models.retrieve(llm_model)` call at start can check the key
and the model ID before any work). An API or network error after the SDK's default two retries
stops the run with a non-zero exit, keeps the answers already recorded and writes no load
file. A refusal, a cut-off at `max_tokens` or an answer that fails validation sends that title
to a person with the reason, and is recorded so replay repeats it. The stand-in is never
substituted. The program never reads the key from a file. The README tells the reader to
export the variable in the shell, or to keep it in a file outside version control and pass
that file to `uv run --env-file`. During the build the variable is set from a file outside the
repository for the recording step only (Q02); every other test of the live adapter runs
against a fake client.

**Basis.** 2.5: "The only secret is `ANTHROPIC_API_KEY`, read by the SDK from the environment
and only when `llm = anthropic`. It is never written to a file in the repository." and "never
fall back silently". 1.7: a failed check sends the title to a person. The explicit check is
needed because the SDK's no-argument client also accepts another token variable or a stored
login profile, so a missing key would not fail by itself and a run could bill an account the
owner did not intend, against "the only secret is `ANTHROPIC_API_KEY`". Owner decision 2.

### Q40. The answer label: where it is kept and what its date and model are

**Open point.** "Label every answer as 'stand-in', 'live' or 'recorded <date>, <model>'."
Open: where the per-answer label is stored and shown, the date's format and time zone, whether
the model is the configured string or the one the API reports, whether the embedder is
mentioned, and what the footer line says when a recording spans several dates.

**Decision.** Every stored `choose` and `review` answer carries a `source` field holding
exactly one of those three strings. It is shown in the review-queue file and summarized in the
footer line; the load file gets no extra column, since 1.6 step 9 defines its content. Date =
the UTC day of the API call in ISO form (YYYY-MM-DD), written into the recording when it is
made and read back on replay. Model = the configured `llm_model` string, the same one used in
the recording key. The footer lists the distinct labels, shows a date range when a recording
spans several days, and names the embedder setting.

**Basis.** 2.5 (quoted above). 2.6: the footer has "one line saying whether answers came from
stand-ins, a recording or a live model". 1.6 step 9: the load file is "one row per employee:
the new titles and their status". Using the configured model string keeps the label consistent
with the recording key and does not depend on what the API reports back. Owner decision 8.

---

## Contracts settled when the build was cut

The entries above leave some names, formats and interfaces to the code. They were fixed when
the shared code was written, so that parts built at the same time agree. Each is a decision
taken in the spirit of the design; where one resolves a conflict between two entries above, it
says so.

### S01. The data files

The seven files of 2.3 are `catalog_roles.csv`, `level_ladder.csv`, `grade_levels.csv`,
`abbreviations.csv`, `employees.csv`, `seed_lookup.csv` and `truth.csv`, with exactly the
column names the design lists, in that order. UTF-8 CSV with a header row, fields quoted only
where needed, lines ending in a line feed. A flag is `1` or `0`; a missing grade is an empty
field; a grade is otherwise a whole number. Nothing is trimmed on reading, so a job title
comes back byte for byte. A reader refuses a header that differs in any way, a row of the
wrong width, a repeated key and an empty required field, naming the file and the line.
Checks across files: every family has exactly one generic role; every grade names a level of
the ladder; a seed or true role exists, is not generic and, where the family code is one of
the catalog's, is in that family (Q09). Six files are read and written by
`titlestd/datafiles.py`; the truth file is read and written only by `titlestd/people.py`
(Q22, S15).

### S02. Numbers the design fixes are constants, not settings

The design states these as numbers and no entry above makes them settings, so they are named
constants in `titlestd/contracts.py`: a shortlist of 10 roles; the ML model's top 3; the first
200 words of a description; at most 4 agent searches; an audit sample of 60; the accuracy bar
0.90; the pass chance 0.10 and the stop share 0.20 behind the audit lines of 2.4; a quote of
at least 3 characters and at most 25 words (Q15); the masks `[NAME]` and `[EMAIL]` (Q14). The
audit functions take the sample size as an argument, so the lines for 200 (14 and 40) are
tested beside those for 60 (2 and 12). A catalog with fewer than ten model-visible roles gives
a shorter shortlist; the mock data never does.

### S03. The settings

One immutable object, `titlestd.settings.Settings`, checked on every construction. Sources in
rising precedence: the default, an optional TOML file given with `--config` (a flat table of
setting names), a command-line flag named after the setting with dashes. An unknown name, a
wrong type or an impossible value is an error that names the setting. `llm = anthropic`
without `llm_model` is refused and the message names `claude-haiku-4-5` as the example (Q31).

| Setting | Default | Entry |
|---|---|---|
| `embedder` | `offline` | 2.5, Q12, Q30 |
| `llm` | `offline` | 2.5 |
| `llm_model` | none | 2.5, Q31 |
| `seed` | 20260101 | Q06 |
| `data_dir` | `data` | Q38 |
| `out_dir` | `out` | Q38 |
| `recording_dir` | `recordings` | Q29 |
| `request_cap` | 3000 | Q02 |
| `wave1_size` | 40 | Q24 |
| `golden_size` | 40 | Q08 |
| `ml_alone` | off | Q04 |
| `ml_margin_cutoff` | 0.10 | Q04 |
| `ml_min_examples` | 3 | Q04 |
| `embed_dim` | 32768 | Q12 |
| `agent_search_k` | 5 | Q10 |
| `standin_min_title_words` | 2 | Q17 |
| `standin_no_match_below` | 2 | Q17 |
| `sme_error_rate` | 0.02 | Q27 |
| `undecided_rate` | 0.10 | Q01 |
| `generic_cap_share` | 0.02 | Q25 |
| `release_stopped_paths` | off | 2.4, Q21 |

### S04. Seeding

`titlestd.seeding.rng(seed, stream, *key)` returns a fresh `random.Random` seeded from the
SHA-256 of the compact UTF-8 JSON array `[seed, stream, *key]`; `unit(seed, stream, *key)`
returns one number in [0, 1) taken straight from that digest, for a rate test. The stream
names are fixed: `shuffle` (the ten roles, per title), `golden` (the golden-set draw), `audit`
(the audit sample, keyed by the machine path's name), `sme_error` (whether and how a simulated
label is wrong, per title) and `undecided` (whether a simulated reviewer cannot decide, per
title). The item key of a per-title stream is the family code, then the cleaned title. Tests
pin the digest and the first draws, parse the package to assert that only this module imports
`random` and that nothing calls `hash()`, and run on every supported Python.

### S05. The decision key, the title row and its record

The key of a decision row is `TitleKey(family_code, cleaned_title)`. `family_code` is the
recorded code with surrounding whitespace removed; a blank or unknown code is kept as it is
(Q26). `cleaned_title` is the title after cleaning and masking (Q14). Sorting keys gives the
processing order of Q18. A `TitleRow` holds the record of Q11 (cleaned title, family code,
masked department, masked description of at most 200 words), the sorted employee ids, the
distinct raw titles as received, and two flags: the title held a person's name; the family
code is usable. Because a row keeps its raw titles, the simulated people look the truth up by
raw title and family code and need no cleaner (Q09 still holds: the files are keyed on raw
titles).

Family codes are compared after removing surrounding whitespace on both sides. The key holds
the trimmed code, while the employees, seed and truth files hold the code as recorded and the
readers trim nothing (S01). So whoever matches a file's row to a key trims the file's code
first: `prepare.title_key` for the seed look-up, the simulated people for the truth. Without
that a padded code, or one made of spaces only, would miss the truth and come back as "no
suitable role". The mini data set has one padded code (`"FIN "` on one employee and on its
truth row), so both stage 1 branches meet the case.

### S06. The order of cleaning

Within "clean" (Q14) the steps run in this order: lower-case; remove requisition numbers;
strip punctuation; remove trailing location tokens; expand abbreviations as whole tokens;
collapse spaces. Then mask names. Requisition numbers go first because their patterns contain
punctuation. "Sr. Acct Mgr - NYC" becomes "senior account manager", and a title with a name
becomes, for example, "customer success manager [NAME]".

### S07. The reference lists

`titlestd/reference.py` holds 12 given names, 12 surnames and 10 location tokens, all invented
for this mock, lower-case single tokens in sorted order, with no word in two lists; the two
requisition patterns of Q14, matched without regard to case; and the reserved email domain
`example.com`.

### S08. Components exchange plain data; the pipeline writes the state file

Every component is a set of functions over the plain types of `titlestd/contracts.py`, held in
memory. The look-up table and the examples are one in-memory object during a run. The pipeline
alone writes `out/state.sqlite` (Q06), through `titlestd/store.py`, whose schema is its own
business and not a contract for any other part. This keeps the parts independent of a shared
schema and lets each be tested without a database. A type that crosses a module boundary is
defined in `contracts.py`; that file imports nothing from the rest of the package.

### S09. The model switch

`titlestd.switch.build_models(settings, ladder, abbreviations, data_fingerprint)` returns the
embedder and the LLM for the settings in force. Each provider is one function in its own
module, imported only when its setting is chosen:

| Setting | Provider |
|---|---|
| `embedder = offline` | `titlestd.offline_embedder.build_embedder(settings)` |
| `embedder = local` | `titlestd.local_embedder.build_embedder(settings)` |
| `llm = offline` | `titlestd.standin.build_llm(settings, ladder, abbreviations)` |
| `llm = replay` | `titlestd.replay.build_llm(settings, data_fingerprint)` |
| `llm = anthropic` | `titlestd.live.build_llm(settings, data_fingerprint)` |

The three calls are interfaces in `contracts.py`: `Embedder.embed(texts)`,
`Chooser.choose(record, roles)` and `Reviewer.next_turn(transcript)` (Q20). An embedding is a
sparse mapping of index to weight, L2-normalised (Q12); a dense vector is the mapping of every
index. An LLM provider also returns `finish`, which the pipeline calls once after a run that
completed; live mode completes its recording there (Q29). A provider is given the level ladder
and the abbreviations, never the catalog and never the truth. The data fingerprint in a
recording's manifest (Q18, Q29) is a SHA-256 over the six input files, combined by the
pipeline with a SHA-256 of the truth file, because simulated decisions feed the examples and
so shape later prompts.

**A missing optional library.** The command that Q30 and Q38 ask for is one text,
`contracts.run_with_extra(extra)`:

    uv run --extra live python -m titlestd <command> ...
    uv run --extra local python -m titlestd <command> ...

It installs the extra into the project environment and runs the program there, in one step;
both extras are given as `--extra live --extra local`. A provider whose library cannot be
imported raises a `TitlestdError` whose message holds that text; the switch lets it through,
and the README gives the same command. Two other forms were ruled out. The project is not an
installable distribution (`package = false`, no build system), so there is no `pip install`
of an extra. And `uv sync --extra <name>` only installs: the `python3 -m titlestd` of a bare
run still cannot import the library, because it does not use `.venv`, and the exact sync
removes the other extra.

**Seams for tests.** Two functions take an optional argument through which a test hands in a
double. They are declared in the stubs and a scaffold test pins them, so the two stage 4
branches do not each invent a seam. The branches that fill the functions in keep these
parameters and may add further optional ones.

- `titlestd.pipeline.run_pipeline(settings, models=None)`. With `models` given the pipeline
  uses them and does not call the switch. It serves the tests with an injected misbehaving
  model (T2, T4), the spying providers (T5, Q14, Q26) and the round trip of a whole run
  (Q32 b).
- `titlestd.live.build_llm(settings, data_fingerprint, client=None)`. With `client` given the
  key is not checked and the SDK is not imported. A scripted object serves tier (b) of Q32
  where the SDK is not installed; a real SDK client on a mock transport serves tier (c).

`titlestd.pipeline.data_fingerprint(data_dir)` returns the fingerprint described above, so a
test that builds a provider itself gives it the value the pipeline would. The switch takes no
double: `build_models` always builds the real providers for the settings in force.

### S10. An answer that is not usable

`choose` and `review` answers have one more kind beside those of Q15: `invalid`, with a detail
saying why (refusal, cut-off, failed validation). It is how a live backend reports the cases
of Q39 and how the agent loop reports a broken protocol (Q20). The rules send it to a person
with the reason "invalid model answer". It is recorded like any answer, so replay repeats it.

### S11. Vocabularies and the decision record

Reasons, as shown in the review-queue file: "role not in catalog", "quote not in record",
"compliance-flagged role", "insufficient data", "models disagree", "role outside recorded
family", "no match", "agent escalated", "invalid model answer", "person's name", "no usable
family code", "ML alone", "no suitable role, waiting for catalog owner". A number that maps to
no shown role fails the catalog check (Q15).

Reasons on the exceptions list (Q05): "stopped path", "over the generic cap", "undecided
widely held or flagged title", "no suitable role", "no grade", and one that Q05 does not
name: "undecided title with a person's name or no usable family", for the two kinds of
leftover that Q25 and Q26 bar from a generic role without their being widely held or flagged.

There is exactly one final decision per title row. It carries the table row it is counted in
(Q16), who chose the role (look-up, person, machine, code, or nobody), the role, the status,
the persona and the simulated mark, the path's error rate where Q05 asks for it, and the
exception reason when the title stays on its legacy title.

A look-up hit has status Approved (Q13). Approved means "A named person decided it" (1.9), so
its `decided_by` is not left empty: it carries the fixed label "Seed look-up (mock data)"
(`contracts.LOOKUP_DECIDED_BY`) in place of a persona. Within one run only a seed entry can
be hit, because every title row is decided once, and the seed file names no person. It has no
simulated mark: it is input data, not a decision simulated during the run. This is the one
case in which an Approved row carries no persona, as Q05 and Q27 say.

The result of a run (`RunResult`) carries, beside the decisions, the two things the report
prints that the decisions do not hold: the Wave 1 list, for the "sent to a person" line of
Q16, and the number of employees with no grade who received a new title, the second count of
Q05. The report reads no data file.

### S12. The review-queue file

`review_queue.csv` has the columns `cleaned_title`, `job_family_code`, `headcount`, `reasons`
(joined by "; "), `ml_first_choice`, `llm_pick`, `agent_recommendation` (role ids or blank),
`answer_source` (the answer label, Q40), `decision` (the person's role id, or "could not
decide" or "no suitable role"), `decided_by`, `simulated` and `outcome` (the table row the
title ended in). Rows are sorted by key.

### S13. The command line

`python3 -m titlestd <command>` with the commands `run`, `generate-data` and `record`. Every
command takes `--config` and the setting flags. `record` is `run` with `llm` forced to
`anthropic`: live mode always records (2.5), so the command needs nothing of its own. Exit
status: 0 success; 1 the run stopped with an error, or the command is not built yet; 2 the
command line or a setting is wrong. Each command is one function in the module that owns it
(`titlestd.pipeline.run_command`, `titlestd.mockdata.generate_command`), so the command-line
module does not change as parts are built. The smoke run of Q02 is `record` on the mini data
set (`--data-dir tests/fixtures/mini`, with `--out-dir` and `--recording-dir` outside the
repository): 15 title rows, about ten of which reach the model.

### S14. The check command

`make check` is the merge gate: `ruff check`, `ruff format --check`, then pytest on everything
except tests marked `live` or `local_model`, twice.

- The first run (`make test`) is in the uv environment pinned by `.python-version` (3.12) and
  `uv.lock`, with the `live` extra installed, so that the request-shape tests of Q32 (c) run
  in the gate.
- The second run (`make test-bare`) is in a throwaway environment that holds the standard
  library, pytest and ruff and no optional extra. It is what proves tier (a) of Q32, and that
  the tests of tier (c) skip where the SDK is absent.

The ruff calls and the first run sync `.venv` exactly (`--exact`), so it holds the extras
named in `EXTRAS` and nothing an earlier command installed; plain `uv run` keeps whatever is
already there. The Makefile passes the extras it installed to the tests in the variable
`TITLESTD_GATE_EXTRAS`, which one tooling test compares with the optional libraries that can
actually be imported; nothing in the package reads it. So `make test` fails if the SDK is
missing and `make test-bare` fails if it is present.

`make check-all` runs `make check` and then the suite on Python 3.13 and 3.14, each in a
throwaway environment (Q38). During this build nothing else runs those two interpreters: the
CI workflow only takes effect once the owner pushes, and the `python3` a reader runs the
program with need not be 3.12. `UV_PYTHON=3.13 make check UV_FLAGS=--isolated` runs the whole
gate on another interpreter without touching `.venv`.

The gate leaves the working tree clean: everything it creates is ignored. Pytest collects only
from `tests/`, and ruff and pytest ignore `.worktrees/`, `.claude/`, `.venv/` and `out/`.
Markers: `integration` (set automatically on everything under `tests/integration`), `e2e`,
`replay`, `live`, `local_model`.

### S15. Rules the tests enforce on the whole package

- Only `titlestd/people.py` contains the truth file's name, reader and writer. Only
  `titlestd/pipeline.py` and `titlestd/mockdata.py` may import that module; no model module
  (`switch`, `offline_embedder`, `standin`, `local_embedder`, `live`, `replay`, `recording`)
  may (Q22).
- Only `titlestd/seeding.py` imports `random`; nothing calls `hash()` (Q06).
- Importing any module of the package imports nothing from outside the standard library and
  the package itself, so none of `anthropic`, `httpx2`, `model2vec`, `numpy` (2.5, Q32). It is
  checked in a fresh interpreter without site-packages whose import system records and
  refuses every such attempt: the record is what catches an optional import wrapped in
  `try`/`except ImportError` at module level, which swallows the refusal. It is checked again
  with site-packages present, where such an import would succeed. The guard is `IMPORT_GUARD`
  in `tests/conftest.py` (fixtures `import_guard` and `bare_python`); the blocked runs of Q32
  (a) use the same one.
- The package is flat: every module is a file directly under `titlestd/`.
- Every component module's docstring names its owner branch and the design sections and
  entries that govern it.

### S16. What a model is shown

The record block shows the record's cleaned title, family code, department and description,
each verbatim on a labelled line of its own; the stand-ins quote from the title or from a run
of consecutive description words (Q15, Q17). The numbered list of the ten roles shows, per
role, its number, name, family code and definition, the same fields Q10 gives a catalog search
result. No rendered text shows a role id or a similarity score. The renderers are shared code
used by every `llm` setting, so a prompt is the same string whoever answers it.

---

## Diagram rules not to copy

The two diagrams in `docs/` are older than the design. The design names three differences:
they list three audited paths where there are two, say the ML model switches on in Wave 2, and
give a "no suitable role" title a generic role. Beyond those three, the diagrams differ from
the design in the following places, and none of these is to be built. The human boxes H1 to H5
and the waves box are identical in both diagrams.

1. Set-up: "map 100 most common titles by hand". The design has no hand-mapping step; the
   mock's seed look-up is "a few titles".
2. Golden set of "300 rows", in Set-up and box 6. The design gives no size, and 300 is most of
   the mock's long tail.
3. Box 1 lists "historical data" as an input and box 4 shows it to the LLM. In the design
   title history is optional, for reviewers only, never embedded, and out of the mock.
4. Box 1 embeds title and description during preparation, before the look-up; the Microsoft
   box 3 repeats this ("vectors from Item 1"). In the design a look-up hit uses no model, and
   a title with a person's name goes to no model.
5. Box 1 cleaning omits lower-casing, punctuation and requisition numbers. Neither diagram has
   the rule that a name in a title sends it to a person.
6. Box 2 asks whether a human approved "a standard title in the catalogue for this role". The
   design matches on cleaned title plus family.
7. Box 3 compares the "title" with "every catalogue role". The design embeds title plus 200
   words of description and excludes generic family roles.
8. Edge label "matcher alone ... on during sustaining", and the Sustaining text that lets
   look-up hits or model agreement settle new titles. In the design a person approves every
   new title after handoff.
9. Box 4 says "not enough information"; the design says "insufficient data" and a number from
   1 to 10.
10. Box 5 gives the agent one search, "for additional catalogue entries". The design gives two
    searches, in-family picks only, generic roles excluded, at most four calls in the mock,
    and a re-check by the rules.
11. Box 6 and H3 draw "200 blind rows". The mock samples 60 titles, with cut-offs of 2 and 12,
    not 14 and 40.
12. H3 has a second HR SME adjudicate disagreements between the workflow and the first SME.
    The design has no adjudication: wrong means the machine differs from the one SME label.
13. H3 names the outcomes "accept, provisional, escalate"; the design uses Confirmed,
    Provisional, Stopped. Box 6 names two paths and H3 three, so the diagram disagrees with
    itself.
14. H2 says "90 rows in 100 correct at 90% confidence" and blind labeling in "Wave 2 and 3".
    The design never states a confidence level and audits paths in Wave 2 only.
15. H2 caps generic roles at an absolute "900 max" rows. The design uses 2% of
    employees and bars Wave 1 and compliance-flagged titles, which the diagram omits.
16. H4 sends "too few approved titles" to human review. In the design too few examples sends
    the title to the LLM.
17. H4 has SMEs "select family and role". The design has no family reassignment by reviewers.
18. H4 sends "no match in Item 5" to people. In the design an LLM "no match" goes to the agent
    first.
19. H1 approves the catalog in "two passes (Wave 1 and Wave 2 families)". The design approves
    it once in set-up, and waves split by headcount, not by family.
20. Waves box: Wave 1 is "~400 most-held titles", which is the whole mock. Wave 3 aims at
    "100% coverage", which the design's cap and legacy outcome do not promise.
21. The look-up "hit" arrow runs into box 5 (validation). The design does not say that look-up
    hits pass through the rules.
22. Systems: "PostgreSQL for embedding vector similarity in Items 2, 3, and 5" implies vector
    search in the look-up. The design's look-up is exact with no model.
23. Bottom boxes: the batch program "also runs review app", and the database holds a full
    audit trail. The mock has no review application, and Part 2 does not specify an
    audit-trail schema.
24. Microsoft only: "text-embedding-3-small", "Data Zone Standard", managed identities, an
    in-memory vector store, and deferring Azure Machine Learning. None is in the design and
    none applies to the mock.
25. Vocabulary differs: catalogue/catalog, matcher/ML model, agreed/models agree,
    Sustaining/After handoff, Weeks 1-3 timings.

---

## Arithmetic checked

Computed with the standard library. Every number the design states holds.

- 14 in 200: a path at exactly 90% right has 14 or fewer wrong 9.3% of the time, and 15 or
  fewer 14.3%. So 14 is the largest count under one in ten.
- 2 of 60: 2 or fewer wrong 5.3% of the time, 3 or fewer 13.7%. So 2 is correct, but the mock
  passes a path at the bar about one time in 19, and the observed bar moves from 93% right
  (14/200) to 96.7% right (2/60).
- Stop line: 12/60 and 40/200 are both exactly 20% and provisional; 13/60 and 41/200 are the
  first stopped counts. Consistent.
- A path that is truly 95% right is confirmed 41.7% of the time at 60 (92.2% at 200) and
  provisional 58.3%. The design's claim holds before any simulated SME error is added.
- No confirm count exists for a sample under 22 (0.9^n is above 0.1). This matters only if the
  sample size becomes a setting.
- Cap: 2% of 1,500 is 30 employees, about 12 average long-tail titles.
- Sizes are mutually achievable: 600 employees on 40 titles (average 15) and 900 on 360
  (average 2.5). A Zipf curve needs an exponent near 0.65 to 0.7; a plain 1/rank curve gives
  65%, not 40%. There are 6 roles per family, so a shortlist of ten always holds at least 4
  out-of-family roles.
- A sample of 60 is feasible for the models-agree path (about 170 to 250 titles) only if 400
  counts cleaned rows. For the agent-settled path it is feasible only if the stand-ins
  disagree on about 29% or more of scored titles; otherwise that path is fully labeled.

---

## Library facts

Checked on 2026-10-04 against the package index, the hosted API's reference and, where noted,
by running the library. They are what the entries above rely on.

**The anthropic SDK (extra `live`)**

- Package: anthropic 1.11.0, requires Python 3.10 or later, classifiers 3.10 to 3.14, a
  pure-Python wheel of 1.7 MB. It resolves to 15 packages, about 5 MB, on 3.12 and 3.14
  (pydantic 2.13.5, pydantic-core 2.46.5 with a cp314 wheel, jiter 0.17.0, httpx2 2.13.1).
- The 1.x SDK is built on httpx2, not httpx. Mocking tools that patch httpx silently stop
  seeing its traffic. For tests, pass
  `http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))` or
  inject a fake client object.
- Model: `claude-haiku-4-5` is a valid alias (dated ID `claude-haiku-4-5-20251001`), 200K
  context, 64K output, USD 1 input and USD 5 output per million tokens. Use the alias as
  written; do not append a date.
- Structured output is supported on that model. Either
  `client.messages.parse(..., output_format=PydanticModel)` giving `response.parsed_output`,
  or `client.messages.create(..., output_config={"format": {"type": "json_schema", "schema":
  {...}}})` and `json.loads` of the text block. The raw-dict `output_format={...}` form is
  gone in 1.x. Every object needs `additionalProperties: false` and `required`; `enum` is
  allowed; `minimum`, `maximum`, `minLength` and `maxLength` are not supported, so the 1-to-10
  range and a non-empty quote are enforced in code. Structured output cannot be combined with
  the API's citations feature, so the quote is an ordinary schema field checked by shared
  code.
- Tools and `output_config.format` can be used in the same request. Strict tools: `strict:
  true` on the tool plus `additionalProperties: false`. `tool_choice` `auto` with
  `disable_parallel_tool_use: true` limits a turn to one tool call; without it one turn may
  hold several tool calls and all results must go back in one user message. `{"type":
  "none"}` withdraws tools. Forced tool choice (`any` or a named tool) works on the example
  model but returns an error on the newest models.
- Sampling: `temperature`, `top_p` and `top_k` were removed from the 1.x method signatures.
  The example model still accepts them through `extra_body`, but newer models reject them. So
  repeatability comes from the recording, not from temperature 0. The effort parameter errors
  on the example model, which does no thinking unless asked.
- Loop: the SDK's tool runner is beta; a manual loop on `client.messages.create` checking
  `stop_reason` (`tool_use`, `end_turn`, `max_tokens`, `refusal`) fits "plain loop, no agent
  framework". The SDK retries 408, 409, 429, 5xx and connection errors twice by default; the
  default timeout is 10 minutes.

**Embedding libraries (resolution for Linux x86_64)**

- model2vec 0.9.0 (extra `local`): requires Python 3.10 or later; classifiers stop at 3.13 but
  the wheel is pure Python and every dependency has a 3.14 wheel (numpy 2.5.3 requires 3.12 or
  later; tokenizers 0.23.2 and safetensors 0.8.0 are abi3). 23 packages, about 29 MB. Run with
  `minishlab/potion-base-8M` (MIT licence, revision
  `bf8b056651a2c21b8d2565580b8569da283cab23`) on Python 3.14.7 and 3.12.13: 256-dimension
  float32 vectors, warm load 0.5 s, 1,600 texts in 0.03 s, peak memory 130 to 190 MB,
  byte-identical vectors across repeat runs and across the two Python versions on one machine;
  not tested on another machine. The default download fetches the whole model repository (59
  MB on disk, including an ONNX copy) and contacts the hub on every load unless offline mode
  or a local path is used. It is a static (non-contextual) model: unexpanded abbreviations
  score poorly (cosine 0.15 between "senior account manager manages a portfolio of enterprise
  accounts" and "acct mgr ii"), so it depends on the abbreviation expansion in the prepare
  step.
- Not chosen: sentence-transformers 6.1.0 needs torch, transformers, scikit-learn and scipy;
  its default resolution is 60 packages and about 3,099 MB of wheels. fastembed 0.8.1 is 28
  packages and about 59 MB with onnxruntime, with models of 67 to 90 MB. Neither was installed
  or run.

---

## How the build is cut

The work is cut into one shared branch and fifteen component branches in five stages. Branches
within a stage are built in parallel; each stage starts from main after the previous stage is
merged.

### Rules for every branch

1. **Edit only the files you own.** The table below names them. Tests go in new files named
   after the module, `tests/unit/test_<module>*.py` and `tests/integration/test_<module>*.py`.
   The same file name may exist in both folders. Rule 9 and "Work that crosses a branch
   boundary" below name the only exceptions.
2. **The shared files are frozen** once the scaffold is merged: `pyproject.toml`, `uv.lock`,
   `Makefile`, `.gitignore`, the CI workflow, this record, `titlestd/contracts.py`,
   `settings.py`, `seeding.py`, `datafiles.py`, `reference.py`, `cli.py`, `tests/conftest.py`,
   the mini data set and the scaffold's own tests. If a change to one cannot be avoided, make
   it in a commit of its own and say so in the merge message, because another branch may be
   reading the same file.
3. **Do not re-decide.** Where an entry above settles a point, build it as written and cite
   the entry. Where nothing settles it, decide in the spirit of the design, write the decision
   into the docstring of the module it concerns and report it for the merge message. New
   decisions are not added to this record: it is one of the frozen files, and branches built
   at the same time would all be editing it.
4. **A branch in a stage cannot call a module built in the same stage.** Each stub's docstring
   says which modules it must not call and what to use instead.
5. **Keep each module's docstring header**: its owner branch, and the design sections and
   entries that govern it.
6. **Randomness and order** go through `titlestd/seeding.py` (S04, S15). Sort before letting a
   set reach an output.
7. **The gate is `make check`** (S14), on the branch and again on the merge result; merge
   with `--no-ff`; nothing is pushed (Q34). A branch also passes `make check-all` before it is
   merged, and main passes it after the last merge of each stage: that is where the suite runs
   on Python 3.13 and 3.14 during this build (Q38).
8. **Hygiene** (2.8, Q28): nothing from outside the repository folder; no real company, client
   or person in code, data, docs, branch names or commit messages; no secrets, keys or `.env`
   files; invented emails use `example.com`.
9. **A defect in a module merged in an earlier stage is fixed by the branch that finds it.**
   The owner branch is finished by then, so nobody else will. Fix it in a commit of its own,
   with a test that fails without the fix, and report the module and the defect for the merge
   message. Change only what the defect needs: the module's interface and docstring header
   stay. A module of a branch in your own stage is never edited, because its owner is changing
   it at the same moment: work around it and report it. This is how a stage 3, 4 or 5 branch
   deals with a fault it finds in earlier work, and stage 5 is the first place where the
   pipeline, live mode and replay run together.

The mini data set and its hand-written title rows (`MINI_ROWS` in `tests/conftest.py`) let
stage 1 and stage 2 branches test before the generated mock data and the prepare step exist.
The prepare step must reproduce those rows exactly.

### The tests of section 2.7

| | Test |
|---|---|
| T1 | No title outside the catalog can reach the load file. |
| T2 | A quote that is not in the record sends the title to a person. |
| T3 | A compliance-flagged role is never settled by a machine. |
| T4 | The agent cannot pick a role it was not shown or one outside the family. |
| T5 | No model is ever shown a generic family role: it is never on a shortlist, in a numbered list or in a search result. |
| T6 | Golden-set titles never enter the look-up table or the examples. |
| T7 | The generic family role is never assigned past its cap, or to a Wave 1 or compliance-flagged title. |
| T8 | Applying the reverse file to the load file restores every legacy title. |

### The stage plan

The last column lists every test a branch owns: the tests of section 2.7 by their number
above, then the tests and other deliverables that an entry decides, with the entry. A test an
entry asks for and no branch owned would be dropped in an unattended build, so each has one
owner here.

| Stage | Branch | What it delivers | Files it owns | Depends on | Tests it owns |
|---|---|---|---|---|---|
| 0 | `chore/scaffold-contracts` | This record; tooling and `make check`; settings; seeding; shared types and the three model interfaces; readers and writers for the data files; the command line; a stub for every component; the mini data set | `pyproject.toml`, `uv.lock`, `.python-version`, `Makefile`, `.gitignore`, `.github/workflows/ci.yml`, `docs/build-decisions.md`; `titlestd/__init__.py`, `__main__.py`, `cli.py`, `contracts.py`, `settings.py`, `seeding.py`, `datafiles.py`, `reference.py`; the truth-file functions in `people.py`; `tests/conftest.py`, `tests/fixtures/mini/` and the tests for all of these | none | None of the eight. The structural tests of S15, including the structural half of Q22 and the import guard; the gate environment (S14); the seams (S09). |
| 1 | `feat/catalog-naming` | Catalog access; grade to level; the naming rule with and without a level word; `is_catalog_title` | `titlestd/catalog.py` | scaffold | T1 (the guard function), T5 (`model_visible_roles`). The worked example of 1.3 (Q36). |
| 1 | `feat/prepare` | Cleaning, masking, grouping to title rows, the record of each row | `titlestd/prepare.py` | scaffold | None of the eight. It reproduces `MINI_ROWS`; the records do not depend on the order of the employees file (Q11). |
| 1 | `feat/model-switch-offline` | The switch; the offline embedder; the stand-in LLM and agent | `titlestd/switch.py`, `offline_embedder.py`, `standin.py` | scaffold | None of the eight. Behavioural test (a) of Q22. |
| 1 | `feat/simulated-people` | Wave 1 approvals, blind labels and review-queue decisions simulated from the truth | `titlestd/people.py` (all but the truth-file functions) | scaffold | None of the eight. One label per title whatever the order of calls (Q27); "could not decide" only on review-queue decisions (Q01); "no suitable role" only without a truth row (Q03); family codes matched after trimming (S05). |
| 2 | `feat/mock-data` | The seeded generator and its committed output | `titlestd/mockdata.py`, `data/` | prepare, catalog-naming | None of the eight. The data tests of Q09, Q24 and Q35; byte-for-byte regeneration (Q06); every title has a truth row (Q03); the reference lists share no word with the data (Q14); the three flagged roles sit in three families (Q23). Not the run shape of Q23: see below. |
| 2 | `feat/lookup-ml-match` | The look-up table and the examples with the golden-set guard; the nearest-neighbour match; ML-alone eligibility | `titlestd/lookup.py`, `matcher.py` | catalog-naming, prepare, model-switch-offline | T6 (the guard), T5 (never embedded, never on a shortlist). ML-alone eligibility (Q04); a title with a person's name joins the look-up and never the examples (Q14); a decision on a title with no usable family joins under its key (Q26). |
| 2 | `feat/llm-fallback-rules` | The record block, the seeded shuffle and numbering, the number-to-role mapping, the quote check; the rules of 1.7 | `titlestd/fallback.py`, `rules.py` | catalog-naming, prepare, model-switch-offline | T2, T3 (the rule), T5 (never in a numbered list). The failed quote and the out-of-range number, with an injected answer (Q15, Q17). |
| 2 | `feat/golden-audit-status` | The golden-set draw and scoring; the audit sample, its limits and three outcomes; the statuses of an audited path | `titlestd/golden.py`, `audit.py` | simulated-people | None of the eight: T6 needs the look-up, built beside it, so it is tested there and end to end. The three outcomes, the path of 60 or fewer and the stopped-path release setting, on constructed counts (Q23); the frame of the golden set (Q08, Q14, Q26). |
| 2 | `feat/close-release-files` | The generic role under the cap; the load, reverse, exceptions and review-queue files; `apply_reverse` | `titlestd/close.py`, `release.py` | catalog-naming | T1 (the writer refuses), T7, T8. The file checks of Q05; "no suitable role" is never generic and never counts against the cap (Q03); a leftover with no usable family stays on its legacy title (Q26). |
| 2 | `feat/local-embedder` | `embedder = local` behind the `local` extra | `titlestd/local_embedder.py` | model-switch-offline | None of the eight. The lazy import with a stub, and the message for a missing extra (Q30, S09); one smoke test marked `local_model` (Q30). |
| 3 | `feat/agent` | The agent loop, the two searches, the numbering, the case | `titlestd/agent.py` | lookup-ml-match, llm-fallback-rules, model-switch-offline | T4, T5 (never in a search result), T3 (an injected reviewer that returns a flagged role). The cap, not-shown, out-of-family and zero-search tests on the shared loop with a scripted backend (Q20). |
| 4 | `feat/pipeline-report-cli` | The run: set-up, Wave 1, Wave 2, the close, learning between waves; the table and footer; the state file; the tuning of the run shape and the frozen table (below) | `titlestd/pipeline.py`, `report.py`, `store.py`, `tests/fixtures/expected/`; for the tuning only, the difficulty constants in `titlestd/mockdata.py` and `data/` | every branch of stages 1 to 3 | All eight, end to end: T1, T3, T5, T6, T7 and T8 on the run's files and state, T2 and T4 with an injected misbehaving model. Behavioural test (b) of Q22. The run shape of Q23 and its invariants (Q05, Q16); two runs in separate processes give the same bytes (Q06, Q23); the frozen table (Q23, Q38); one run at a high `undecided_rate` (Q01); the default run under the import guard (Q32 a); the 60-second limit (Q38); spies showing no model is called for a title with a person's name or no usable family (Q14, Q26); a seed hit on the Wave 1 list (Q13); the ML-alone route switched on (Q04). |
| 4 | `feat/live-replay` | `llm = anthropic`, `llm = replay`, the recorder and its key | `titlestd/live.py`, `replay.py`, `recording.py` | model-switch-offline, llm-fallback-rules, agent | None of the eight. Tiers (b) and (c) of Q32 at the level of `choose` and the agent loop over the mini data set; the start-up errors of Q31; the failures of Q39 with a fake client; the message for a missing extra (S09). |
| 5 | `feat/recordings-replay-e2e` | The round trip of a whole run with a scripted client, first; then the smoke run, the one live recording, the committed recordings and the replay run in the gate | `recordings/`, `tests/integration/test_recordings*.py` | pipeline-report-cli, live-replay | None of the eight of its own: the replay run passes the same invariants as the default run. The round trip of a whole run (Q32 b); the replay run under the import guard (Q32 a); the committed recording replays with zero misses, marked `replay` (Q02, Q32); no recording holds a key-shaped string (Q29). |
| 5 | `docs/readme-hygiene` | The README; the hygiene checks of 2.8 | `README.md`, `tests/unit/test_hygiene.py` | pipeline-report-cli | None of the eight. Mock emails use `example.com`; no tracked file holds a key-shaped string and no `.env` file is tracked (Q02, Q28). The README statements listed below. |

### Work that crosses a branch boundary

**The run shape (Q23).** The bar of Q23 is reached by shaping the generator, but it can first
be measured when the pipeline exists, two stages after the generator is merged. So the two
halves have two owners. In stage 2, `feat/mock-data` is done when its data tests pass and its
difficulty knobs are named constants at the top of `titlestd/mockdata.py`. In stage 4,
`feat/pipeline-report-cli` measures the default run against the bar. If the run misses it,
that branch changes those constants, regenerates `data/` with the `generate-data` command and
commits the two together in a commit of its own: one tuning round. There are at most three
rounds (Q23). It changes nothing else of the generator, the data tests of
`feat/mock-data` must still pass, and the default seed, the stand-ins and every threshold stay
as they are. If the bar is still missed after the third round, the branch ships what the seed
yields: its test asserts the part of the bar that holds, and the part that does not is stated
in that test, in the docstring of `titlestd/pipeline.py` and for the merge message, so that
the README can say it. Nothing in stage 4 collides with this: `feat/live-replay` tests on the
mini data set and does not read `data/`.

**The frozen table (Q23, Q38).** `tests/fixtures/expected/default_run.txt` holds the text the
default run prints. `feat/pipeline-report-cli` writes it from a real run on Python 3.12, never
by hand, as its last commit, after any tuning round, together with an integration test that
compares the output of the default run with it byte for byte and is skipped unless the
interpreter is Python 3.12. Q23 freezes the table only once the shape is final, and that is
the moment. Whoever later changes what the default run prints, for example with a fix under
rule 9, regenerates the file in the same commit.

**The round trip of a whole run (Q32 b).** `feat/live-replay` tests recording and replay at
the level of `choose` and the agent loop, because the pipeline is built beside it. The whole
run is the first deliverable of `feat/recordings-replay-e2e`, before the smoke run and before
any money is spent: on the mini data set, live mode with a scripted client under the model ID
`test-double` runs through the pipeline and the real recorder into a temporary folder; replay
then gives the same decisions and files, apart from the answer label; one changed input gives
a miss, a non-zero exit and no output files. It uses the seams of S09 and commits no
recording.

**The runs under the import guard (Q32 a).** Both use `IMPORT_GUARD` of `tests/conftest.py`
(S15): the default run in `feat/pipeline-report-cli`, the replay run in
`feat/recordings-replay-e2e`, which is the first branch that has a committed recording.

### What the README must say

`docs/readme-hygiene` writes the README. These statements are required, and the README says
each in its own words:

| Source | Statement |
|---|---|
| 2.4 | Offline accuracy shows the flow, not model quality. |
| 2.4 | At 60 titles per path the audit has little power, so provisional is the usual result in the mock. |
| Q01 | The "could not decide" rate of simulated reviewers is a simulation device. |
| Q14 | The invented name list stands in for a real name masker. |
| Q18 | Changing the seed, the data or the embedder invalidates a recording. |
| Q23 | The shape of the default run, and that a machine path of 60 titles or fewer is labeled in full and gets no status. |
| Q30 | How `embedder = local` gets its model, and the result of the `local_model` smoke test if it failed or timed out. |
| Q37 | The three-SME agreement exercise of set-up is left out. |
| Q02, Q39 | The `record` command; the key comes only from the environment, exported in the shell or passed from a file outside version control with `uv run --env-file`. |
| S09 | The command that runs the program with an optional extra. |
