# Job-title standardization: design and mock scope

A large employer has tens of thousands of employees on thousands of free-text job titles
("Sr. Acct Mgr - NYC", "ACCT MGR II", "Senior Account Manager"). The same job has many
spellings, so pay analysis, headcount planning and reporting are unreliable. This system maps
every employee onto a standard role from an HR-approved catalog and gives each a standard Short
Title and Long Title.

This file is the starting point for a small mock of that system: mock data, a mock workflow, no
real employer and no real people. Part 1 describes the design. Part 2 says what the mock builds.

Where this file and the diagrams in this folder differ, this file wins. Do not copy rules from
the diagrams: they are older and differ in three places. They list three audited paths where
there are two, say the ML model switches on in Wave 2, and give a "no suitable role" title a
generic role.

---

## Part 1. The design

### 1.1 The idea in four lines

- **Machines pick, they never write.** Every Short Title and Long Title comes from a catalog that
  HR approved. A model only chooses a catalog role by number, so it cannot invent a title.
- **Cheapest check first.** An exact look-up, then an ML match, then an LLM, then an agent, then
  a person.
- **Two methods must agree.** A title goes out without a person only if the LLM and the ML model
  agree, or an agent settles their disagreement, and only after HR has audited a random sample
  of that machine path.
- **Every title carries a status**, so anyone using it knows how far to rely on it.

### 1.2 Vocabulary

| Term | Meaning |
|---|---|
| Title | What the employer has today: free text on an employee record. |
| Role | A catalog entry: a job within a job family, for example Account Manager in Sales. |
| Catalog | Roles per family, one level ladder, a grade-to-level table, and the naming rule for Short and Long Titles. HR SMEs approve it. Some roles carry a compliance flag. |
| Level | Junior, Associate, Senior, Lead, Principal. It comes from the employee's grade through HR's grade-to-level table. No model predicts it. |
| Look-up table | Titles a person has already decided, each mapped to a role. |
| Golden set | A fixed random sample of long-tail titles that HR SMEs label without seeing the system's answer. It is re-scored at every wave and never becomes a look-up entry or an example. |
| Machine path | How a title was decided without a person: "models agree" or "agent-settled". |
| Random audit | Before a machine path is released, HR SMEs label a random sample of it, again without seeing the system's answer. |
| Release thresholds | What the HR lead and CIO sign before any result exists: the accuracy bar and what happens when a path misses it. |
| Generic family role | One catalog entry per family that names the family and no job. Only code assigns it, as a last resort. It is never embedded, never on a shortlist, never numbered for the LLM or the agent, and never returned by a search. |

### 1.3 The catalog

A catalog role is Family x Role, and an employee's standard title is Family x Role x Level.

| | Example |
|---|---|
| Legacy titles | "ACCT MGR II" (Sales, grade 8) and "Senior Account Manager" (Sales, grade 7) |
| Role | Sales x Account Manager |
| Level, from grade | Grades 7 and 8 both read as Senior in HR's table |
| Long Title | Senior Account Manager |
| Short Title | Sr Acct Mgr |

The naming rule is fixed in the catalog: Long Title is the level word plus the role name, and
Short Title is the short level word plus the short role name. A role also has a title with no
level word, used when an employee has no usable grade.

### 1.4 Inputs and what each is used for

| Input | Used for |
|---|---|
| Cleaned title plus the first 200 words of the job description | The ML model's embedding |
| Job family code | The family rule; the agent may only pick in-family roles |
| Grade | Sets the level through HR's grade-to-level table |
| Department | Context for the LLM, the agent and reviewers |
| Headcount per title | The most widely held titles always go to a person |
| Approved titles per role | The "enough examples" test |
| Title history, where it exists (optional) | Context for reviewers; not embedded |
| Other free-text notes about the job (optional) | Off until privacy and usefulness are cleared; could stand in for a missing description |

Nothing about the person (name, pay, age, tenure) is used. Names are masked before any text
reaches a model. A title that contains a person's name goes to a person and to no model.

The unit of decision is one row per cleaned title within a family, with its headcount. The
decision is the role. Level is then applied per employee from the grade.

### 1.5 The models

- **ML model: nearest neighbors on text embeddings.** It matches on meaning, shows what it
  matched for the audit trail, needs no labels to start, and improves with every human approval.
  The embedding model is off the shelf and is not trained.
- **LLM and agent: one small hosted LLM.** It reasons on the hard cases. Structured output and a
  required quote keep it from inventing. It is not fine-tuned.
- **Why not a trained classifier?** There are no labels on day one. This design produces the
  labels a classifier would need later.
- **Why not the LLM alone?** The ML model gives the same answer every time on the key fields, and
  it is a second opinion the LLM's pick must match.

### 1.6 The flow

1. **Prepare.** Clean each title (lower-case, strip punctuation, locations and requisition
   numbers, expand a small abbreviation list such as sr to senior), mask names, group to one row
   per title and family.
2. **Look-up.** Exact match on cleaned title plus family in the look-up table. A hit takes that
   role with no model.
3. **ML match.** An off-the-shelf embedding model turns the title and description into a vector.
   A nearest-neighbor match against every catalog role except the generic family roles (its
   name and one-line definition) and every human-approved title returns a shortlist of ten
   roles, a first choice, and a margin (the gap between first and second).
4. **LLM fallback.** The ML model's choice may stand alone only if the margin clears a cut-off
   and the chosen role has at least three human-approved examples. Otherwise the title goes to
   the LLM. During the initial backfill examples are scarce, so every look-up miss goes to the
   LLM, and the ML model supplies the shortlist and a second opinion.
   The LLM sees the masked record and the ten roles, numbered and shuffled. It must answer with
   a number from 1 to 10, or "no match", or "insufficient data", plus a short quote from the
   record as evidence.
5. **Rules.** Code checks every proposal (table in 1.7).
6. **Agent.** For a disagreement between the LLM and the ML model, a family mismatch, or a
   "no match", one LLM agent reviews the case. Unlike the fallback it has two read-only searches:
   the whole catalog (generic family roles excluded), and similar human-approved titles. It
   answers the same way as the LLM: code numbers every in-family role it has been shown, and
   the agent returns a quote and a number, or escalates. Picking the LLM's role confirms it;
   picking another corrects it. Its answer goes back through the rules.
7. **People.** HR SMEs and reviewers decide everything no machine may settle (1.8).
8. **Random audit.** Each machine path is audited before release (1.9).
9. **Release.** After the HR lead signs, the system writes a load file (one row per employee:
   the new titles and their status) and a reverse file (how to restore each employee's legacy
   title). The employer's HR-system team loads them; the system never writes to the HR system.
10. **Learning.** Nothing is trained. Titles a person approved or labeled join the look-up table
    and the ML model's examples. A title the agent settled does not, until a person approves it.

### 1.7 The rules

| Check | On failure |
|---|---|
| The role exists in the catalog | A person |
| The LLM's quote appears word for word in the masked record | A person |
| The proposed role, or any of the ML model's top three, is compliance-flagged | A person |
| The LLM answered "insufficient data" | A person |
| The LLM and the ML model's first choice agree | The agent |
| The proposed role is in the employee's recorded family | The agent |
| The LLM answered "no match" | The agent |

If several fail, a person beats the agent. Everything the agent returns is checked by the same
rules except agreement. If it fails any, or the agent escalates, the title goes to a person.

### 1.8 What people always decide

- The most widely held titles (the Wave 1 list), each approved by an HR SME.
- Any title that touches a compliance-flagged role.
- Any failed catalog or quote check, any "insufficient data", anything the agent escalates.
- A title marked "no suitable role" goes to the catalog owner, who adds a role or names one.
- At the close, a title nobody could decide takes its family's generic role, marked provisional
  and capped at 2% of employees. A widely held or compliance-flagged title never does.

### 1.9 Confidence, the random audit and statuses

The ML margin and the example count only decide who looks next. Trust comes from people's
labels, never from a model's own score.

In set-up, three HR SMEs label the same 30 titles, which gives an estimate of how often people
agree with each other.

Before a machine path is released, HR SMEs label a random sample of 200 titles from it. The
accuracy bar is 90 in 100.

| Wrong in 200 | Outcome |
|---|---|
| At most 14 | Confirmed |
| 15 to 40 | Provisional: released, marked, with its measured error rate |
| 41 or more | Stopped: the HR lead and CIO release it as provisional or keep those employees on legacy titles |

Fourteen is stricter than 90% on purpose: a path sitting exactly at the bar passes only about
one time in ten.

| Status | Meaning |
|---|---|
| Approved | A named person decided it |
| Confirmed | No person read it, but its machine path passed its random audit |
| Provisional | Neither: a path that missed the bar, or a generic family role |

### 1.10 Waves

| Wave | What happens |
|---|---|
| Set-up | Catalog approved; release thresholds signed; a small pilot timed; golden set drawn |
| Wave 1 | The most widely held titles: the machine proposes, an HR SME approves every one. The golden set is labeled. The agent only recommends. A dry run scores every title once and releases nothing, to size the Wave 2 review queue. |
| Wave 2 | The long tail is scored for release. The two machine paths are audited and released. Everything else goes to a review queue. From here the agent's answer stands. |
| Wave 3 | People clear the queue; leftovers take a generic family role; final release |
| After handoff | A weekly run on new titles. Known titles hit the look-up. A person approves every new title, and the agent only recommends. |

### 1.11 Architecture

One batch program and one database, with a small review application for people.

| Box | Tech-agnostic | Microsoft |
|---|---|---|
| Files | Object store (in, config, out) | Azure Storage |
| Batch program (the automated workflow) | One container | Azure Container Apps job, image in Azure Container Registry |
| Agent | Inside the batch program | Microsoft Agent Framework, inside the batch program |
| Models | Embedding model and LLM, API or self-hosted; open-source name masker | Microsoft Foundry: embedding and chat deployments; Azure Language to mask names |
| Database (system of record) | PostgreSQL | Azure SQL Database |
| Review application | Web app behind single sign-on, four roles | Azure App Service with Entra ID, four app roles |
| Platform | Private network, logs, single set-up file | Private endpoints, Log Analytics, one Bicep file |

Detailed diagrams: `architecture-tech-agnostic.png` and `architecture-microsoft.png`.

---

## Part 2. The mock

### 2.1 Purpose and size

Show the whole idea in under a minute, offline, on made-up data. Keep it small and plain: one
package, the standard library by default, a local SQLite file or plain files for state.

Suggested size: about 1,500 employees on about 400 titles, 8 job families, about 48 roles plus
one generic role per family, 5 levels, grades 1 to 10, 3 compliance-flagged roles. Headcount is
skewed: the 40 most widely held titles cover roughly 40% of employees.

### 2.2 In and out

In the mock: prepare, look-up, ML match with its margin, LLM fallback, the rules, the agent,
simulated people, the golden set, the random audit and its three outcomes, statuses, the
generic family role and its cap, the load file and the reverse file, Wave 1, Wave 2 and the
close.

Not in the mock: a review application (people are simulated), a real HR system, cloud
deployment, sign-in, the set-up pilot, the dry run, the weekly run after handoff, and the two
optional inputs.

### 2.3 Mock data

| File | Columns |
|---|---|
| Catalog roles | role_id, family_code, role_name, short_name, definition, compliance_flag, is_generic |
| Level ladder | level, long_word, short_word |
| Grade-to-level table | grade, level |
| Abbreviations | short_form, long_form |
| Employees | employee_id, job_title, job_family_code, department, grade, job_description |
| Seed look-up (a few titles a person has already decided) | job_title, job_family_code, role_id |
| Truth (hidden from the system) | job_title, job_family_code, true_role_id: one true role per title within a family |

Make the titles messy on purpose: abbreviations, mixed case, location and requisition suffixes,
level words in several forms, a few with a person's name, some empty descriptions, a few missing
grades. All names, emails and companies are invented; use the reserved `example.com` domain.

### 2.4 Settled choices

- **Stand-in models never read the truth file.** The offline LLM and agent decide by word
  overlap after abbreviation expansion. The README says offline accuracy shows the flow, not
  model quality.
- **People are simulated from the truth file**, with a small seeded error rate, and every
  simulated decision is marked as simulated.
- **The seed look-up** exists so that the look-up step has hits to show.
- **Role score** is the best cosine similarity across a role's catalog text and its approved
  titles. The margin cut-off and the three-example rule are settings. The ML-alone route is
  built and off by default.
- **Audit at mock scale.** Sample 60 titles per path. Use the same statistics as 14 in 200: a
  path is confirmed if the number wrong is at most the largest count that a path at exactly 90%
  right would pass at most one time in ten, computed from the binomial for the sample size
  (2 of 60). It is stopped above 20% wrong (more than 12 of 60, the same line as 41 or more in
  200); otherwise it is provisional. If a path holds 60 titles or fewer, every title in it is
  labeled: each takes its label with status Approved, and the path gets no status. A stopped
  path is left on legacy titles unless a setting releases it.
- **Say what the small audit can show.** At 60 per path the audit has little power: a path that
  is truly 95% right is confirmed less than half the time, so provisional is the usual result
  in the mock. The README says so.
- **The agent** is a plain loop: at most four search calls, then one structured answer. No
  agent framework.
- **Wrong**, in the random audit and on the golden set, means the machine's role differs from
  the simulated SME's label for that title. The audit draws each title once, so it measures
  accuracy by title, not by employee. Level is not scored.
- **Missing data.** No description: use the title alone. No grade: the title with no level word,
  and the employee is listed. No usable family code: a person.

### 2.5 Models and settings

Three calls sit behind a switch: `embed(texts)`, `choose(record, ten roles)` and
`review(case, tools)`. Masking, numbering, the quote check, the search cap and the
number-to-role mapping stay in shared code, so no model output can become a role by itself.

| Setting | Values | Default |
|---|---|---|
| `embedder` | `offline` (built-in, no install), `local` (a small local embedding model) | `offline` |
| `llm` | `offline` (deterministic stand-in), `replay` (recorded answers), `anthropic` (live) | `offline` |
| `llm_model` | a small hosted model, for example `claude-haiku-4-5` | |

- The only secret is `ANTHROPIC_API_KEY`, read by the SDK from the environment and only when
  `llm = anthropic`. It is never written to a file in the repository.
- Import the SDK and the embedding library lazily, so the default run installs nothing.
- In live mode, record each answer in a file keyed by a hash of the step, model ID, prompt and
  request. In replay mode a miss is an error; never fall back silently. Recordings are never
  written by hand.
- Label every answer as "stand-in", "live" or "recorded <date>, <model>".

### 2.6 What a run shows

One command runs set-up, Wave 1, Wave 2 and the close on the mock data and prints one table.

- Rows: look-up hit, approved in Wave 1, models agree, agent-settled, review queue, generic
  family role, left on legacy title.
- Columns: titles, employees, share of employees, audited, wrong, status.
- Footer: coverage by status, the share of titles sent to a person, golden-set accuracy of the
  ML model's first choice and of the LLM's pick, and one line saying whether answers came from
  stand-ins, a recording or a live model.

It also writes the load file, the reverse file and the review queue.

### 2.7 Tests worth having

- No title outside the catalog can reach the load file.
- A quote that is not in the record sends the title to a person.
- A compliance-flagged role is never settled by a machine.
- The agent cannot pick a role it was not shown or one outside the family.
- No model is ever shown a generic family role: it is never on a shortlist, in a numbered list
  or in a search result.
- Golden-set titles never enter the look-up table or the examples.
- The generic family role is never assigned past its cap, or to a Wave 1 or compliance-flagged title.
- Applying the reverse file to the load file restores every legacy title.

### 2.8 Repository hygiene

- Nothing from outside this folder goes into the repository.
- No real company, client or person is named anywhere, including commit messages.
- No secrets. `.env` is ignored. No keys in recordings.
