# The architecture page

A local page that shows each stage of a run as a flowchart, the components that act in it, and
each component's job in the words of `DESIGN.md`, beside counts read from the state file of a
run. It is `titlestd/explorer.py`, on the standard library and the two browser libraries in
`static/vendor/`.

## Starting it

From the repository root, make a free run first, then start the page:

```sh
python3 -m titlestd run
python3 -m titlestd.explorer
```

Open http://127.0.0.1:7654/ and stop the page with Ctrl+C. It opens `out/state.sqlite`
read-only and answers only at that address and only to GET. For a run made with `--out-dir`,
pass the page the same `--out-dir`. It does not start without a state file, and it shows only a
free run: one whose `llm` setting is `offline` or `replay` and none of whose answers is labelled
live. A later run replaces the file, and the page then shows the new one. While that run is
under way, or when a run stopped, the page shows the state as it stood at the last completed
step, and says so above each table of counts. What a run writes only at its end, such as the
final decisions and the audits, then reads "not yet". Until set-up has recorded the run's
settings, the page answers that the state is not a free run's.

## What it shows

- A page per stage: the row of the stage in `DESIGN.md` 1.10, the stage as a flowchart with the
  components acting in it drawn with a heavy border and listed again as text, and counts from
  the state file.
- A page per component: its job quoted from `DESIGN.md`, its module, the entries of
  `docs/build-decisions.md` that govern it, the stages it acts in, and counts from the state
  file.

The counts are numbers of rows in the state file and labels defined in the code. No title,
description, quote or other text of a record is shown.

## The four flowcharts

These are the flowcharts the page draws, word for word; a test holds the two equal. Each draws
one stage as `titlestd/pipeline.py` runs it. The nodes with a heavy border are the components
acting in that stage; the others are inputs, lists and outcomes. They are drawn from the code.
The two older diagrams in `docs/`, `architecture-microsoft.png` and
`architecture-tech-agnostic.png`, are the ones `DESIGN.md` says not to copy.

### Set-up

```mermaid
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
    classDef acting fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#1c1917
    class prepare,lookup,golden,switch,statefile acting
```

### Wave 1

```mermaid
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
    classDef acting fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#1c1917
    class lookup,ml_match,fallback,rules,agent,people,golden,learning,statefile acting
```

### Wave 2

```mermaid
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
    audit -->|60 titles or fewer, all audited| nostatus([The path gets no status])
    audit -->|at most 2 wrong of 60| confirmed([The rest Confirmed])
    audit -->|3 to 12 wrong of 60| provisional([The rest Provisional])
    audit -->|13 or more wrong of 60| stopped([Stopped, the rest kept on legacy titles by default])
    people -->|audit labels| learning[Learning]
    rules -.->|golden titles scored again| golden[Golden set]
    rules -.->|answers and routes recorded| statefile[(State file)]
    classDef acting fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#1c1917
    class ml_match,fallback,rules,agent,people,audit,golden,learning,statefile acting
```

### Close

The close is the mock's Wave 3: people clear the queue, leftovers take a generic family role,
and the files are released.

```mermaid
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
    classDef acting fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#1c1917
    class people,learning,golden,ml_match,fallback,rules,generic,release,report,statefile acting
```
