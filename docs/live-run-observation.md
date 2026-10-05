# Live run observation: a third of the model's quotes fail the word-for-word check

In the one live run of this mock, the hosted model's evidence quote was not in the record word
for word for about a third of the titles it mapped. The quote rule of section 1.7 of the design
sent every one of them to a person, as it should, and that roughly doubled the number of titles
a person had to decide.

Nothing was changed in response. The prompt and the committed recording are exactly as the run
made them, so everything below can be reproduced from the repository (see the end of this note).

## The run

One run on the default data, seed and settings on 2026-10-05, with `claude-haiku-4-5` as the LLM
and the agent and the offline embedder for the ML match: 480 requests. People were simulated, as
in every run of the mock.

| | Offline stand-ins | Live model |
|---|---|---|
| Models agree | 252 titles; 1 wrong of 60 audited; Confirmed | 176 titles; 0 wrong of 60 audited; Confirmed |
| Agent-settled | 34 titles, labeled in full | 9 titles, labeled in full |
| Sent to a person | 106 of 400 titles (26.5%) | 207 of 400 titles (51.8%) |
| Generic family role | 6 employees | 30 employees, the cap exactly |
| Golden set, LLM's pick right | 34 of 40 | 36 of 40 |

The live model picked the right role slightly more often than the stand-in on the golden set.
What rose was the review workload, and the reasons in the review-queue file show why. A title
can carry more than one reason.

| Reason a title was in the review queue | Offline | Live |
|---|---|---|
| Quote not in record | 0 | 117 |
| Compliance-flagged role | 35 | 36 |
| Models disagree | 13 | 26 |
| Insufficient data | 4 | 15 |
| Agent escalated | 20 | 3 |
| Role outside recorded family | 12 | 0 |
| No match | 10 | 4 |
| Person's name | 5 | 5 |
| No usable family code | 3 | 3 |
| Invalid model answer | 0 | 1 |
| **Titles in the queue** | **66** | **167** |

The stand-in never fails the quote check, because it copies its quote out of the record by
construction. That is one reason the design says offline accuracy shows the flow, not model
quality.

## Why the quotes fail

In Wave 2 the model named a role for 322 titles, and 112 of those answers failed the check. It
had been told to copy the words "character for character from the title or the description".
What it returned instead:

| What the quote was | Answers |
|---|---|
| A listed role's whole definition, copied from the list of roles instead of the record | 45 |
| Other text that is not one unbroken passage of the record: its clauses reordered, or the title joined to a definition | 64 |
| More than 25 words | 3 |

Two examples:

- **A definition instead of the record.** The record's description is "Handles employee
  relations cases and advises managers on staffing and performance." The quote was "Advises
  managers on staffing and performance and handles employee relations cases", which is the
  definition of the role the model chose.
- **A composed sentence.** A title with no description, "customer insights analyst ii", got the
  quote "customer insights analyst analyses satisfaction surveys and reports complaint trends":
  the title followed by the chosen role's definition.

The mock data makes this likely. A description is written from the same duties as its role's
definition, often in another order, so a near-copy of the record sits a few lines below it in
the same prompt.

The picks themselves were sound. Of the 117 titles, 83 were in the queue for this reason alone.
The simulated reviewer named a role for 75 of those, and for 74 it was the role the model had
picked. Those titles would have been released on the models-agree path had the quote been
copied from the record.

## A clearer instruction helps, but only partly

A reworded quote instruction was tried on 42 of the Wave 2 titles, outside the program, for
about five cents. It asked for "a single unbroken run of whole words in the record's own order"
and added "Quote the record only, never the name or definition of a listed role, and do not
join, reorder, shorten or reword what you copy."

| Titles | Quotes passing with the clearer instruction |
|---|---|
| 30 whose quote had failed | 18 |
| 12 whose quote had passed | 11 |

Scaled to Wave 2, that is a failure rate of roughly 20% instead of 35%. The sample is small and
each title was asked once, so the figure is an indication, not a measurement.

The rewording was not adopted. The prompt text is part of every recording key, so adopting it
means recording again, and the repository owner chose to keep the one planned live run and
treat this as a finding.

## Three answers were cut off

Three of the 452 recorded `choose` answers stopped at the limit of 1,024 output tokens
(`max_tokens`) before they were complete, and each was recorded as an invalid answer, which
the rules send to a person. One was the Wave 2 answer for "senior business development
representative", the one title under "Invalid model answer" in the table above; the other two
were the golden-set passes of Wave 1 and of the close for "senior internal communications
specialist", whose Wave 2 answer was complete. The limit was left as it is: like the prompt it
is part of every recording key, so raising it means recording again, and the recording keeps
no text of a cut-off answer to show what the model was writing.

## What it says about the design

- **The rule does its job.** No answer with unverifiable evidence was released by a machine.
  The model cannot talk its way past a check that code performs on the record.
- **The cost is review workload, not accuracy.** The quote check's failure rate is a number to
  measure in a pilot and to size the review queue with.
- **The levers are ordinary ones:** the wording of the instruction, a more capable model, and
  records whose text is not mirrored in the list of roles. None of them changes the flow.

## Reproduce it

Replaying the committed recording needs no key and nothing installed:

```sh
python3 -m titlestd run --llm replay
```

The table it prints is the "Live model" column above. Then count the reasons in the
review-queue file:

```sh
python3 -c "
import collections, csv
reasons = collections.Counter()
for row in csv.DictReader(open('out/review_queue.csv', encoding='utf-8')):
    reasons.update(row['reasons'].split('; '))
for reason, titles in reasons.most_common():
    print(titles, reason)
"
```

The default run, `python3 -m titlestd run`, gives the "Offline stand-ins" column the same way.
