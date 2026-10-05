"""The mock data generator: writes the seven data files of docs/design.md 2.3.

Owner: feat/mock-data (stage 2, after prepare and catalog-naming, because the
generator must use the real cleaner and naming rule). It also owns the
committed files in data/.
Governed by: docs/design.md 2.1 (sizes), 2.3 (files and deliberate mess);
docs/build-decisions.md Q06 (own fixed seed, committed output, byte-for-byte
regeneration), Q09 (raw titles; one true role per cleaned row), Q13 (seed
titles sit in the long tail), Q23 (the difficulty knobs, and the flagged roles
in three families), Q24 (the 40th and 41st rows differ), Q26 (rows with no
usable family code), Q35 (the counts), Q36 (the grade table and the worked
example).

The generator has its own fixed seed, separate from the run seed, so changing
the seed setting never changes the data. Its difficulty knobs are named
constants at the top of this module, never settings of a run. Names, locations
and requisition patterns come from titlestd.reference; every email uses
reference.EMAIL_DOMAIN. All randomness goes through titlestd.seeding.

The run shape of Q23 cannot be measured in stage 2, where the matcher, the
rules, the agent and the pipeline do not exist yet. This branch is done when
its data tests pass (Q03, Q06, Q09, Q14, Q23 for the flagged roles, Q24, Q35)
and the knobs are exposed. feat/pipeline-report-cli measures the shape in
stage 4 and may then change those constants and regenerate data/, in at most
three rounds ("How the build is cut" in docs/build-decisions.md).

It writes six files with titlestd.datafiles and the truth file with
titlestd.people.write_truth; apart from the pipeline it is the only module
that may import titlestd.people.

How the data is made. Every role of the catalog gets ROWS_PER_ROLE title rows:
a wording (the role's own name, one that shares a word with it, or one that
shares none) written at a level form (no level word, a ladder word in front, a
numeral or a level word behind). A title is written out by the catalog's own
naming rule, titlestd.catalog.standard_title, once with long words and once
with short ones, and then recorded in one or more spellings: short forms,
dots, upper or lower case, a location or a requisition number at the end.
Every spelling is cleaned with titlestd.prepare.clean_title when it is made
and must give the planned row, and the finished employees are put through
titlestd.prepare.prepare, which must read back exactly the planned rows with
their headcounts. So what is written here is what the prepare step reads.

Decided on this branch, where the record leaves the detail open:

- The knobs. WORDING_WEIGHTS says how a role's title rows word it, beyond the
  three rows every role has; DESCRIPTION_WEIGHTS how much a description says
  about the job; FLAGGED_ROLE_IDS which three roles carry the compliance
  flag. Every count that Q35 fixes is a constant below the knobs and is met
  exactly whatever the knobs say, because it is filled by taking the rows
  with the lowest seeded draws, not by a chance per row. A draw is keyed on
  the row it is for, so changing a knob changes only what that knob governs.
- Draws go through titlestd.seeding on the stream name "mockdata", which no
  run draws from: seeding.unit, and of the generator seeding.rng returns
  only random() and shuffle(), the two the scaffold pins on every supported
  Python.
- 400 title rows are 48 roles x 8 rows and 16 special rows: 5 with a
  person's name, 3 with no usable family code, 4 worded like no role
  (NO_MATCH_TITLES) and 4 bare titles with no description (BARE_TITLES).
  The special rows hold 20 employees and are written out by hand below.
- Every role has the rows "<name>", "Senior <name>" and "<name> II", so the
  two legacy titles of the worked example of 1.3 are rows of the data:
  "account manager ii" and "senior account manager" in the Sales family.
- Headcounts fall with the rank of a row as rank ** -HEADCOUNT_SKEW. A row
  past the 40th that would tie with it gives one employee to a row of the
  Wave 1 list (Q24). Which row is how widely held follows the size of its
  role, its wording and its level form, with a seeded spread.
- The seed look-up holds one long-tail row of each family, eight in all,
  under the row's commonest spelling (Q13, Q35).
- A row has one description. MIXED_DESCRIPTION_ROWS rows show the rule of
  Q11: in half of them about a third of the employees have another
  description, in the other half none. A wording of one word always has a
  description, so the only rows that draw "insufficient data" from the
  stand-in LLM are BARE_TITLES (Q17).
- The rows with no usable family code carry "", "ADM" and "TBD"; the last
  two are codes the catalog does not have (Q26).
- An employee's grade is one of the grades of the level the title names, or
  any grade when it names none; one time in six it is any grade, as in the
  example of 1.3, where a "II" sits at grade 8. NO_GRADE_EMPLOYEES have none.
- A catalog definition is the first two duties of the role in one sentence.
  A short name is the name in the short forms of the abbreviations file, so
  a role's Short Title cleans to the same row as its Long Title.
- The catalog of the mini data set is part of this one: its two families
  keep their role ids, names and short names here. The ladder and the grade
  table are the same and the mini abbreviations are among these, so a test
  may put the mini rows against this catalog.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from titlestd import catalog as naming
from titlestd import datafiles, people, prepare, reference, seeding
from titlestd.contracts import (
    Abbreviation,
    Catalog,
    CatalogRole,
    Employee,
    GradeLevel,
    InputData,
    LadderLevel,
    SeedLookupRow,
    TitlestdError,
)
from titlestd.settings import Settings

# --- Difficulty knobs (Q23) -----------------------------------------------------
# feat/pipeline-report-cli may change these three, and nothing else here, when
# it tunes the run shape in stage 4; it then regenerates data/ with
# "python3 -m titlestd generate-data" and commits the two together.

# How a title row words its role, beyond the three rows every role has under
# its own name: "name" is the role's name at another level form, "near" a
# wording that shares a word with the name, "far" one that shares none, which
# only its description can place. More weight on "far" means more titles the
# two models disagree on or cannot place.
WORDING_WEIGHTS = {"name": 5, "near": 3, "far": 2}

# How much a description says about the job, for the rows that have one:
# "close" restates the catalog definition, "partial" keeps half of it, "loose"
# describes the job in other words, "vague" says nothing about the job.
DESCRIPTION_WEIGHTS = {"close": 5, "partial": 3, "loose": 1, "vague": 1}

# The three compliance-flagged roles, in three different families (Q23). A
# title goes to a person when its role, or one of the ML model's top three, is
# flagged (1.7), so a role whose name shares its words with many others would
# send many more titles to people than a role with a name of its own.
FLAGGED_ROLE_IDS = ("FIN-04", "LEG-03", "OPS-06")

# --- Sizes: fixed by the record, not knobs (2.1, Q24, Q35) ----------------------

GENERATOR_SEED = 20260930  # the generator's own seed, never the run seed (Q06)
EMPLOYEES = 1500
ROWS_PER_ROLE = 8  # 48 roles x 8 rows + 16 special rows = 400 title rows
WAVE1_SIZE = 40  # "the 40 most widely held titles" (2.1): the default of wave1_size
HEADCOUNT_SKEW = 0.67  # the top 40 rows then hold about 40% of the employees
NO_DESCRIPTION_ROWS = 60  # 15% of the title rows have no description at all
MIXED_DESCRIPTION_ROWS = 12  # rows whose employees do not all have the same one (Q11)
NO_GRADE_EMPLOYEES = 20
NAMES_IN_DESCRIPTIONS = 6  # descriptions that name a person, to be masked
EMAILS_IN_DESCRIPTIONS = 6  # descriptions that hold an email address, to be masked

# --- The special rows, written out by hand --------------------------------------
# Each is (job title, family code, true role, employees, description), as recorded.

# Titles that hold a person's name: to a person and to no model (1.4, Q14).
PERSON_NAME_TITLES = (
    (
        "Cust Success Mgr (Tavi Quillfeather)",
        "SAL",
        "SAL-06",
        1,
        "Helps existing customers adopt products after the sale.",
    ),
    ("Paralegal - Isolde Fennimore", "LEG", "LEG-02", 1, ""),
    (
        "Accountant (cover for Liora Ashgrove)",
        "FIN",
        "FIN-02",
        1,
        "Keeps the general ledger while a colleague is on leave.",
    ),
    ("Software Engineer, Dariel's team", "TEC", "TEC-01", 1, "Writes application code."),
    ("Yarrow Stonebrook - Graphic Designer", "MKT", "MKT-03", 1, ""),
)
# Titles with a blank or unknown family code: to a person (2.4, Q26).
NO_FAMILY_TITLES = (
    (
        "Staff Accountant",
        "",
        "FIN-02",
        2,
        "Keeps the general ledger and prepares monthly accounts.",
    ),
    ("Office Manager", "ADM", "OPS-01", 2, "Runs the front office and orders supplies."),
    ("Data Entry Clerk", "TBD", "HRS-06", 1, ""),
)
# Titles worded like no role, with a description that says little: "no match"
# from the stand-in LLM (Q17).
NO_MATCH_TITLES = (
    ("Special Projects", "FIN", "FIN-01", 2, "Works on assignments set by the leadership team."),
    ("Culture Champion", "HRS", "HRS-06", 1, "Looks after our values and company events."),
    ("Growth Ninja", "MKT", "MKT-04", 1, "Tries new ideas to get the word out."),
    ("Problem Solver", "TEC", "TEC-04", 1, "The person to ask when something breaks."),
)
# Bare one-word titles with no description: "insufficient data" (Q17).
BARE_TITLES = (
    ("BDR", "SAL", "SAL-05", 2, ""),
    ("Analyst", "FIN", "FIN-01", 1, ""),
    ("Tech", "TEC", "TEC-04", 1, ""),
    ("Operator", "OPS", "OPS-03", 1, ""),
)
SPECIAL_TITLES = (*PERSON_NAME_TITLES, *NO_FAMILY_TITLES, *NO_MATCH_TITLES, *BARE_TITLES)

# --- The catalog and what the generator knows about each role -------------------

# fmt: off
# level, long word, short word (Q36)
_LADDER = (
    ("Junior", "Junior", "Jr"), ("Associate", "Associate", "Assoc"), ("Senior", "Senior", "Sr"),
    ("Lead", "Lead", "Lead"), ("Principal", "Principal", "Prin"),
)
# grade, level (Q36): grades 7 and 8 both read as Senior
_GRADES = (
    (1, "Junior"), (2, "Junior"), (3, "Junior"), (4, "Associate"), (5, "Associate"),
    (6, "Associate"), (7, "Senior"), (8, "Senior"), (9, "Lead"), (10, "Principal"),
)
# short form, long form. Every short word of the ladder and of a role's short
# name is here, so a Short Title cleans to the same row as its Long Title.
_ABBREVIATIONS = (
    ("sr", "senior"), ("jr", "junior"), ("assoc", "associate"), ("prin", "principal"),
    ("acct", "account"), ("mgr", "manager"), ("rep", "representative"), ("ops", "operations"),
    ("eng", "engineer"), ("biz", "business"), ("dev", "development"), ("cust", "customer"),
    ("fin", "financial"), ("spec", "specialist"), ("int", "internal"), ("ap", "accounts payable"),
    ("hr", "human resources"), ("comp", "compensation"), ("admin", "administrator"),
    ("coord", "coordinator"), ("sys", "systems"), ("tech", "technician"),
    ("qa", "quality assurance"), ("whse", "warehouse"), ("insp", "inspector"),
    ("mktg", "marketing"), ("mkt", "market"), ("comms", "communications"), ("corp", "corporate"),
    ("svc", "service"), ("ctr", "center"), ("supv", "supervisor"),
)
# fmt: on


class _Job(NamedTuple):
    """A role's catalog entry and what else the generator knows about it."""

    name: str
    short: str
    size: int  # how widely held its titles are: 3 common, 1 rare
    department: str
    # Four things the job does. The first two make the catalog definition.
    duties: tuple[str, str, str, str]
    near: tuple[str, str]  # wordings that share a word with the name
    far: str  # a wording that shares none


# fmt: off
# family code, family name, its short name, the six roles of the family. The
# family name and short name are those of the generic family role. A role is
# its name, short name, size, department, four duties, two near wordings and
# one far wording, as _Job has them.
_FAMILIES = (
    ("SAL", "Sales", "Sales", (
        _Job("Account Manager", "Acct Mgr", 3, "Enterprise Sales",
             ("manages a portfolio of customer accounts", "renews customer contracts",
              "acts as the main contact for assigned clients",
              "grows revenue from existing business"),
             ("Key Account Manager", "Account Executive"), "Client Partner"),
        _Job("Sales Representative", "Sales Rep", 3, "Inside Sales",
             ("sells products to new customers", "closes deals",
              "works an assigned territory by phone and in person", "meets a monthly quota"),
             ("Inside Sales Representative", "Sales Consultant"), "Territory Executive"),
        _Job("Sales Operations Analyst", "Sales Ops Analyst", 1, "Sales Operations",
             ("maintains sales forecasts and territories", "prepares pipeline reports",
              "keeps the customer database tidy", "tracks quota attainment for leadership"),
             ("Sales Analyst", "Revenue Operations Analyst"), "CRM Administrator"),
        _Job("Sales Engineer", "Sales Eng", 2, "Enterprise Sales",
             ("explains the technical side of products during a sale",
              "runs product demonstrations", "answers detailed questions from prospects",
              "builds proofs of concept"),
             ("Presales Engineer", "Solutions Engineer"), "Technical Presales Consultant"),
        _Job("Business Development Representative", "Biz Dev Rep", 2, "Inside Sales",
             ("finds new sales leads by outreach", "qualifies leads for the sales team",
              "books first meetings with prospects",
              "makes cold calls and sends introductory emails"),
             ("Business Development Specialist", "Sales Development Representative"),
             "Outbound Prospector"),
        _Job("Customer Success Manager", "Cust Success Mgr", 2, "Customer Success",
             ("helps existing customers adopt products", "keeps customers satisfied after the sale",
              "runs onboarding and regular check-ins", "works to reduce churn"),
             ("Customer Success Specialist", "Client Success Manager"),
             "Client Onboarding Advisor"),
    )),
    ("FIN", "Finance", "Finance", (
        _Job("Financial Analyst", "Fin Analyst", 2, "Corporate Finance",
             ("builds budgets and forecasts", "explains variances against plan",
              "models revenue and cost scenarios", "prepares the monthly management pack"),
             ("Finance Analyst", "Financial Planning Analyst"), "Budget Planner"),
        _Job("Accountant", "Accountant", 3, "Accounting",
             ("keeps the general ledger", "prepares monthly accounts",
              "reconciles balance sheet items", "books journal entries at period end"),
             ("Staff Accountant", "General Ledger Accountant"), "Bookkeeper"),
        _Job("Payroll Specialist", "Payroll Spec", 2, "Accounting",
             ("runs payroll", "resolves pay queries", "checks timesheets and deductions",
              "files wage tax returns"),
             ("Payroll Administrator", "Payroll Clerk"), "Salary Processing Clerk"),
        _Job("Internal Auditor", "Int Auditor", 1, "Internal Audit",
             ("tests internal controls", "reports findings to the audit committee",
              "plans audit fieldwork", "follows up on agreed actions"),
             ("Internal Controls Auditor", "Internal Audit Analyst"),
             "Controls Assurance Specialist"),
        _Job("Accounts Payable Clerk", "AP Clerk", 2, "Accounting",
             ("processes supplier invoices", "schedules payments",
              "matches invoices to purchase orders", "answers vendor questions about remittances"),
             ("Accounts Payable Specialist", "Accounts Payable Administrator"),
             "Invoice Processor"),
        _Job("Treasury Analyst", "Treasury Analyst", 1, "Treasury",
             ("manages cash positions and bank accounts", "arranges short-term funding",
              "forecasts liquidity", "executes currency trades"),
             ("Treasury Specialist", "Cash Management Analyst"), "Cash Manager"),
    )),
    ("HRS", "Human Resources", "HR", (
        _Job("Recruiter", "Recruiter", 2, "Talent Acquisition",
             ("sources candidates", "fills open positions",
              "screens applications and schedules interviews", "extends job offers"),
             ("Technical Recruiter", "Corporate Recruiter"), "Talent Acquisition Partner"),
        _Job("Human Resources Business Partner", "HR Business Partner", 2, "People Operations",
             ("advises managers on staffing and performance", "handles employee relations cases",
              "supports reorganisations", "coaches leaders on people matters"),
             ("Human Resources Partner", "People Business Partner"), "Employee Relations Advisor"),
        _Job("Compensation Analyst", "Comp Analyst", 1, "Total Rewards",
             ("benchmarks pay against the market", "maintains salary bands",
              "prepares the annual bonus cycle", "evaluates jobs for grading"),
             ("Compensation Specialist", "Rewards Analyst"), "Pay Benchmarking Consultant"),
        _Job("Benefits Administrator", "Benefits Admin", 1, "Total Rewards",
             ("enrols employees in health and retirement plans", "answers benefits questions",
              "works with insurance providers", "processes leave requests"),
             ("Benefits Specialist", "Benefits Coordinator"), "Pensions Clerk"),
        _Job("Training Specialist", "Training Spec", 2, "Learning",
             ("designs training courses", "delivers courses to employees",
              "maintains the learning platform", "measures course completion"),
             ("Training Coordinator", "Training Facilitator"), "Learning Designer"),
        _Job("Human Resources Coordinator", "HR Coord", 2, "People Operations",
             ("keeps employee records", "schedules onboarding for new hires",
              "prepares employment letters", "answers routine policy questions"),
             ("Human Resources Administrator", "Human Resources Assistant"),
             "People Operations Generalist"),
    )),
    ("TEC", "Technology", "Technology", (
        _Job("Software Engineer", "Software Eng", 3, "Engineering",
             ("designs and writes application code", "reviews changes before release",
              "fixes defects found in production", "builds services and interfaces"),
             ("Software Developer", "Backend Engineer"), "Programmer"),
        _Job("Data Analyst", "Data Analyst", 2, "Data",
             ("queries data and builds dashboards", "explains trends to the business",
              "cleans and joins data sets", "writes reports for decision makers"),
             ("Business Data Analyst", "Data Specialist"), "Business Intelligence Developer"),
        _Job("Systems Administrator", "Sys Admin", 2, "Infrastructure",
             ("runs servers and networks", "manages user accounts",
              "applies patches and monitors uptime", "automates backups"),
             ("Systems Engineer", "Network Administrator"), "Infrastructure Engineer"),
        _Job("Help Desk Technician", "Help Desk Tech", 2, "Service Desk",
             ("resolves hardware and software problems for employees", "sets up laptops and phones",
              "logs tickets and escalates outages", "resets passwords"),
             ("Help Desk Analyst", "Desktop Support Technician"), "IT Support Specialist"),
        _Job("Security Analyst", "Security Analyst", 1, "Information Security",
             ("monitors systems for threats", "investigates security incidents",
              "reviews access rights", "runs vulnerability scans"),
             ("Information Security Analyst", "Security Engineer"), "Cyber Threat Specialist"),
        _Job("Quality Assurance Engineer", "QA Eng", 2, "Engineering",
             ("writes test plans", "verifies releases before launch", "automates regression tests",
              "tracks defects to closure"),
             ("Quality Assurance Analyst", "Test Engineer"), "Software Tester"),
    )),
    ("OPS", "Operations", "Ops", (
        _Job("Operations Manager", "Ops Mgr", 2, "Site Operations",
             ("runs daily site operations", "improves processes", "sets shift plans and staffing",
              "tracks output against targets"),
             ("Site Operations Manager", "Operations Supervisor"), "Shift Supervisor"),
        _Job("Supply Chain Analyst", "Supply Chain Analyst", 1, "Supply Chain",
             ("plans inventory levels", "analyses supplier performance",
              "forecasts demand by product", "reports on stock turns"),
             ("Supply Chain Planner", "Supply Analyst"), "Demand Planner"),
        _Job("Warehouse Operator", "Whse Operator", 3, "Distribution",
             ("picks and packs orders", "loads goods for shipment", "drives a forklift",
              "counts stock during cycle checks"),
             ("Warehouse Worker", "Warehouse Picker"), "Forklift Driver"),
        _Job("Logistics Coordinator", "Logistics Coord", 2, "Distribution",
             ("schedules shipments", "tracks deliveries with carriers",
              "prepares customs paperwork", "resolves delays with transport partners"),
             ("Logistics Specialist", "Transport Coordinator"), "Dispatcher"),
        _Job("Procurement Specialist", "Procurement Spec", 2, "Purchasing",
             ("sources suppliers and negotiates prices", "issues purchase orders", "runs tenders",
              "reviews supplier agreements"),
             ("Procurement Officer", "Purchasing Specialist"), "Buyer"),
        _Job("Safety Inspector", "Safety Insp", 1, "Health and Safety",
             ("inspects sites for hazards", "enforces safety regulations", "investigates accidents",
              "trains staff on safe working"),
             ("Health and Safety Inspector", "Safety Officer"), "Site Hazard Advisor"),
    )),
    ("MKT", "Marketing", "Mktg", (
        _Job("Marketing Manager", "Mktg Mgr", 2, "Brand",
             ("plans campaigns", "manages the marketing budget",
              "sets brand priorities with agencies", "presents results to leadership"),
             ("Product Marketing Manager", "Marketing Campaign Manager"), "Brand Director"),
        _Job("Content Writer", "Content Writer", 2, "Digital Marketing",
             ("writes articles and web pages", "edits product copy", "keeps a publishing calendar",
              "interviews experts for stories"),
             ("Content Specialist", "Technical Writer"), "Copywriter"),
        _Job("Graphic Designer", "Graphic Designer", 2, "Brand",
             ("creates visual designs for campaigns", "maintains brand materials",
              "prepares artwork for print and web", "lays out presentations"),
             ("Visual Designer", "Graphic Artist"), "Creative Artworker"),
        _Job("Digital Marketing Specialist", "Digital Mktg Spec", 2, "Digital Marketing",
             ("runs paid search and email campaigns", "manages social media channels",
              "optimises landing pages", "tracks clicks and conversions"),
             ("Digital Marketing Coordinator", "Email Marketing Specialist"),
             "Social Media Coordinator"),
        _Job("Market Research Analyst", "Mkt Research Analyst", 1, "Market Insights",
             ("surveys customers", "analyses market trends and competitors", "runs focus groups",
              "sizes new segments"),
             ("Market Analyst", "Research Analyst"), "Consumer Insights Specialist"),
        _Job("Communications Specialist", "Comms Spec", 1, "Communications",
             ("prepares press releases", "writes internal announcements", "briefs spokespeople",
              "edits the staff newsletter"),
             ("Communications Coordinator", "Internal Communications Specialist"), "Press Officer"),
    )),
    ("LEG", "Legal", "Legal", (
        _Job("Corporate Counsel", "Corp Counsel", 1, "Legal Affairs",
             ("advises the business on contracts and disputes", "interprets corporate law",
              "negotiates terms with outside lawyers", "reviews board papers"),
             ("Legal Counsel", "Commercial Counsel"), "In-House Lawyer"),
        _Job("Paralegal", "Paralegal", 2, "Legal Affairs",
             ("prepares legal documents", "supports counsel on filings",
              "keeps case files in order", "researches precedents"),
             ("Litigation Paralegal", "Corporate Paralegal"), "Legal Assistant"),
        _Job("Compliance Officer", "Compliance Officer", 1, "Compliance",
             ("monitors adherence to regulations", "reports breaches to regulators",
              "trains staff on the code of conduct", "reviews policies each year"),
             ("Compliance Analyst", "Regulatory Compliance Officer"), "Regulatory Affairs Advisor"),
        _Job("Contracts Administrator", "Contracts Admin", 2, "Contracts",
             ("drafts commercial contracts", "tracks renewal dates", "files signed agreements",
              "checks terms against templates"),
             ("Contracts Specialist", "Contracts Manager"), "Agreements Coordinator"),
        _Job("Privacy Analyst", "Privacy Analyst", 1, "Privacy Office",
             ("assesses how personal data is handled", "answers data subject requests",
              "keeps the record of processing", "checks new systems for privacy gaps"),
             ("Data Privacy Analyst", "Privacy Specialist"), "Data Protection Advisor"),
        _Job("Risk Analyst", "Risk Analyst", 1, "Compliance",
             ("assesses operational and regulatory risks", "maintains the risk register",
              "tests business continuity plans", "reports exposures to the board"),
             ("Operational Risk Analyst", "Risk Specialist"), "Business Continuity Planner"),
    )),
    ("CUS", "Customer Service", "Cust Svc", (
        _Job("Customer Service Representative", "Cust Svc Rep", 3, "Contact Center",
             ("answers customer questions by phone, email and chat", "logs each contact",
              "takes orders and updates details", "passes complex cases to specialists"),
             ("Customer Service Agent", "Customer Care Representative"), "Contact Center Agent"),
        _Job("Product Support Specialist", "Product Support Spec", 2, "Product Support",
             ("troubleshoots product problems reported by customers", "documents fixes",
              "reproduces faults and raises defects", "guides users through installation"),
             ("Technical Support Specialist", "Product Support Engineer"),
             "Application Support Analyst"),
        _Job("Call Center Supervisor", "Call Ctr Supv", 2, "Contact Center",
             ("supervises a team of agents", "monitors call quality", "sets rosters and breaks",
              "coaches new starters"),
             ("Call Center Manager", "Contact Center Supervisor"), "Customer Care Team Leader"),
        _Job("Billing Specialist", "Billing Spec", 2, "Billing Services",
             ("resolves invoice disputes", "adjusts customer bills", "issues credit notes",
              "explains charges to callers"),
             ("Billing Coordinator", "Billing Analyst"), "Customer Accounts Clerk"),
        _Job("Customer Experience Analyst", "Cust Experience Analyst", 1, "Service Quality",
             ("analyses satisfaction surveys", "reports complaint trends", "maps customer journeys",
              "recommends service improvements"),
             ("Customer Insights Analyst", "Customer Experience Specialist"),
             "Satisfaction Survey Coordinator"),
        _Job("Returns Coordinator", "Returns Coord", 2, "Billing Services",
             ("processes product returns", "arranges refunds and exchanges",
              "checks returned goods", "books collections with couriers"),
             ("Returns Specialist", "Returns Clerk"), "Refunds Administrator"),
    )),
)
# fmt: on
_JOBS = {
    f"{code}-{number:02d}": job
    for code, _, _, jobs in _FAMILIES
    for number, job in enumerate(jobs, start=1)
}

_DEPARTMENTS = {code: sorted({job.department for job in jobs}) for code, _, _, jobs in _FAMILIES}

# Descriptions that say nothing about the job.
_VAGUE = (
    "Supports the group and takes on other tasks as assigned.",
    "Works with colleagues across the business to meet agreed goals.",
    "Carries out the duties of the post as directed.",
    "Contributes to departmental objectives and ad hoc projects.",
    "Full-time position based at the main office.",
    "See the attached profile for details of this post.",
    "Duties vary with the needs of the department.",
    "Position within the wider group, working on shared priorities.",
)


class _Form(NamedTuple):
    """One way a title names a level (2.3: "level words in several forms")."""

    level: str | None  # the ladder level the form stands for; it sets the grades
    suffix: str | None  # what is written after the title; None puts the level's word in front
    weight: int  # how common the form is, among titles and among employees


_PLAIN = _Form(None, "", 6)
_SENIOR = _Form("Senior", None, 3)
_TWO = _Form("Associate", " II", 2)
_FORMS = (
    _PLAIN,
    _SENIOR,
    _TWO,
    _Form("Junior", None, 2),
    _Form("Associate", None, 1),
    _Form("Lead", None, 1),
    _Form("Principal", None, 1),
    _Form("Junior", " I", 1),
    _Form("Senior", " III", 2),
    _Form("Junior", " 1", 1),
    _Form("Associate", " 2", 1),
    _Form("Senior", " 3", 1),
    _Form("Senior", ", Senior", 1),
)
_FORM_WEIGHTS = {form: form.weight for form in _FORMS}
# How widely held a wording is, beside the size of its role and its level form.
_WORDING_POPULARITY = {"name": 1.0, "near": 0.5, "far": 0.35}

# How a spelling is written: which words, which case, what follows the title.
_STYLE_WEIGHTS = {"long": 5, "short": 3, "mixed": 2}
_CASE_WEIGHTS = {"as written": 7, "upper": 2, "lower": 1}
_DOTTED = 0.25  # share of abbreviated spellings that put a dot after a short form
_ENDINGS = {
    "": 14,
    " - {place}": 2,
    " ({place})": 1,
    ", {place}": 1,
    " {number}": 1,
    " ({number})": 1,
    " - {place} {number}": 1,
    " {number} ({place})": 1,
}
_OTHER_DEPARTMENT = 0.2  # share of spellings recorded under another department of the family
_OFF_GRADE = 1 / 6  # share of employees whose grade is not one of the title's level
_ATTEMPTS = 200  # draws before the generator gives up on finding something new

_STREAM = "mockdata"  # the generator's own stream name; no run draws from it

# Long forms and what replaces them in a short spelling, the longest first.
_SHORTENINGS = tuple(
    (
        re.compile(rf"\b{re.escape(long)}\b", re.IGNORECASE),
        short.upper() if " " in long else short.capitalize(),
    )
    for short, long in sorted(_ABBREVIATIONS, key=lambda pair: -len(pair[1]))
)
_SHORT_FORM = re.compile(
    rf"\b({'|'.join(short for short, _ in _ABBREVIATIONS)})\b(?!\.)", re.IGNORECASE
)


class MockDataError(TitlestdError):
    """The knobs or the tables of the generator give data that breaks a rule it
    guarantees. The message says which."""


class _Draw:
    """The seeded draws of one item of the data (Q06).

    Each item has its own generator, keyed on what the item is, so changing
    one thing never moves the draws of another. Only random() and shuffle()
    are used: the scaffold pins both on every supported Python.
    """

    def __init__(self, *key: str) -> None:
        self._rng = seeding.rng(GENERATOR_SEED, _STREAM, *key)

    def unit(self) -> float:
        return self._rng.random()

    def chance(self, share: float) -> bool:
        return self._rng.random() < share

    def pick[T](self, options: Sequence[T]) -> T:
        return options[int(self._rng.random() * len(options))]

    def weighted[T](self, weights: Mapping[T, float]) -> T:
        *options, last = weights
        point = self._rng.random() * sum(weights.values())
        for option in options:
            point -= weights[option]
            if point < 0:
                return option
        return last

    def shuffle(self, items: list) -> None:
        self._rng.shuffle(items)


def _lowest[T](
    items: Iterable[T], count: int, stream: str, key: Callable[[T], tuple[str, ...]]
) -> list[T]:
    """The count items with the lowest seeded draws: an exact number of them,
    and the same ones whatever else about the data changes."""

    def draw(item: T) -> tuple[float, tuple[str, ...]]:
        return seeding.unit(GENERATOR_SEED, _STREAM, stream, *key(item)), key(item)

    return sorted(items, key=draw)[:count]


@dataclass(slots=True)
class _Title:
    """One title row as planned: what the prepare step must read back (Q09)."""

    family_code: str  # as recorded on its employees
    role: CatalogRole  # its one true role
    long: str  # the title written out
    short: str  # the same title in short forms
    cleaned: str  # the cleaned title the row is keyed on
    level: str | None = None  # the ladder level the title names
    kind: str = "special"  # how it words its role: a key of WORDING_WEIGHTS, or "special"
    words: int = 0  # how many words the wording has, level form aside
    popularity: float = 0.0
    headcount: int = 0
    spellings: tuple[str, ...] = ()  # the job titles as recorded, the commonest first
    description: str = ""
    # What every third employee has instead, where the row is a mixed one (Q11).
    other_description: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.family_code.strip(), self.cleaned


def _key(title: _Title) -> tuple[str, str]:
    return title.key


# --- The catalog ------------------------------------------------------------------


def _sentence(first: str, second: str) -> str:
    return f"{first[0].upper()}{first[1:]} and {second}."


def _catalog() -> Catalog:
    roles: list[CatalogRole] = []
    for code, name, short, jobs in _FAMILIES:
        for number, job in enumerate(jobs, start=1):
            role_id = f"{code}-{number:02d}"
            definition = _sentence(*job.duties[:2])
            flagged = role_id in FLAGGED_ROLE_IDS
            roles.append(CatalogRole(role_id, code, job.name, job.short, definition, flagged))
        generic = f"Generic role for the {name} family."
        roles.append(CatalogRole(f"{code}-GEN", code, name, short, generic, is_generic=True))
    flagged_families = {role.family_code for role in roles if role.compliance_flag}
    if len(FLAGGED_ROLE_IDS) != 3 or len(flagged_families) != 3:
        raise MockDataError(
            "FLAGGED_ROLE_IDS must name three roles of the catalog in three different "
            f"families (Q23); it is {FLAGGED_ROLE_IDS}"
        )
    return Catalog(
        roles=tuple(roles),
        ladder=tuple(LadderLevel(*level) for level in _LADDER),
        grade_levels=tuple(GradeLevel(grade, level) for grade, level in _GRADES),
    )


# --- The title rows ---------------------------------------------------------------


def _abbreviate(text: str, draw: _Draw | None = None) -> str:
    """The text with long forms replaced by their short forms: all of them, or,
    given a draw, about half of them."""
    for long_form, short_form in _SHORTENINGS:
        if draw is None or draw.chance(0.5):
            text = long_form.sub(short_form, text)
    return text


def _title(
    catalog: Catalog,
    abbreviations: Sequence[Abbreviation],
    role: CatalogRole,
    wording: str,
    form: _Form,
) -> _Title:
    """A wording of a role written at a level form: the naming rule of the
    catalog (1.3), applied to the wording in place of the role's name."""
    job = _JOBS[role.role_id]
    kind = "name" if wording == job.name else "near" if wording in job.near else "far"
    worded = role
    if kind != "name":
        worded = dataclasses.replace(role, role_name=wording, short_name=_abbreviate(wording))
    # A level word in front comes from a grade of that level, as for an employee.
    in_front = form.suffix is None
    grade = next(
        (row.grade for row in catalog.grade_levels if in_front and row.level == form.level), None
    )
    named = naming.standard_title(catalog, worded, grade)
    suffix = form.suffix or ""
    long, short = named.long_title + suffix, named.short_title + _abbreviate(suffix)
    cleaned = prepare.clean_title(long, abbreviations)
    if cleaned.has_person_name or prepare.clean_title(short, abbreviations) != cleaned:
        raise MockDataError(
            f"{long!r} and {short!r} must clean to one title with no name in it; "
            "see _ABBREVIATIONS and titlestd.reference"
        )
    return _Title(
        family_code=role.family_code,
        role=role,
        long=long,
        short=short,
        cleaned=cleaned.text,
        level=form.level,
        kind=kind,
        words=len(wording.split()),
        popularity=job.size * _WORDING_POPULARITY[kind] * form.weight,
    )


def _role_titles(
    catalog: Catalog, abbreviations: Sequence[Abbreviation], role: CatalogRole
) -> list[_Title]:
    """The ROWS_PER_ROLE title rows of one role: its name plain, with "Senior"
    and with "II", then wordings and level forms drawn by their weights."""
    job = _JOBS[role.role_id]
    wordings = {"name": (job.name,), "near": job.near, "far": (job.far,)}
    draw = _Draw("titles", role.role_id)
    chosen = [(job.name, form) for form in (_PLAIN, _SENIOR, _TWO)]
    attempts = 0
    while len(chosen) < ROWS_PER_ROLE:
        attempts += 1
        if attempts > _ATTEMPTS:
            raise MockDataError(
                f"WORDING_WEIGHTS {WORDING_WEIGHTS} leaves role {role.role_id} with fewer "
                f"than {ROWS_PER_ROLE} titles"
            )
        pair = (draw.pick(wordings[draw.weighted(WORDING_WEIGHTS)]), draw.weighted(_FORM_WEIGHTS))
        if pair not in chosen:
            chosen.append(pair)
    return [_title(catalog, abbreviations, role, wording, form) for wording, form in chosen]


def _headcounts(rows: int, employees: int) -> list[int]:
    """How many employees the row of each rank holds, the most widely held first.

    A row's share falls as rank ** -HEADCOUNT_SKEW. Shares are rounded down and
    the employees left over go to the rows that lost the largest fraction. A
    row past the WAVE1_SIZE-th that would tie with it then gives one employee
    to a row of the Wave 1 list, so the list never rests on a tie-break (Q24).
    """
    weights = [rank**-HEADCOUNT_SKEW for rank in range(1, rows + 1)]
    shares = [employees * weight / sum(weights) for weight in weights]
    counts = [int(share) for share in shares]
    by_lost_fraction = sorted(range(rows), key=lambda at: counts[at] - shares[at])
    for at in by_lost_fraction[: employees - sum(counts)]:
        counts[at] += 1
    cut = counts[WAVE1_SIZE - 1]
    tied = [at for at in range(WAVE1_SIZE, rows) if counts[at] == cut]
    for top, at in enumerate(tied):
        counts[at] -= 1
        counts[top % WAVE1_SIZE] += 1
    if min(counts) < 1:
        raise MockDataError(f"HEADCOUNT_SKEW {HEADCOUNT_SKEW} leaves a title row with no employee")
    return counts


def _special_titles(catalog: Catalog, abbreviations: Sequence[Abbreviation]) -> list[_Title]:
    titles = []
    for job_title, family_code, role_id, headcount, description in SPECIAL_TITLES:
        role = naming.get_role(catalog, role_id)
        if role is None:
            raise MockDataError(f"{job_title!r} names role {role_id}, which is not in the catalog")
        titles.append(
            _Title(
                family_code=family_code,
                role=role,
                long=job_title,
                short=job_title,
                cleaned=prepare.clean_title(job_title, abbreviations).text,
                headcount=headcount,
                spellings=(job_title,),
                description=description,
            )
        )
    return titles


def _ending(draw: _Draw) -> str:
    """What follows a title: nothing, a location, a requisition number or both."""
    place = draw.pick(reference.LOCATIONS)
    place = place.upper() if len(place) == 3 else place.capitalize()
    number = draw.pick(("REQ-", "#")) + str(1000 + int(draw.unit() * 999000))
    return draw.weighted(_ENDINGS).format(place=place, number=number)


def _spelling(title: _Title, draw: _Draw) -> str:
    """One way the title is recorded (2.3: "messy on purpose")."""
    style = draw.weighted(_STYLE_WEIGHTS)
    text = {"long": title.long, "short": title.short}.get(style) or _abbreviate(title.long, draw)
    if text != title.long and draw.chance(_DOTTED):
        text = _SHORT_FORM.sub(r"\1.", text)
    case = draw.weighted(_CASE_WEIGHTS)
    text = {"upper": text.upper(), "lower": text.lower()}.get(case, text)
    return text + _ending(draw)


def _spell(title: _Title, abbreviations: Sequence[Abbreviation]) -> None:
    """Give the row its spellings: more of them the more widely held it is."""
    draw = _Draw("spellings", *title.key)
    held = title.headcount
    wanted = 1 + (held >= 4) + (held >= 10) + (held >= 25)
    if held in (2, 3) and draw.chance(0.5):
        wanted = 2
    spellings: list[str] = []
    for _ in range(_ATTEMPTS):
        if len(spellings) == wanted:
            break
        spelling = _spelling(title, draw)
        if prepare.clean_title(spelling, abbreviations).text != title.cleaned:
            raise MockDataError(f"{spelling!r} does not clean to {title.cleaned!r}")
        if spelling not in spellings:
            spellings.append(spelling)
    title.spellings = tuple(spellings)


def _titles(catalog: Catalog, abbreviations: Sequence[Abbreviation]) -> list[_Title]:
    """Every title row with its headcount and spellings, the Wave 1 list first."""
    titles = [
        title
        for role in naming.model_visible_roles(catalog)
        for title in _role_titles(catalog, abbreviations, role)
    ]
    for title in titles:
        title.popularity *= 0.5 + _Draw("popularity", *title.key).unit()
    titles.sort(key=lambda title: (-title.popularity, title.key))
    special = _special_titles(catalog, abbreviations)
    held_by_special = sum(title.headcount for title in special)
    for title, headcount in zip(
        titles, _headcounts(len(titles), EMPLOYEES - held_by_special), strict=True
    ):
        title.headcount = headcount
        _spell(title, abbreviations)
    titles += special
    keys = [title.key for title in titles]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise MockDataError(f"two titles of one family clean to the same row: {repeated}")
    return titles


# --- Descriptions (Q11, Q35) ------------------------------------------------------


def _description(job: _Job, kind: str, draw: _Draw) -> str:
    if kind == "vague":
        return draw.pick(_VAGUE)
    defining, other = list(job.duties[:2]), list(job.duties[2:])
    draw.shuffle(defining)
    draw.shuffle(other)
    return _sentence(*{"close": defining, "partial": [defining[0], other[0]], "loose": other}[kind])


def _describe(titles: Sequence[_Title]) -> None:
    """Give the ordinary rows their descriptions; the special rows have theirs."""
    ordinary = [title for title in titles if title.kind != "special"]
    silent = NO_DESCRIPTION_ROWS - sum(
        1 for title in titles if title.kind == "special" and not title.description
    )
    # A wording of one word keeps its description: a bare title with none is
    # "insufficient data" for the stand-in, and BARE_TITLES are those (Q17).
    wordy = [title for title in ordinary if title.words >= 2]
    without = {title.key for title in _lowest(wordy, silent, "no description", _key)}
    described = [title for title in ordinary if title.key not in without]
    for title in described:
        draw = _Draw("description", *title.key)
        kind = draw.weighted(DESCRIPTION_WEIGHTS)
        title.description = _description(_JOBS[title.role.role_id], kind, draw)

    contacts = _lowest(described, NAMES_IN_DESCRIPTIONS + EMAILS_IN_DESCRIPTIONS, "contact", _key)
    for at, title in enumerate(contacts):
        draw = _Draw("contact", *title.key)
        given, surname = draw.pick(reference.GIVEN_NAMES), draw.pick(reference.SURNAMES)
        if at < NAMES_IN_DESCRIPTIONS:
            title.description += f" Reports to {given.capitalize()} {surname.capitalize()}."
        else:
            title.description += f" Questions to {given}.{surname}@{reference.EMAIL_DOMAIN}."

    shared = [title for title in described if title.headcount >= 2]
    for at, title in enumerate(_lowest(shared, MIXED_DESCRIPTION_ROWS, "mixed", _key)):
        # No description, or one that _description never writes: a duty that
        # is not in the definition, then one that is.
        duties = _JOBS[title.role.role_id].duties
        title.other_description = "" if at % 2 else _sentence(duties[3], duties[0])


# --- Employees, the seed look-up and the truth ------------------------------------


def _split(headcount: int, parts: int) -> list[int]:
    """The employees of a row over its spellings: the first holds the most."""
    rest = [max(1, headcount >> at) for at in range(2, parts + 1)]
    return [headcount - sum(rest), *rest]


def _employees(titles: Sequence[_Title], catalog: Catalog) -> tuple[Employee, ...]:
    every_grade = [row.grade for row in catalog.grade_levels]
    records: list[Employee] = []
    for title in titles:
        draw = _Draw("employees", *title.key)
        job = _JOBS[title.role.role_id]
        grades = [row.grade for row in catalog.grade_levels if row.level == title.level]
        minority = 1 if title.headcount == 2 else 2
        position = 0
        for spelling, held in zip(
            title.spellings, _split(title.headcount, len(title.spellings)), strict=True
        ):
            department = job.department
            if draw.chance(_OTHER_DEPARTMENT):
                department = draw.pick(_DEPARTMENTS[title.role.family_code])
            for _ in range(held):
                grade = draw.pick(grades or every_grade)
                if draw.chance(_OFF_GRADE):
                    grade = draw.pick(every_grade)
                description = title.description
                if title.other_description is not None and position % 3 == minority:
                    description = title.other_description
                records.append(
                    Employee("", spelling, title.family_code, department, grade, description)
                )
                position += 1
    _Draw("employee order").shuffle(records)
    employees = [
        dataclasses.replace(record, employee_id=f"E{number:04d}")
        for number, record in enumerate(records, start=1)
    ]
    ungraded = _lowest(
        employees, NO_GRADE_EMPLOYEES, "no grade", lambda employee: (employee.employee_id,)
    )
    without_grade = {employee.employee_id for employee in ungraded}
    return tuple(
        dataclasses.replace(employee, grade=None)
        if employee.employee_id in without_grade
        else employee
        for employee in employees
    )


def _seed_lookup(titles: Sequence[_Title]) -> tuple[SeedLookupRow, ...]:
    """One long-tail row of each family, already decided by a person (Q13, Q35)."""
    long_tail = [title for title in titles[WAVE1_SIZE:] if title.kind != "special"]
    seeds = []
    for code, _, _, _ in _FAMILIES:
        (seed,) = _lowest(
            [title for title in long_tail if title.family_code == code], 1, "seed", _key
        )
        seeds.append(SeedLookupRow(seed.spellings[0], seed.family_code, seed.role.role_id))
    return tuple(seeds)


def _check(data: InputData, truth: Sequence[people.TruthRow], titles: Sequence[_Title]) -> None:
    """What the generator guarantees (Q09, Q14, Q24, Q26), checked on what it is
    about to write, with the real readers' checks and the real prepare step."""
    datafiles.validate_inputs(data)
    people.validate_truth(truth, data.catalog)
    rows = prepare.prepare(data.employees, data.abbreviations, naming.family_codes(data.catalog))
    if {row.key: row.headcount for row in rows} != {title.key: title.headcount for title in titles}:
        raise MockDataError("the prepare step does not read back the planned title rows")
    named = {raw for row in rows if row.has_person_name for raw in row.raw_titles}
    unfiled = {raw for row in rows if not row.family_usable for raw in row.raw_titles}
    if named != {title for title, *_ in PERSON_NAME_TITLES}:
        raise MockDataError(f"the titles that hold a person's name are {sorted(named)}")
    if unfiled != {title for title, *_ in NO_FAMILY_TITLES}:
        raise MockDataError(f"the titles with no usable family code are {sorted(unfiled)}")
    true_role = {(row.job_title, row.job_family_code.strip()): row.true_role_id for row in truth}
    for row in rows:
        if len({true_role[(raw, row.key.family_code)] for raw in row.raw_titles}) != 1:
            raise MockDataError(f"the spellings of {row.key} have more than one true role")
    held = sorted((row.headcount for row in rows), reverse=True)
    if len(data.employees) != EMPLOYEES or held[WAVE1_SIZE - 1] == held[WAVE1_SIZE]:
        raise MockDataError("the headcounts do not add up, or the Wave 1 list ends in a tie")


def build() -> tuple[InputData, tuple[people.TruthRow, ...]]:
    """The content of the six input files and of the truth file, the same on
    every call, on every supported Python and in every process."""
    for name, weights, kinds in (
        ("WORDING_WEIGHTS", WORDING_WEIGHTS, _WORDING_POPULARITY),
        ("DESCRIPTION_WEIGHTS", DESCRIPTION_WEIGHTS, ("close", "partial", "loose", "vague")),
    ):
        shares = list(weights.values())
        if not set(weights) <= set(kinds) or min(shares, default=-1) < 0 or sum(shares) <= 0:
            raise MockDataError(f"{name} must give weights, none negative, to some of {[*kinds]}")
    catalog = _catalog()
    abbreviations = tuple(Abbreviation(short, long) for short, long in _ABBREVIATIONS)
    titles = _titles(catalog, abbreviations)
    _describe(titles)
    data = InputData(
        catalog=catalog,
        abbreviations=abbreviations,
        employees=_employees(titles, catalog),
        seed_lookup=_seed_lookup(titles),
    )
    truth = tuple(
        people.TruthRow(spelling, title.family_code, title.role.role_id)
        for title in sorted(titles, key=_key)
        for spelling in sorted(title.spellings)
    )
    _check(data, truth, titles)
    return data, truth


def generate(data_dir: Path) -> None:
    """Write the seven data files into data_dir, the same bytes every time."""
    data, truth = build()
    datafiles.write_inputs(data_dir, data)
    people.write_truth(data_dir, truth)


def generate_command(settings: Settings) -> int:
    """The generate-data command: generate(settings.data_dir). Returns the exit status."""
    generate(settings.data_dir)
    print(f"wrote the seven data files of the mock data set to {settings.data_dir}")
    return 0
