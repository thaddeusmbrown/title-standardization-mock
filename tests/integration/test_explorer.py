"""The architecture page on the state of a real run of the mini data set.

Owner: feat/architecture-page (after the build).
Governed by: DESIGN.md 1.10 (the waves), 2.8 (the two browser libraries);
docs/build-decisions.md Q06 (the same bytes twice), Q40 (the answer labels),
S08 (the schema of the state file is titlestd/store.py's).

One run of titlestd.pipeline on tests/fixtures/mini, written under a
temporary folder, feeds every test. The WSGI application of
titlestd/explorer.py is called directly: no server is started.

Decided here, where the record is silent:

- "Every route" is every address of explorer.PAGES and explorer.STATIC_FILES,
  and the pages link to exactly those: following every href and src from the
  overview reaches all of them and nothing else.
- "Hostile strings in every text column" is two copies of the state: one with
  a hostile row added to every table, its every text column hostile and the
  run's rows left as they were, so the counts still meet real values; and one
  with the hostile text added to every text value of every row. Every column
  with the declared type TEXT counts, and so does the column with no declared
  type (model_answers.number), which may hold a model's text. No response,
  body or header, may hold the marker the hostile strings carry. The one
  value put back clean is the llm setting: the page refuses a state without
  a free one, so a hostile one reaches no page either.
- "The state file unchanged" is its bytes and the listing of its folder, after
  every route was asked twice, also with the file and its folder made
  read-only.
- The counts are held to the run itself (RunResult), so a change to the
  schema of the state file shows here (S08). Where RunResult holds no such
  count (the kinds and routes of the answers, the occasions of people's
  decisions, the stage a title was learned at), each set of counts that
  splits a total is held to add up to it, which a count that read a wrong
  value would break.
- "A run that did not finish" is a copy of the state with the tables that
  store.save_run writes at the end of a run emptied: what a run that stopped
  after its last scoring leaves.
"""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import sqlite3
import stat
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest

from titlestd import explorer, pipeline, store
from titlestd.contracts import MachinePath, RunResult, Stage, Status, TableRow
from titlestd.settings import Settings

HOST = "127.0.0.1:7654"
ROUTES = (*explorer.PAGES, *explorer.STATIC_FILES)
MARKER = "zq7hostile"
HOSTILE = (
    f"<script>alert('{MARKER}')</script>",
    f'"><img src=x onerror=alert(1)>{MARKER}',
    f"%%{{init: {{'securityLevel': 'loose'}}}}%%{MARKER}",
    f"{MARKER}\nclick rules call alert()",
    f"&lt;b&gt;{MARKER}&#x3C;/b&#x3E;",
    f"{MARKER}' OR '1'='1",
    f"{MARKER}{chr(0x202E)}{chr(0)}",
)
COUNT_ROW = re.compile(r'<tr><th scope="row">(.*?)</th><td>(.*?)</td></tr>')
LATER_STAGES = (Stage.WAVE1, Stage.WAVE2, Stage.CLOSE)
# What store.save_run writes in one step at the end of a run.
WRITTEN_AT_THE_END = (
    "decisions",
    "audits",
    "audit_samples",
    "golden_scores",
    "review_queue",
    "answer_labels",
)


@pytest.fixture(scope="module")
def run(tmp_path_factory, mini_dir: Path) -> tuple[RunResult, Path]:
    """One run of the mini data set: its result and its state file."""
    folder = tmp_path_factory.mktemp("explorer")
    settings = Settings(
        data_dir=mini_dir,
        out_dir=folder / "out",
        recording_dir=folder / "recordings",
        wave1_size=4,
        golden_size=3,
    )
    return pipeline.run_pipeline(settings), settings.out_dir / store.STATE_FILE


@pytest.fixture(scope="module")
def app(run):
    return explorer.make_app(run[1])


@pytest.fixture(scope="module")
def default_app(tmp_path_factory, repo_root: Path):
    """The page on a run of the default data, where the agent has work to do."""
    folder = tmp_path_factory.mktemp("explorer-default")
    settings = Settings(
        data_dir=repo_root / "data",
        out_dir=folder / "out",
        recording_dir=folder / "recordings",
    )
    pipeline.run_pipeline(settings)
    return explorer.make_app(settings.out_dir / store.STATE_FILE)


def call(app, path: str):
    """Call the WSGI application directly: (status, headers, body)."""
    seen = {}

    def start_response(status, headers):
        seen["status"], seen["headers"] = status, headers

    environ = {"REQUEST_METHOD": "GET", "PATH_INFO": path, "HTTP_HOST": HOST}
    body = b"".join(app(environ, start_response))
    return seen["status"], seen["headers"], body


class Page(HTMLParser):
    """What a test reads from a page: its tags with their attributes, and its text."""

    def __init__(self, body: bytes):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.scripts: list[str] = []
        self.charts: list[str] = []
        self._in: list[str] = []
        self.feed(body.decode("utf-8"))
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "script":
            self.scripts.append("")
        if tag == "pre" and attributes.get("class") == "mermaid":
            self.charts.append("")
        self._in.append(tag)

    def handle_endtag(self, tag):
        if self._in and self._in[-1] == tag:
            self._in.pop()

    def handle_data(self, data):
        if self._in and self._in[-1] == "script":
            self.scripts[-1] += data
        if self._in and self._in[-1] == "pre":
            self.charts[-1] += data

    def links(self) -> list[dict[str, str | None]]:
        return [attributes for tag, attributes in self.tags if tag == "a"]

    def with_id(self, value: str) -> list[str]:
        return [tag for tag, attributes in self.tags if attributes.get("id") == value]


def counts(body: bytes) -> dict[str, str]:
    """The table of counts of a page, label to value."""
    return {html.unescape(k): html.unescape(v) for k, v in COUNT_ROW.findall(body.decode())}


def numbers(app, path: str) -> dict[str, int]:
    """The counts of a page that are numbers, label to number."""
    return {
        label: int(value) for label, value in counts(call(app, path)[2]).items() if value.isdigit()
    }


# --- Every route -------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_every_route_answers(app, path):
    status, headers, body = call(app, path)
    assert status == "200 OK"
    content_type = dict(headers)["Content-Type"]
    if path in explorer.STATIC_FILES:
        assert content_type == explorer.STATIC_FILES[path][1]
        assert body == (explorer.STATIC_DIR / explorer.STATIC_FILES[path][0]).read_bytes()
    else:
        assert content_type == "text/html; charset=utf-8"
        assert body.startswith(b"<!doctype html>\n")


def test_the_pages_link_to_every_route_and_nothing_else(app):
    seen, waiting = set(), ["/"]
    while waiting:
        path = waiting.pop()
        if path in seen:
            continue
        seen.add(path)
        status, _, body = call(app, path)
        assert status == "200 OK", path
        if path in explorer.STATIC_FILES:
            continue
        page = Page(body)
        for _, attributes in page.tags:
            for name in ("href", "src", "hx-get"):
                if attributes.get(name):
                    waiting.append(attributes[name])
    assert seen == set(ROUTES)


@pytest.mark.parametrize("path", explorer.PAGES)
def test_a_link_is_a_real_href_that_htmx_follows_into_the_panel(app, path):
    page = Page(call(app, path)[2])
    for link in page.links():
        assert link["href"] and link["hx-get"] == link["href"]
    body = next(attributes for tag, attributes in page.tags if tag == "body")
    assert body["hx-target"] == "#panel" and body["hx-select"] == "#panel"
    assert body["hx-select-oob"] == "#nav" and body["hx-push-url"] == "true"
    assert body["hx-swap"].split()[0] == "outerHTML"
    assert page.with_id("panel") == ["main"] and page.with_id("nav") == ["nav"]
    current = [link["href"] for link in page.links() if link.get("aria-current") == "page"]
    assert current == ([] if path == "/" else [path])


@pytest.mark.parametrize("path", explorer.PAGES)
def test_a_page_holds_no_inline_script_handler_style_or_title_attribute(app, path):
    body = call(app, path)[2]
    page = Page(body)
    assert page.scripts == [""] * 3
    sources = [attributes["src"] for tag, attributes in page.tags if tag == "script"]
    assert sources == [
        "/static/vendor/htmx.min.js",
        "/static/vendor/mermaid.min.js",
        "/static/app.js",
    ]
    for tag, attributes in page.tags:
        for name, value in attributes.items():
            assert not name.startswith("on"), (tag, name)
            assert name not in {"title", "style"}, (tag, name)
            assert "javascript:" not in (value or "").lower(), (tag, name)
        if tag == "meta" and attributes.get("name") == "htmx-config":
            assert "transition:true" not in (attributes.get("content") or "")
    swaps = [attributes.get("hx-swap") for _, attributes in page.tags if "hx-swap" in attributes]
    assert all("transition" not in swap for swap in swaps)
    assert explorer.HTMX_CONFIG_META.encode() in body


@pytest.mark.parametrize("stage", explorer.STAGES, ids=lambda stage: stage.slug)
def test_a_stage_page_draws_its_flowchart_and_lists_its_components(app, stage):
    status, _, body = call(app, f"/stage/{stage.slug}")
    assert status == "200 OK"
    page = Page(body)
    assert page.charts == [stage.flowchart]
    # The chart goes in escaped, and Mermaid decodes it (static/vendor/README.md).
    assert html.escape(stage.flowchart).encode() in body
    assert stage.flowchart.encode() not in body
    acting = re.search(r'<ul class="acting">\n(.*?)</ul>', body.decode(), re.DOTALL)
    names = [html.unescape(name) for name in re.findall(r">([^<>]+)</a>", acting[1])]
    assert names == [c.name for c in explorer.COMPONENTS if c.node in stage.acting]


@pytest.mark.parametrize("component", explorer.COMPONENTS, ids=lambda c: c.slug)
def test_a_component_page_shows_its_job_module_sections_and_entries(app, component):
    status, _, body = call(app, f"/component/{component.slug}")
    assert status == "200 OK"
    text = html.unescape(body.decode())
    for passage in component.job:
        assert f"<p>{passage}</p>" in text
    assert f"<code>{component.module}</code>" in text
    for number, _ in component.design:
        assert f"{number} {explorer.DESIGN_SECTIONS[number]}" in text
    assert f"docs/build-decisions.md {', '.join(component.entries)}" in text
    assert len(counts(body)) == len(component.facts)


# --- The counts are the run's ------------------------------------------------------


def test_the_counts_are_those_of_the_run(app, run):
    result, _ = run
    rows = Counter(decision.row for decision in result.decisions)
    statuses = Counter(decision.status for decision in result.decisions)
    audits = {audit.path: audit for audit in result.audits}

    def page(path):
        return counts(call(app, path)[2])

    overview = page("/")
    assert overview["Title rows"] == str(len(result.rows))
    assert overview["Answers came from"] == "stand-ins"
    assert overview["LLM setting"] == "offline" and overview["Embedder setting"] == "offline"

    prepare = page("/component/prepare")
    assert prepare["With a person's name"] == str(sum(row.has_person_name for row in result.rows))
    assert prepare["With no usable family code"] == str(
        sum(not row.family_usable for row in result.rows)
    )

    setup = page("/stage/set-up")
    assert setup["Titles on the Wave 1 list"] == str(len(result.wave1_keys))
    assert setup["Titles in the golden set"] == str(len(result.golden_keys))

    wave1 = page("/stage/wave-1")
    assert wave1["Look-up hits"] == str(rows[TableRow.LOOKUP])
    assert wave1["Golden-set labels"] == str(len(result.golden_keys))

    wave2 = page("/stage/wave-2")
    for path, name in (
        (MachinePath.MODELS_AGREE, "models-agree"),
        (MachinePath.AGENT_SETTLED, "agent-settled"),
    ):
        assert wave2[f"On the {name} path"] == str(audits[path].path_size)
        assert wave2[f"Audited on the {name} path"] == str(audits[path].audited)
        assert wave2[f"Outcome of the {name} path"] == audits[path].outcome.value

    close = page("/stage/close")
    assert close["Given a generic family role"] == str(rows[TableRow.GENERIC])
    assert close["Left on a legacy title"] == str(rows[TableRow.LEGACY])
    assert close["Titles in the review-queue file"] == str(len(result.queue))

    report = page("/component/report")
    assert {row: report[f"Titles in the row {row.value}"] for row in TableRow} == {
        row: str(rows[row]) for row in TableRow
    }
    release = page("/component/release")
    assert {s: release[f"Status {s.value}"] for s in Status} == {
        s: str(statuses[s]) for s in Status
    }
    golden = page("/component/golden-set")
    assert golden["Times scored"] == str(len(result.golden_scores))
    assert golden["Golden-set titles learned"] == "0"
    state = page("/component/state-file")
    assert state["Rows in decisions"] == str(len(result.rows))
    assert state["Rows in answer_labels"] == str(len(result.answer_labels))


@pytest.mark.parametrize("data", ["mini", pytest.param("default", marks=pytest.mark.e2e)])
def test_the_counts_that_split_a_total_add_up_to_it(request, data):
    app = request.getfixturevalue("app" if data == "mini" else "default_app")
    fallback = numbers(app, "/component/llm-fallback")
    answers = sum(fallback[f"LLM answers in {stage.value}"] for stage in LATER_STAGES)
    kinds = ("Answered with a number", "Answered no match", "Answered insufficient data")
    assert answers == sum(fallback[kind] for kind in (*kinds, "No usable answer"))

    agent = numbers(app, "/component/agent")
    reviews = agent["Reviews in Wave 1"] + agent["Reviews in Wave 2"]
    assert (
        reviews == agent["Answered with a number"] + agent["Escalated"] + agent["No usable answer"]
    )

    rules = numbers(app, "/component/rules")
    assert answers == sum(
        rules[label]
        for label in (
            "LLM answers that passed every check",
            "LLM answers sent to the agent",
            "LLM answers sent to a person",
            "LLM answers of the golden-set passes, only scored",
        )
    )
    # The agent reviews every LLM answer the rules send it, outside the golden-set passes.
    assert rules["LLM answers sent to the agent"] == reviews
    assert reviews == (
        rules["Agent answers that passed every check"] + rules["Agent answers sent to a person"]
    )

    people = numbers(app, "/component/people")
    occasions = ("Wave 1 approvals", "Golden-set labels", "Audit labels", "Review-queue decisions")
    assert people["Marked as simulated"] == sum(people[occasion] for occasion in occasions)

    close = numbers(app, "/stage/close")
    assert close["Review-queue decisions"] == sum(
        close[answer] for answer in ("Given a role", "Could not decide", "No suitable role")
    )

    lookup, learning = numbers(app, "/component/look-up"), numbers(app, "/component/learning")
    assert lookup["Entries at the end of the run"] == lookup["Entries from the seed look-up"] + sum(
        learning[f"Look-up entries learned in {stage.value}"] for stage in LATER_STAGES
    )


def test_the_mini_run_gives_the_pages_something_to_count(app):
    # Positive control: on a real run the pages show more than zeros.
    wave2 = numbers(app, "/stage/wave-2")
    assert wave2["Titles scored by the models"] > 0
    assert wave2["On the models-agree path"] > 0
    assert numbers(app, "/stage/wave-1")["Wave 1 approvals"] > 0


@pytest.mark.e2e
def test_the_default_run_gives_every_total_split_above_something_to_split(default_app):
    # Positive control for the sums: each total is more than zero on the default data.
    rules = numbers(default_app, "/component/rules")
    assert rules["LLM answers sent to the agent"] > 0
    assert rules["LLM answers of the golden-set passes, only scored"] > 0
    assert rules["Agent answers sent to a person"] > 0
    assert numbers(default_app, "/component/people")["Review-queue decisions"] > 0
    assert numbers(default_app, "/stage/close")["Could not decide"] > 0
    learning = numbers(default_app, "/component/learning")
    assert all(learning[f"Look-up entries learned in {stage.value}"] > 0 for stage in LATER_STAGES)


@pytest.mark.parametrize("path", explorer.PAGES)
def test_a_finished_run_is_shown_as_finished(app, path):
    body = call(app, path)[2]
    assert explorer._e(explorer.FINISHED_HEADING).encode() in body
    assert explorer._e(explorer.UNFINISHED_HEADING).encode() not in body


def test_a_run_that_did_not_finish_is_shown_as_it_stood(run, tmp_path):
    folder = tmp_path / "stopped"
    folder.mkdir()
    copy = folder / store.STATE_FILE
    shutil.copyfile(run[1], copy)
    connection = sqlite3.connect(copy)
    with connection:
        for table in WRITTEN_AT_THE_END:
            connection.execute(f"DELETE FROM {table}")
    connection.close()
    app = explorer.make_app(copy)
    for path in explorer.PAGES:
        status, _, body = call(app, path)
        assert status == "200 OK", path
        assert explorer._e(explorer.UNFINISHED_HEADING).encode() in body, path
        assert explorer._e(explorer.FINISHED_HEADING).encode() not in body, path
    # The model answers carry their own labels before the run writes the answer labels.
    assert counts(call(app, "/")[2])["Answers came from"] == "stand-ins"
    assert numbers(app, "/component/llm-fallback")["LLM answers in Wave 2"] > 0


# --- Hostile text ------------------------------------------------------------------


def text_columns(connection: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    """Every column of every table, by table: (name, declared type)."""
    tables = [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    ]
    return {
        table: [(row[1], row[2]) for row in connection.execute(f"PRAGMA table_info({table})")]
        for table in tables
    }


def hostile_copy(state: Path, folder: Path, *, add_row: bool) -> Path:
    """A copy of the state with hostile text in every text column."""
    folder.mkdir()
    copy = folder / store.STATE_FILE
    shutil.copyfile(state, copy)
    connection = sqlite3.connect(copy)
    columns = text_columns(connection)
    assert sorted(columns) == sorted(explorer.TABLES)
    texts = 0
    with connection:
        for table, described in columns.items():
            text = [name for name, kind in described if kind in {"TEXT", ""}]
            texts += len(text)
            empty = not connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if not add_row:
                for at, name in enumerate(text):
                    connection.execute(
                        f"UPDATE {table} SET {name} = COALESCE({name}, '') || ?",
                        (HOSTILE[at % len(HOSTILE)],),
                    )
            if add_row or empty:
                values = []
                for at, (name, kind) in enumerate(described):
                    if name in text:
                        values.append(f"{HOSTILE[at % len(HOSTILE)]} {table}.{name}")
                    else:
                        values.append(1 if kind == "INTEGER" else 0.5)
                marks = ", ".join("?" for _ in values)
                connection.execute(f"INSERT INTO {table} VALUES ({marks})", values)
        # The free llm setting the page needs to serve at all (the hostile one stays too).
        connection.execute("INSERT OR REPLACE INTO settings VALUES ('llm', 'offline')")
        # Every text column now holds the marker in some row.
        for table, described in columns.items():
            for name, kind in described:
                if kind in {"TEXT", ""}:
                    found = connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE CAST({name} AS TEXT) LIKE ?",
                        (f"%{MARKER}%",),
                    ).fetchone()[0]
                    assert found, (table, name, add_row)
    connection.close()
    assert texts > 40  # the schema's text columns, not an empty listing
    return copy


@pytest.mark.parametrize("add_row", [True, False], ids=["hostile rows", "hostile values"])
def test_hostile_text_in_every_text_column_reaches_no_response(run, tmp_path, add_row):
    copy = hostile_copy(run[1], tmp_path / "state", add_row=add_row)
    app = explorer.make_app(copy)
    for path in (*ROUTES, "/nothing-here"):
        status, headers, body = call(app, path)
        assert status in {"200 OK", "404 Not Found"}, path
        assert MARKER.encode() not in body, path
        assert all(MARKER not in value for _, value in headers), path


def test_the_hostile_check_sees_a_page_that_shows_a_value_of_the_state(run, tmp_path, monkeypatch):
    # Positive control for the test above: let one count show a stored label instead.
    copy = hostile_copy(run[1], tmp_path / "state", add_row=True)

    def shows_a_label(connection, fact):
        query = "SELECT label FROM answer_labels WHERE label LIKE ?"
        return connection.execute(query, (f"%{MARKER}%",)).fetchone()[0]

    monkeypatch.setattr(explorer, "_value", shows_a_label)
    status, _, body = call(explorer.make_app(copy), "/")
    assert status == "200 OK" and MARKER.encode() in body


# --- The state file and the bytes --------------------------------------------------


def fingerprint(state: Path) -> tuple[str, list[str]]:
    return (
        hashlib.sha256(state.read_bytes()).hexdigest(),
        sorted(path.name for path in state.parent.iterdir()),
    )


def test_the_state_file_is_left_as_it_was(run):
    state = run[1]
    before = fingerprint(state)
    app = explorer.make_app(state)
    for _ in range(2):
        for path in ROUTES:
            call(app, path)
    assert fingerprint(state) == before


def test_a_read_only_state_in_a_read_only_folder_is_served(run, tmp_path):
    folder = tmp_path / "read-only"
    folder.mkdir()
    state = folder / store.STATE_FILE
    shutil.copyfile(run[1], state)
    before = fingerprint(state)
    state.chmod(stat.S_IRUSR)
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        app = explorer.make_app(state)
        for path in ROUTES:
            assert call(app, path)[0] == "200 OK", path
        assert fingerprint(state) == before
    finally:
        folder.chmod(stat.S_IRWXU)
        state.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_every_route_gives_the_same_bytes_twice(run, app):
    again = explorer.make_app(run[1])
    for path in (*ROUTES, "/nothing-here"):
        first = call(app, path)
        assert call(app, path) == first, path
        assert call(again, path) == first, path
