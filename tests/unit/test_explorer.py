"""The architecture page: its fixed text, its source and what it refuses.

Owner: feat/architecture-page (after the build).
Governed by: DESIGN.md 1.10 (the waves), 2.8 (the two browser libraries);
docs/build-decisions.md "Diagram rules not to copy", Q40 (the answer labels);
static/vendor/README.md (how the page loads the libraries).

Everything here runs without a run of the pipeline: the flowcharts and the
cited text against docs/architecture.md, DESIGN.md and the decisions record,
the source of titlestd/explorer.py, the two static files, and the refusals on
a state file made empty with titlestd.store. The pages on a real run are in
tests/integration/test_explorer.py.

Decided here, where the record is silent:

- "Text from DESIGN.md" is held word for word: each quoted passage, with
  its white space collapsed, is in DESIGN.md with its white space collapsed
  and its bold and code marks taken out.
- "One html.escape helper for every value" is held on the source: html.escape
  is named once, inside _e, and every value put into an f-string that builds
  markup is a call of _e. A checker that let a value through would pass on
  silence, so it is shown to find one first.
- A flowchart "free of directives, clicks and HTML" is held as a short list of
  characters a line may use, which has no room for a directive, an entity, a
  tag or a quoted label, and no line may name click, call or href.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import pytest

from titlestd import datafiles, explorer, people, store
from titlestd.contracts import LABEL_LIVE, LABEL_STAND_IN, recorded_label
from titlestd.settings import LLMS

HOST = "127.0.0.1:7654"
EXPLORER_SOURCE = Path(explorer.__file__)
FENCE = re.compile(r"^```mermaid\n(.*?)^```$", re.MULTILINE | re.DOTALL)
# What a line of a flowchart may hold beside the style line: names, labels and arrows.
CHART_LINE = re.compile(r"[A-Za-z0-9 _,.()\[\]|>-]*")
NODE = re.compile(r"\b([a-z][a-z0-9_]*)(\[\(|\(\[|\[)([^\]\[()|]*)(\)\]|\]\)|\])")


def collapsed(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def design(repo_root: Path) -> str:
    return (repo_root / "DESIGN.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def design_words(design: str) -> str:
    return collapsed(design.replace("**", "").replace("`", ""))


@pytest.fixture(scope="module")
def record(repo_root: Path) -> str:
    return (repo_root / "docs" / "build-decisions.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def vendor_readme(repo_root: Path) -> str:
    return (repo_root / "static" / "vendor" / "README.md").read_text(encoding="utf-8")


def call(app, path: str = "/", method: str = "GET", host: str | None = HOST):
    """Call the WSGI application directly: (status, headers, body)."""
    environ = {"REQUEST_METHOD": method, "PATH_INFO": path}
    if host is not None:
        environ["HTTP_HOST"] = host
    seen = {}

    def start_response(status, headers):
        seen["status"], seen["headers"] = status, dict(headers)

    body = b"".join(app(environ, start_response))
    return seen["status"], seen["headers"], body


def empty_state(folder: Path, *labels: str) -> Path:
    """A state file as titlestd.store makes it, holding only the given answer labels."""
    path = folder / store.STATE_FILE
    connection = store.create_store(path)
    with connection:
        connection.executemany("INSERT INTO answer_labels VALUES (?)", [(x,) for x in labels])
    connection.close()
    return path


# --- The flowcharts --------------------------------------------------------------


def test_each_flowchart_is_its_fence_in_docs_architecture(repo_root: Path):
    document = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert FENCE.findall(document) == [stage.flowchart for stage in explorer.STAGES]


def test_the_stages_are_the_four_of_a_run_in_order():
    assert [stage.stage.value for stage in explorer.STAGES] == [
        "set-up",
        "Wave 1",
        "Wave 2",
        "close",
    ]


@pytest.mark.parametrize("stage", explorer.STAGES, ids=lambda stage: stage.slug)
def test_a_flowchart_holds_no_directive_click_or_html(stage):
    lines = stage.flowchart.splitlines()
    assert lines[0] == "flowchart TD"
    assert stage.flowchart.endswith("\n")
    for line in lines[1:]:
        body = line.strip()
        if body == explorer.ACTING_STYLE:
            continue
        assert CHART_LINE.fullmatch(body), line
        assert not re.search(r"\b(click|call|href|style|linkStyle|init)\b", body), line
    assert "%%" not in stage.flowchart and "<" not in stage.flowchart


@pytest.mark.parametrize("stage", explorer.STAGES, ids=lambda stage: stage.slug)
def test_a_flowchart_styles_exactly_the_components_acting_in_its_stage(stage):
    classes = [line.split() for line in stage.flowchart.splitlines() if "class " in line]
    assert len(classes) == 1
    keyword, nodes, name = classes[0]
    assert (keyword, name) == ("class", "acting")
    assert sorted(nodes.split(",")) == sorted(stage.acting)

    names = {component.node: component.name for component in explorer.COMPONENTS}
    drawn = {node: label for node, _, label, _ in NODE.findall(stage.flowchart)}
    assert {node: drawn.get(node) for node in stage.acting} == {
        node: names[node] for node in stage.acting
    }
    # A component drawn in a stage acts in it: none is drawn unstyled.
    assert set(drawn) & set(names) == set(stage.acting)


def test_every_component_acts_in_some_stage_and_has_its_own_node():
    nodes = [component.node for component in explorer.COMPONENTS]
    assert len(set(nodes)) == len(nodes)
    acting = {node for stage in explorer.STAGES for node in stage.acting}
    assert acting == set(nodes)


def test_the_flowcharts_draw_none_of_the_diagram_rules_not_to_copy():
    # docs/build-decisions.md, "Diagram rules not to copy", in the words a chart would use.
    charts = "\n".join(stage.flowchart for stage in explorer.STAGES).lower()
    for words in ("catalogue", "matcher", "sustaining", "200", "300", "escalate", "adjudicat"):
        assert words not in charts, words
    assert charts.count("path]") == 2  # two machine paths, not three
    wave1, wave2 = explorer.WAVE1_FLOWCHART, explorer.WAVE2_FLOWCHART
    assert "ml_match[ML match]" in wave1  # the ML match runs from Wave 1 on
    assert "at most 2 wrong of 60" in wave2 and "13 or more wrong of 60" in wave2
    # A look-up hit goes to no model and not through the rules.
    assert "lookup -->|hit, no model| hit(" in wave1 and "hit -->" not in wave1
    # "No suitable role" goes to the catalog owner, never to a generic role.
    assert "people -->|no suitable role| catalogowner(" in explorer.CLOSE_FLOWCHART
    assert "catalogowner --> generic" not in explorer.CLOSE_FLOWCHART


# --- What the pages cite ---------------------------------------------------------


def test_the_cited_design_sections_exist_with_their_headings(design: str):
    headings = set(re.findall(r"^### (\d\.\d+) (.+)$", design, re.MULTILINE))
    assert set(explorer.DESIGN_SECTIONS.items()) <= headings
    for component in explorer.COMPONENTS:
        assert component.design
        assert {number for number, _ in component.design} <= set(explorer.DESIGN_SECTIONS)


@pytest.mark.parametrize("component", explorer.COMPONENTS, ids=lambda c: c.slug)
def test_a_component_cites_modules_that_exist_and_entries_they_name(
    component, repo_root: Path, record: str
):
    docstrings = ""
    for module in component.modules:
        assert (repo_root / module).is_file(), module
        tree = ast.parse((repo_root / module).read_text(encoding="utf-8"))
        docstrings += ast.get_docstring(tree) or ""
    assert component.entries
    for entry in component.entries:
        assert re.search(rf"^### {entry}\. ", record, re.MULTILINE), entry
        assert re.search(rf"\b{entry}\b", docstrings), entry


def test_every_quoted_passage_is_word_for_word_in_design(design_words: str):
    passages = [*explorer.IDEA, explorer.RUN_TEXT]
    for component in explorer.COMPONENTS:
        passages += component.job
    for stage in explorer.STAGES:
        passages += [stage.design, *stage.left_out]
    for passage in passages:
        assert collapsed(passage) in design_words, passage


@pytest.mark.parametrize("stage", explorer.STAGES, ids=lambda stage: stage.slug)
def test_a_stage_quotes_its_row_of_the_waves_table(stage, design: str):
    assert f"| {stage.design_wave} | {stage.design} |" in design.splitlines()


def test_the_words_check_finds_a_passage_that_is_not_in_design(design_words: str):
    # Positive control for the check above.
    assert collapsed("A model writes the Long Title itself.") not in design_words


def test_the_page_reads_every_table_the_store_makes(tmp_path: Path):
    connection = sqlite3.connect(empty_state(tmp_path))
    made = [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    ]
    connection.close()
    assert sorted(made) == sorted(explorer.TABLES)


# --- The source of the page ------------------------------------------------------


def unescaped_values(source: str) -> list[int]:
    """Lines of f-strings that build markup and put in a value other than _e(...)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
        if "<" not in literal:
            continue
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                call = part.value
                escaped = (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_e"
                    and part.conversion == -1
                    and part.format_spec is None
                )
                if not escaped:
                    found.append(part.lineno)
    return found


def test_every_value_in_markup_passes_the_one_escape():
    assert unescaped_values(EXPLORER_SOURCE.read_text(encoding="utf-8")) == []


def test_the_escape_check_finds_a_value_put_in_unescaped():
    # Positive control: without it the test above could pass on silence.
    source = 'title = "x"\npage = f"<h1>{_e(title)}</h1>" + f"<p>{title}</p>"\n'
    assert unescaped_values(source) == [2]


def test_html_escape_is_named_once_inside_the_one_escape():
    tree = ast.parse(EXPLORER_SOURCE.read_text(encoding="utf-8"))
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "escape"
        and isinstance(node.value, ast.Name)
        and node.value.id == "html"
    ]
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_e"
    )
    assert len(uses) == 1 and uses[0] in list(ast.walk(helper))


def test_the_escape_escapes_every_value_and_keeps_markup():
    assert (
        explorer._e("<a href=\"x\">'&'</a>")
        == "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"
    )
    assert explorer._e(7) == "7"
    built = explorer._e("<b>")
    assert explorer._e(built) is built


def test_the_page_names_no_data_file():
    source = EXPLORER_SOURCE.read_text(encoding="utf-8")
    for name in (*datafiles.INPUT_FILES, people.TRUTH_FILE):
        assert name not in source, name


def test_the_page_imports_only_the_standard_library_and_the_package():
    tree = ast.parse(EXPLORER_SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported - {"titlestd"} <= set(sys.stdlib_module_names)


# --- The static files ------------------------------------------------------------


@pytest.fixture(scope="module")
def app_js(repo_root: Path) -> str:
    return (repo_root / "static" / "app.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_css(repo_root: Path) -> str:
    return (repo_root / "static" / "app.css").read_text(encoding="utf-8")


def test_the_script_sets_up_mermaid_and_htmx_as_the_vendor_readme_says(app_js, vendor_readme):
    initialize = 'mermaid.initialize({startOnLoad: false, securityLevel: "strict"})'
    assert initialize in vendor_readme
    assert app_js.count(initialize) == 1
    assert "htmx.onLoad(" in app_js and "mermaid.run({nodes: charts})" in app_js
    assert 'element.matches("pre.mermaid")' in app_js  # the loaded element itself included
    assert app_js.index("mermaid.initialize(") < app_js.index("htmx.onLoad(")


def test_the_script_writes_no_markup_and_evaluates_nothing(app_js):
    for words in (
        "innerHTML",
        "outerHTML",
        "eval(",
        "Function(",
        "setAttribute",
        '"title"',
        ".title",
    ):
        assert words not in app_js, words


def test_nothing_on_the_page_moves_on_its_own(app_js, app_css):
    for text in (app_js, app_css):
        for words in ("transition", "animation", "@keyframes"):
            assert words not in text.lower(), words
    assert "@import" not in app_css and "url(" not in app_css  # nothing fetched from elsewhere


def test_the_page_serves_only_the_two_static_files_and_the_two_libraries(repo_root: Path):
    served = {name for name, _ in explorer.STATIC_FILES.values()}
    assert served == {"app.js", "app.css", "vendor/htmx.min.js", "vendor/mermaid.min.js"}
    for name in served:
        assert (repo_root / "static" / name).is_file(), name
    assert repo_root / "static" == explorer.STATIC_DIR


def test_the_policy_and_the_htmx_settings_are_the_vendor_readmes(vendor_readme):
    policy = re.search(r"sends:\*\* `([^`]*)`", vendor_readme)
    assert policy and collapsed(policy[1]) == explorer.CONTENT_SECURITY_POLICY
    assert explorer.HTMX_CONFIG_META in vendor_readme


# --- What it refuses ---------------------------------------------------------------


def test_a_missing_state_file_is_refused(tmp_path: Path):
    with pytest.raises(explorer.ExplorerError, match="no state file"):
        explorer.make_app(tmp_path / store.STATE_FILE)


@pytest.mark.parametrize("content", [b"", b"not a database\n"], ids=["empty", "text"])
def test_a_file_that_is_no_state_file_is_refused(tmp_path: Path, content: bytes):
    path = tmp_path / store.STATE_FILE
    path.write_bytes(content)
    with pytest.raises(explorer.ExplorerError, match="not a state file"):
        explorer.make_app(path)


def test_a_state_with_live_answers_is_refused(tmp_path: Path):
    path = empty_state(tmp_path, LABEL_STAND_IN, LABEL_LIVE)
    with pytest.raises(explorer.ExplorerError, match="live run"):
        explorer.make_app(path)


@pytest.mark.parametrize(
    ("table", "row"),
    [
        (
            "model_answers",
            ("Wave 1", "FIN", "t", "choose", "role", 1, "q", LABEL_LIVE, "", "R", "a", ""),
        ),
        ("settings", ("llm", "anthropic")),
    ],
    ids=["a live answer", "the live llm setting"],
)
def test_a_live_run_that_stopped_before_its_labels_is_refused(tmp_path: Path, table, row):
    # The answer labels are written when a run finishes; a stopped live run has none.
    path = empty_state(tmp_path)
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in row)})", row)
    connection.close()
    with pytest.raises(explorer.ExplorerError, match="live run"):
        explorer.make_app(path)


def test_the_live_llm_setting_is_one_the_settings_take():
    assert explorer._LIVE_LLM in LLMS and explorer._LIVE_LLM not in {"offline", "replay"}


@pytest.mark.parametrize(
    "labels",
    [(), (LABEL_STAND_IN,), (recorded_label("2026-10-05", "claude-haiku-4-5"),)],
    ids=["no answers", "stand-in", "recorded"],
)
def test_a_free_runs_state_is_served(tmp_path: Path, labels):
    status, _, _ = call(explorer.make_app(empty_state(tmp_path, *labels)))
    assert status == "200 OK"


def test_the_state_is_opened_read_only(tmp_path: Path):
    path = empty_state(tmp_path)
    with explorer._reading(path) as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("INSERT INTO answer_labels VALUES ('x')")


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "localhost:7654",
        "127.0.0.1",
        "127.0.0.1:80",
        "127.0.0.1:7654.example.com",
        "[::1]:7654",
        "example.com",
    ],
)
def test_any_other_host_is_refused(tmp_path: Path, host):
    app = explorer.make_app(empty_state(tmp_path))
    for path in ("/", "/stage/wave-1", "/static/app.js", "/no-page"):
        status, headers, body = call(app, path, host=host)
        assert status == "403 Forbidden"
        assert body == explorer._REFUSED_HOST.encode("utf-8")
        assert headers["Content-Security-Policy"] == explorer.CONTENT_SECURITY_POLICY


@pytest.mark.parametrize("method", ["HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "get"])
def test_any_method_but_get_is_refused(tmp_path: Path, method):
    app = explorer.make_app(empty_state(tmp_path))
    for path in ("/", "/component/rules", "/static/app.css"):
        status, headers, body = call(app, path, method=method)
        assert status == "405 Method Not Allowed"
        assert headers["Allow"] == "GET"
        assert body == explorer._REFUSED_METHOD.encode("utf-8")


def test_a_page_answers_503_once_the_state_is_gone_or_live(tmp_path: Path):
    path = empty_state(tmp_path)
    app = explorer.make_app(path)
    assert call(app)[0] == "200 OK"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("INSERT INTO answer_labels VALUES (?)", (LABEL_LIVE,))
    connection.close()
    assert call(app)[:1] == ("503 Service Unavailable",)
    path.unlink()
    status, _, body = call(app, "/stage/close")
    assert status == "503 Service Unavailable" and body == explorer._NO_STATE.encode("utf-8")
    assert call(app, "/static/app.js")[0] == "200 OK"  # the static files need no state


def test_an_unknown_address_is_404_and_not_repeated(tmp_path: Path):
    app = explorer.make_app(empty_state(tmp_path))
    for path in ("/nothing-here", "/stage/wave-3", "/static/vendor/LICENSE-htmx.txt", "/../x"):
        status, _, body = call(app, path)
        assert status == "404 Not Found"
        assert path.encode("utf-8") not in body


def test_every_response_carries_the_fixed_policy(tmp_path: Path):
    app = explorer.make_app(empty_state(tmp_path))
    for path in (*explorer.PAGES, *explorer.STATIC_FILES, "/nothing-here"):
        _, headers, body = call(app, path)
        assert headers["Content-Security-Policy"] == explorer.CONTENT_SECURITY_POLICY, path
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Content-Length"] == str(len(body))


# --- Starting it -------------------------------------------------------------------


class FakeServer:
    """Stands in for wsgiref's server: records where it would listen, serves nothing."""

    def __init__(self, host, port, app):
        self.address, self.app = (host, port), app

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def serve_forever(self):
        raise KeyboardInterrupt


def test_the_start_command_refuses_without_a_state_file(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", pytest.fail)
    assert explorer.main() == 1
    assert "no state file at out/state.sqlite" in capsys.readouterr().err


def test_the_start_command_refuses_a_live_state(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "out").mkdir()
    empty_state(tmp_path / "out", LABEL_LIVE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", pytest.fail)
    assert explorer.main() == 1
    assert "live run" in capsys.readouterr().err


def test_the_start_command_listens_only_on_the_loopback_address(
    tmp_path: Path, monkeypatch, capsys
):
    (tmp_path / "out").mkdir()
    state = empty_state(tmp_path / "out", LABEL_STAND_IN)
    before = hashlib.sha256(state.read_bytes()).hexdigest()
    servers = []

    def make_server(host, port, app):
        servers.append(FakeServer(host, port, app))
        return servers[-1]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", make_server)
    assert explorer.main() == 0
    assert [server.address for server in servers] == [("127.0.0.1", 7654)]
    assert explorer.ADDRESS in capsys.readouterr().out
    assert hashlib.sha256(state.read_bytes()).hexdigest() == before
