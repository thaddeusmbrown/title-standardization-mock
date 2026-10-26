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
  and its bold and code marks taken out. A component's passage is held to
  the sections its page cites.
- "One html.escape helper for every value" is held on the source: html.escape
  is named once, inside _e, every value put into an f-string that builds
  markup (one holding "<", or one wrapped in _Html) is a call of _e, and
  _Html wraps only literals, such f-strings, a module constant or _join's
  join of escaped parts. A checker that let a value through would pass on
  silence, so it is shown to find one first.
- "One thread per request" is held on a real server bound to a free port of
  the loopback address: an idle connection is held open while a second one
  asks for a page.
- A flowchart "free of directives, clicks and HTML" is held as a short list of
  characters a line may use, which has no room for a directive, an entity, a
  tag or a quoted label, and no line may name click, call or href.
"""

from __future__ import annotations

import ast
import hashlib
import re
import socket
import sqlite3
import sys
import threading
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, make_server

import pytest

from titlestd import datafiles, explorer, people, store
from titlestd.contracts import LABEL_LIVE, LABEL_STAND_IN, Stage, recorded_label
from titlestd.settings import EXAMPLE_LLM_MODEL, LLMS

HOST = "127.0.0.1:7654"
EXPLORER_SOURCE = Path(explorer.__file__)
FENCE = re.compile(r"^```mermaid\n(.*?)^```$", re.MULTILINE | re.DOTALL)
# What a line of a flowchart may hold beside the style line: names, labels and arrows.
CHART_LINE = re.compile(r"[A-Za-z0-9 _,.()\[\]|>-]*")
NODE = re.compile(r"\b([a-z][a-z0-9_]*)(\[\(|\(\[|\[)([^\]\[()|]*)(\)\]|\]\)|\])")


def collapsed(text: str) -> str:
    return " ".join(text.split())


def words(markdown: str) -> str:
    """The text with its white space collapsed and its bold and code marks taken out."""
    return collapsed(markdown.replace("**", "").replace("`", ""))


@pytest.fixture(scope="module")
def design(repo_root: Path) -> str:
    return (repo_root / "DESIGN.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def design_words(design: str) -> str:
    return words(design)


@pytest.fixture(scope="module")
def design_sections(design: str) -> dict[str, str]:
    """The words of each numbered section of DESIGN.md, by its number."""
    parts = re.split(r"^### (\d\.\d+) .*$", design, flags=re.MULTILINE)
    return {number: words(text) for number, text in zip(parts[1::2], parts[2::2], strict=True)}


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


def add_answers(path: Path, *labels: str) -> None:
    """One model answer for each label, as a run that stopped before its end holds them."""
    connection = sqlite3.connect(path)
    with connection:
        connection.executemany(
            "INSERT INTO model_answers (stage, family_code, cleaned_title, step, answer, quote,"
            " source, detail, route, reasons) VALUES ('Wave 1', 'FIN', ?, 'choose', 'role', 'q',"
            " ?, '', 'accept', '')",
            [(f"title {at}", label) for at, label in enumerate(labels)],
        )
    connection.close()


def empty_state(folder: Path, *labels: str, llm: str | None = "offline") -> Path:
    """A state file as titlestd.store makes it, holding only the llm setting (none
    when llm is None) and one model answer for each label given."""
    path = folder / store.STATE_FILE
    connection = store.create_store(path)
    with connection:
        if llm is not None:
            connection.execute("INSERT INTO settings VALUES ('llm', ?)", (llm,))
    connection.close()
    add_answers(path, *labels)
    return path


# --- The flowcharts --------------------------------------------------------------


def test_each_flowchart_is_its_fence_in_docs_architecture(repo_root: Path):
    document = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert FENCE.findall(document) == [stage.flowchart for stage in explorer.STAGES]


def test_the_stages_are_the_four_of_a_run_in_order():
    assert [stage.stage for stage in explorer.STAGES] == list(Stage)


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
def test_a_component_cites_a_module_that_exists_and_entries_it_names(
    component, repo_root: Path, record: str
):
    assert (repo_root / component.module).is_file(), component.module
    tree = ast.parse((repo_root / component.module).read_text(encoding="utf-8"))
    docstrings = ast.get_docstring(tree) or ""
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


@pytest.mark.parametrize("component", explorer.COMPONENTS, ids=lambda c: c.slug)
def test_a_component_quotes_only_the_sections_it_cites(component, design_sections):
    cited = [design_sections[number] for number, _ in component.design]
    for passage in component.job:
        assert any(collapsed(passage) in section for section in cited), passage


def test_the_sections_check_finds_a_passage_cited_to_the_wrong_section(design_sections):
    # Positive control for the check above: a passage of 1.4, cited to 1.6 alone.
    passage = collapsed("Names are masked before any text reaches a model.")
    assert passage in design_sections["1.4"] and passage not in design_sections["1.6"]


def test_the_page_reads_every_table_the_store_makes(tmp_path: Path):
    connection = sqlite3.connect(empty_state(tmp_path))
    made = [
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    ]
    connection.close()
    assert sorted(made) == sorted(explorer.TABLES)


# --- The source of the page ------------------------------------------------------


def calls(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def wrapped_as_markup(node: ast.AST) -> bool:
    """What _Html may wrap: a literal, an f-string (checked below), a module
    constant, the result of html.escape in _e, or _join's join of escaped parts."""
    if isinstance(node, ast.Constant | ast.JoinedStr):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper()
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if isinstance(node.func.value, ast.Name) and node.func.value.id == "html":
        return node.func.attr == "escape"
    return (
        node.func.attr == "join"
        and calls(node.func.value, "_e")
        and len(node.args) == 1
        and isinstance(node.args[0], ast.GeneratorExp)
        and calls(node.args[0].elt, "_e")
    )


def unescaped_values(source: str) -> list[int]:
    """Lines where a value reaches markup other than through _e: in an f-string
    that builds markup (one holding "<", or one wrapped in _Html), or wrapped
    in _Html itself."""
    tree = ast.parse(source)
    wrapped = [arg for node in ast.walk(tree) if calls(node, "_Html") for arg in node.args]
    found = [arg.lineno for arg in wrapped if not wrapped_as_markup(arg)]
    markup = [node for node in wrapped if isinstance(node, ast.JoinedStr)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
        if "<" not in literal and not any(node is each for each in markup):
            continue
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                escaped = (
                    calls(part.value, "_e") and part.conversion == -1 and part.format_spec is None
                )
                if not escaped:
                    found.append(part.lineno)
    return sorted(found)


def test_every_value_in_markup_passes_the_one_escape():
    assert unescaped_values(EXPLORER_SOURCE.read_text(encoding="utf-8")) == []


def test_the_escape_check_finds_a_value_put_in_unescaped():
    # Positive control: without it the test above could pass on silence.
    source = (
        'title = "x"\n'
        'page = f"<h1>{_e(title)}</h1>" + f"<p>{title}</p>"\n'
        "marks = _Html(f' id=\"{title}\"')\n"
        "named = _Html(title)\n"
        "kept = _Html(f' id=\"{_e(title)}\"') + _Html(HTMX_CONFIG_META) + _Html('')\n"
    )
    assert unescaped_values(source) == [2, 3, 4]


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


def test_the_script_loads_in_full_an_address_htmx_could_not_load(app_js):
    # htmx swaps no error into the page (Decided here in titlestd/explorer.py).
    for event in ("htmx:responseError", "htmx:sendError", "htmx:historyCacheMissLoadError"):
        assert f'"{event}"' in app_js, event
    assert "window.location.assign(event.detail.pathInfo.requestPath)" in app_js
    assert "window.location.reload()" in app_js


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


def test_a_state_with_a_live_answer_is_refused(tmp_path: Path):
    # Before its end a run has no answer labels, but each model answer carries its own.
    path = empty_state(tmp_path, LABEL_STAND_IN, LABEL_LIVE)
    with pytest.raises(explorer.ExplorerError, match="not the state of a free run"):
        explorer.make_app(path)


@pytest.mark.parametrize(
    "llm", [*(llm for llm in LLMS if llm not in {"offline", "replay"}), "bogus", "", None]
)
def test_a_state_whose_llm_setting_is_no_free_mode_is_refused(tmp_path: Path, llm):
    # An allow-list: the live mode, a mode it does not name and no setting at all.
    path = empty_state(tmp_path, LABEL_STAND_IN, llm=llm)
    with pytest.raises(explorer.ExplorerError, match="not the state of a free run"):
        explorer.make_app(path)


def test_the_free_modes_are_the_two_of_design_2_5():
    assert explorer._FREE_LLMS == ("offline", "replay")
    assert set(explorer._FREE_LLMS) < set(LLMS)


@pytest.mark.parametrize(
    ("llm", "labels"),
    [
        ("offline", ()),
        ("offline", (LABEL_STAND_IN,)),
        ("replay", ()),
        ("replay", (recorded_label("2026-10-05", EXAMPLE_LLM_MODEL),)),
    ],
    ids=["offline, no answers", "offline", "replay, no answers", "replay"],
)
def test_a_free_runs_state_is_served(tmp_path: Path, llm, labels):
    status, _, body = call(explorer.make_app(empty_state(tmp_path, *labels, llm=llm)))
    assert status == "200 OK"
    shown = {LABEL_STAND_IN: "stand-ins"}.get(labels[0], "a recording") if labels else "none"
    assert f'<th scope="row">Answers came from</th><td>{shown}</td>'.encode() in body


def test_a_state_with_no_final_decisions_says_its_run_has_not_finished(tmp_path: Path):
    _, _, body = call(explorer.make_app(empty_state(tmp_path, LABEL_STAND_IN)))
    assert explorer._e(explorer.UNFINISHED_HEADING).encode() in body
    assert explorer._e(explorer.FINISHED_HEADING).encode() not in body


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
    add_answers(path, LABEL_LIVE)
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

    def __init__(self, host, port, app, server_class):
        self.address, self.app, self.server_class = (host, port), app, server_class

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def serve_forever(self):
        raise KeyboardInterrupt


@pytest.fixture
def servers(monkeypatch) -> list[FakeServer]:
    """The servers main would have started, none of them listening."""
    made = []

    def make_server(host, port, app, server_class):
        made.append(FakeServer(host, port, app, server_class))
        return made[-1]

    monkeypatch.setattr(explorer, "make_server", make_server)
    return made


def test_the_start_command_refuses_without_a_state_file(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", pytest.fail)
    assert explorer.main([]) == 1
    assert "no state file at out/state.sqlite" in capsys.readouterr().err


def test_the_start_command_refuses_a_live_state(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "out").mkdir()
    empty_state(tmp_path / "out", LABEL_LIVE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", pytest.fail)
    assert explorer.main([]) == 1
    assert "not the state of a free run" in capsys.readouterr().err


def test_the_start_command_listens_only_on_the_loopback_address(
    tmp_path: Path, monkeypatch, capsys, servers
):
    (tmp_path / "out").mkdir()
    state = empty_state(tmp_path / "out", LABEL_STAND_IN)
    before = hashlib.sha256(state.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    assert explorer.main([]) == 0
    assert [server.address for server in servers] == [("127.0.0.1", 7654)]
    assert servers[0].server_class is explorer.ThreadingServer
    assert explorer.ADDRESS in capsys.readouterr().out
    assert hashlib.sha256(state.read_bytes()).hexdigest() == before


def test_the_start_command_serves_the_output_folder_it_is_given(
    tmp_path: Path, monkeypatch, capsys, servers
):
    folder = tmp_path / "another run"
    folder.mkdir()
    empty_state(folder, LABEL_STAND_IN)
    monkeypatch.chdir(tmp_path)
    assert explorer.main(["--out-dir", str(folder)]) == 0
    assert f"Serving {folder / store.STATE_FILE}," in capsys.readouterr().out
    assert len(servers) == 1


def test_the_start_command_takes_the_settings_of_every_command(capsys):
    with pytest.raises(SystemExit) as stopped:
        explorer.main(["--help"])
    assert stopped.value.code == 0
    shown = capsys.readouterr().out
    assert "--config FILE" in shown and "--out-dir PATH" in shown and "--llm TEXT" in shown


def test_a_bad_command_line_or_setting_exits_2(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(explorer, "make_server", pytest.fail)
    with pytest.raises(SystemExit) as stopped:
        explorer.main(["--no-such-flag"])
    assert stopped.value.code == 2
    assert explorer.main(["--llm", "bogus"]) == 2
    assert "setting llm must be one of" in capsys.readouterr().err


class QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):
        pass


def test_an_idle_connection_holds_up_no_other_request():
    # A browser may open a connection ahead of time and leave it idle.
    def app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"answered\n"]

    server = make_server(
        "127.0.0.1", 0, app, server_class=explorer.ThreadingServer, handler_class=QuietHandler
    )
    serving = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    serving.start()
    try:
        with (
            socket.create_connection(server.server_address, timeout=10),
            socket.create_connection(server.server_address, timeout=10) as asking,
        ):
            asking.sendall(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            with asking.makefile("rb") as answer:
                assert answer.readline().rstrip().endswith(b"200 OK")
    finally:
        server.shutdown()
        server.server_close()
        serving.join(timeout=10)
