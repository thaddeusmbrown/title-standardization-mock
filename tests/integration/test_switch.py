"""The model switch in a fresh interpreter: what a setting loads, and what it leaves alone.

docs/design.md 2.5: "Import the SDK and the embedding library lazily, so the
default run installs nothing." The switch goes one step further (S09): a
provider module is imported only when its setting is chosen. Each test starts
a bare interpreter (no site-packages) behind the import guard of
tests/conftest.py.
"""

from __future__ import annotations

import json

import pytest

from titlestd import datafiles, switch
from titlestd.contracts import ChooseKind, InputData, NumberedRole, TitleRow
from titlestd.settings import Settings

# After the import guard: build the default models, use all three calls once,
# and report which modules of the package were loaded at each point.
DEFAULT_RUN = """
import json

from titlestd import switch

def loaded():
    return sorted(name for name in sys.modules if name.partition(".")[0] == "titlestd")

on_import = loaded()

from titlestd.contracts import (
    Abbreviation, AgentCase, AgentTranscript, CatalogRole, LadderLevel, NumberedRole, Record,
)
from titlestd.settings import Settings

models = switch.build_models(
    Settings(), [LadderLevel("Senior", "Senior", "Sr")], [Abbreviation("mgr", "manager")]
)
record = Record("senior account mgr", "SAL", "Enterprise Sales", "")
roles = (NumberedRole(1, CatalogRole("SAL-01", "SAL", "Account Manager", "Acct Mgr", "")),)
(vector,) = models.embedder.embed(["senior account manager"])
answer = models.llm.chooser.choose(record, roles)
case = AgentCase(record, roles, answer, None, None, ())
turn = models.llm.reviewer.next_turn(AgentTranscript(case))
models.llm.finish()
print(json.dumps({
    "on_import": on_import,
    "after_the_calls": loaded(),
    "attempted": sorted(set(attempted_imports)),
    "buckets": len(vector),
    "answer": [answer.answer, answer.number, answer.quote, answer.source],
    "turn": [turn.tool, turn.query],
}))
"""

# After the import guard: stand recorders in for the three providers that are
# not the default, choose them, and report what was loaded and what was called.
OTHER_SETTINGS = """
import json, types

calls = []
for name in ("local_embedder", "replay", "live"):
    module = types.ModuleType(f"titlestd.{name}")
    module.build_embedder = lambda settings, name=name: calls.append(name) or name
    module.build_llm = lambda settings, fingerprint, name=name: calls.append(name) or name
    sys.modules[module.__name__] = module

from titlestd import switch
from titlestd.settings import Settings

llm = sys.argv[1]
model = "claude-haiku-4-5" if llm == "anthropic" else None
settings = Settings(embedder="local", llm=llm, llm_model=model)
models = switch.build_models(settings, [], [], "fingerprint")
print(json.dumps({
    "models": [models.embedder, models.llm],
    "calls": calls,
    "loaded": sorted(name for name in sys.modules if name.partition(".")[0] == "titlestd"),
    "attempted": sorted(set(attempted_imports)),
}))
"""


def test_importing_the_switch_loads_no_provider(bare_python, import_guard):
    result = bare_python(import_guard + DEFAULT_RUN)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["on_import"] == [
        "titlestd",
        "titlestd.contracts",
        "titlestd.settings",
        "titlestd.switch",
    ]


def test_the_default_models_load_the_two_offline_providers_and_nothing_else(
    bare_python, import_guard
):
    result = bare_python(import_guard + DEFAULT_RUN)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert set(report["after_the_calls"]) - set(report["on_import"]) == {
        "titlestd.offline_embedder",
        "titlestd.standin",
    }
    # Nothing from outside the standard library was even looked for.
    assert report["attempted"] == []
    # And the three calls work there, with nothing installed.
    assert report["buckets"] > 20
    assert report["answer"] == ["role", 1, "account mgr", "stand-in"]
    assert report["turn"] == ["search_catalog", "senior account mgr"]


@pytest.mark.parametrize(("llm", "provider"), [("replay", "replay"), ("anthropic", "live")])
def test_another_setting_never_loads_an_offline_provider(bare_python, import_guard, llm, provider):
    result = bare_python(import_guard + OTHER_SETTINGS, llm)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["models"] == ["local_embedder", provider]
    assert report["calls"] == ["local_embedder", provider]  # the embedder first
    assert "titlestd.switch" in report["loaded"]
    assert not {"titlestd.offline_embedder", "titlestd.standin"} & set(report["loaded"])
    assert report["attempted"] == []


# --- On the mini data set, with the real readers --------------------------------


def test_the_default_models_answer_the_worked_example(
    mini_inputs: InputData, mini_rows: tuple[TitleRow, ...], mini_settings: Settings
):
    # 1.3: "ACCT MGR II" and "Senior Account Manager" are both Sales x Account
    # Manager. The switch is given what the pipeline gives it: the settings,
    # the ladder and the abbreviations as read, and a fingerprint of the inputs.
    models = switch.build_models(
        mini_settings,
        mini_inputs.catalog.ladder,
        mini_inputs.abbreviations,
        datafiles.input_fingerprint(mini_settings.data_dir),
    )
    visible = sorted(
        (role for role in mini_inputs.catalog.roles if not role.is_generic),
        key=lambda role: (role.family_code != "SAL", role.role_id),
    )
    shown = [NumberedRole(number, role) for number, role in enumerate(visible[:10], start=1)]
    by_title = {row.key.cleaned_title: row for row in mini_rows}
    for title in ("account manager ii", "senior account manager"):
        answer = models.llm.chooser.choose(by_title[title].record, shown)
        assert (answer.answer, answer.number, answer.quote) == (
            ChooseKind.ROLE,
            1,
            "account manager",
        )
        assert shown[0].role.role_name == "Account Manager"
    assert models.llm.finish() is None
