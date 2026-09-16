from __future__ import annotations

import importlib
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PRODUCT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "STATUS.md",
    ROOT / "docs" / "user" / "quick_start.md",
    ROOT / "docs" / "user" / "user_guide.md",
    ROOT / "docs" / "science" / "pr_model_contracts.md",
    ROOT / "src" / "lcprop" / "pr" / "README.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def test_current_product_documentation_has_no_broken_relative_links():
    failures = []
    for document in CURRENT_PRODUCT_DOCS:
        for raw_target in MARKDOWN_LINK.findall(
            document.read_text(encoding="utf-8")
        ):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"{document.relative_to(ROOT)} -> {raw_target}"
                )
    assert failures == []


def test_quick_start_entry_points_match_the_product():
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["requires-python"] == ">=3.10"
    assert project["project"]["scripts"]["lcprop-pr"] == (
        "lcprop.pr.gui.app:main"
    )
    assert callable(importlib.import_module("lcprop.pr.gui.app").main)
    assert callable(importlib.import_module("lcprop.lc.gui.app").main)


def test_pr_user_matrix_names_all_eight_production_cells_once():
    guide = (ROOT / "docs/user/user_guide.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (Static|Time dependent) \| "
        r"(Reduced x-only|Full transverse) \| "
        r"(Fully nonlinear|Linearized) \|",
        guide,
        flags=re.MULTILINE,
    )
    assert len(rows) == 8
    assert len(set(rows)) == 8


def test_current_pr_contract_replaces_historical_product_dependencies():
    runtime_source = (
        ROOT / "src/lcprop/pr/transverse/linearized_reference.py"
    ).read_text(encoding="utf-8")
    implementation_record = (
        ROOT
        / "docs/development/pr_biased_full_transverse_linearized_reference_solver.md"
    ).read_text(encoding="utf-8")

    assert "docs/science/pr_model_contracts.md" in runtime_source
    assert "docs/science/pr_model_contracts.md" in implementation_record
    assert "docs/research/" not in runtime_source
    assert "docs/research/" not in implementation_record
