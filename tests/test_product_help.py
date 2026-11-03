from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
import pytest

from lcprop.gui.help import HELP_TOPICS, ProductHelpButton
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


EXPECTED_TOPIC_KEYS = (
    "quick_start",
    "model_choices",
    "beam_focusing",
    "boundary_conditions",
    "fanning_scattering",
    "slurm",
    "runtime_estimates",
    "fast_full",
    "results",
)


def test_help_topics_are_complete_unique_and_deterministic():
    assert tuple(topic.key for topic in HELP_TOPICS) == EXPECTED_TOPIC_KEYS
    assert len({topic.title for topic in HELP_TOPICS}) == len(HELP_TOPICS)
    assert all(topic.markdown.strip() for topic in HELP_TOPICS)


@pytest.mark.parametrize("application", ("lc", "pr"))
def test_help_button_opens_every_topic_without_changing_selection(app, application):
    button = ProductHelpButton(application=application)
    assert button.topic_keys == EXPECTED_TOPIC_KEYS
    assert tuple(action.data() for action in button.menu().actions()) == (
        EXPECTED_TOPIC_KEYS
    )

    for topic in HELP_TOPICS:
        dialog = button.open_topic(topic.key)
        app.processEvents()
        assert dialog.topic is topic
        assert dialog.application == application
        assert topic.title in dialog.windowTitle()
        assert topic.title in dialog.browser.toPlainText()
        dialog.close()
        app.processEvents()

    with pytest.raises(ValueError, match="unknown help topic"):
        button.open_topic("not-a-topic")
    button.close()


def test_lc_and_pr_windows_expose_the_same_non_scientific_help_surface(app):
    lc_window = LCPropMainWindow()
    pr_window = PRMainWindow()

    assert lc_window.help_button.topic_keys == EXPECTED_TOPIC_KEYS
    assert pr_window.help_button.topic_keys == EXPECTED_TOPIC_KEYS
    assert lc_window.build_request().grid.Nx == 64
    assert pr_window.build_request().grid.Nx == 128

    lc_window.close()
    pr_window.close()
