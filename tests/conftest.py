"""Shared fixtures.

Tests never read the model in models/: that file is whatever the last
`scripts/train.py` run produced. A small synthetic model is trained into a
temp dir instead, so results don't depend on local state.
"""

from __future__ import annotations

import pytest

from zakkend import config


@pytest.fixture(scope="session", autouse=True)
def _isolated_model(tmp_path_factory):
    from zakkend.data.synthetic import generate
    from zakkend.models.baseline import train

    original = config.MODEL_PATH
    config.MODEL_PATH = tmp_path_factory.mktemp("models") / "subsidence_xgb.ubj"
    train(generate(n=2_000)).save()
    yield
    config.MODEL_PATH = original
