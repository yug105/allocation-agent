"""Keep the suite out of the runtime audit log.

Every `create_app()` in the tests wrote to `artifacts/runs.db` -- the real one,
the file the deployed service appends to. Two consequences, both seen:

* **It grew without bound.** The suite makes ~80 demo runs of a few hundred
  records each, and each decision now records the ranked candidate order, so a
  full run appends tens of megabytes to a file nothing ever truncates. It
  reached 901 MB on a development machine.
* **Two runs could not overlap.** A second pytest process -- or a local server
  left running -- takes the same SQLite file and one of them gets
  `database is locked` partway through, which reads as a flaky test rather
  than as two writers.

`AUDIT_DB` is the documented override and `create_app()` already honours it, so
pointing it at a per-session temp file needs no production change. The audit
behaviour under test is unaffected: same class, same triggers, same schema --
only the path differs.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _audit_db_off_the_real_one(tmp_path_factory):
    previous = os.environ.get("AUDIT_DB")
    os.environ["AUDIT_DB"] = str(tmp_path_factory.mktemp("audit") / "runs.db")
    yield
    if previous is None:
        os.environ.pop("AUDIT_DB", None)
    else:
        os.environ["AUDIT_DB"] = previous
