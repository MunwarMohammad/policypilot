import json

import numpy as np

from policypilot.ranking import hybrid_order
from policypilot.review import connection, enqueue


def test_lexical_evidence_changes_dense_order():
    order = hybrid_order(
        "XYZ123", ["unrelated", "unrelated", "XYZ123"], np.array([0.51, 0.50, 0.49])
    )
    assert order[0] == 2


def test_review_survives_reopen(tmp_path):
    ticket = enqueue(tmp_path, "AAPL", "Explain expenses", {"hits": []})
    with connection(tmp_path) as db:
        status, evidence = db.execute(
            "SELECT status,evidence FROM review WHERE id=?", (ticket,)
        ).fetchone()
        assert status == "pending" and json.loads(evidence) == []
