import numpy as np

from halo.models import TrackState
from halo.risk import assess
from halo.trajectory import Conflict
from halo.ttc import ExpansionTTC


def test_expansion_ttc_only_for_growth():
    estimator = ExpansionTTC()
    assert estimator.update(0.0, 100) is None
    assert estimator.update(0.1, 144) is None
    assert estimator.update(0.2, 196) is not None
    assert ExpansionTTC().update(0.0, 200) is None


def test_near_early_conflict_outscores_far_late_conflict():
    state = TrackState("car-1", "car", .95, np.array([-1., 3.]), np.zeros(2), .5, 2.0, True, 1.0)
    urgent = assess(state, Conflict(.3, .5, .9), "left")
    distant = assess(state, Conflict(3.0, 3.5, .3), "left")
    assert urgent.risk > distant.risk
    assert urgent.direction == "left"
