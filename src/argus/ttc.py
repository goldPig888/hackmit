from __future__ import annotations

from collections import deque


class ExpansionTTC:
    """Smoothed monocular TTC estimate from apparent bounding-box area."""

    def __init__(self, window: int = 8, minimum_rate: float = 1e-3) -> None:
        self.samples: deque[tuple[float, float]] = deque(maxlen=window)
        self.minimum_rate = minimum_rate

    def update(self, timestamp_s: float, area_px2: float) -> float | None:
        self.samples.append((timestamp_s, max(area_px2, 1.0)))
        if len(self.samples) < 3:
            return None
        first_t, first_a = self.samples[0]
        last_t, last_a = self.samples[-1]
        dt = last_t - first_t
        if dt <= 0:
            return None
        # Area grows roughly with inverse distance squared, so TTC≈2A/A_dot.
        rate = (last_a - first_a) / dt
        if rate <= self.minimum_rate:
            return None
        return max(0.05, min(30.0, 2.0 * last_a / rate))
