"""Crowd density labels and history-based five-minute estimates."""

from datetime import datetime


def classify_density(
    people_count: int,
    moderate_max: int = 10,
    high_max: int = 25,
    critical_min: int = 40,
) -> str:
    """Classify a count using the configured upper bounds for each level."""
    if people_count <= moderate_max:
        return "Low"
    if people_count <= high_max:
        return "Moderate"
    if people_count < critical_min:
        return "High"
    return "Critical"


def predict_crowd_trend(
    history: list[dict], horizon_minutes: int = 5
) -> dict[str, int | str | None]:
    """Linearly extrapolate recent real counts; return no estimate if history is short."""
    if len(history) < 3:
        return {"trend": "Collecting data", "estimated_count": None}

    recent = history[-60:]
    times = [datetime.fromisoformat(row["timestamp"]).timestamp() for row in recent]
    counts = [int(row["people_count"]) for row in recent]
    elapsed = times[-1] - times[0]
    if elapsed < 10:
        return {"trend": "Collecting data", "estimated_count": None}

    mean_time = sum(times) / len(times)
    mean_count = sum(counts) / len(counts)
    variance = sum((time_value - mean_time) ** 2 for time_value in times)
    if variance == 0:
        return {"trend": "Collecting data", "estimated_count": None}

    slope = sum(
        (time_value - mean_time) * (count - mean_count)
        for time_value, count in zip(times, counts)
    ) / variance
    estimate = max(0, round(counts[-1] + slope * horizon_minutes * 60))
    difference = estimate - counts[-1]

    if difference > 2:
        trend = "Increasing"
    elif difference < -2:
        trend = "Decreasing"
    else:
        trend = "Stable"

    return {"trend": trend, "estimated_count": estimate}