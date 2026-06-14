"""Shared tripinfo parsing for baselines, evaluation and the pilot benchmark."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np


def parse_tripinfo_means(path: str | Path) -> Dict[str, float]:
    """Parse a SUMO --tripinfo-output file into network-level mean metrics.

    Returns keys:
        n_trips            finished (or written-unfinished) trips
        mean_travel_time_s mean tripinfo 'duration'
        mean_waiting_time_s mean tripinfo 'waitingTime'
        mean_time_loss_s   mean tripinfo 'timeLoss'
        p95_waiting_time_s 95th percentile waiting time (starvation indicator)
        total_co2_mg       sum of <emissions CO2_abs> over trips (NaN if the
                           emissions device was not enabled at run time)
        total_fuel_mg      sum of <emissions fuel_abs> over trips (NaN if absent)
    """
    path = Path(path)
    durations, waits, losses = [], [], []
    co2_vals, fuel_vals = [], []
    for trip in ET.parse(path).getroot().iter("tripinfo"):
        durations.append(float(trip.get("duration", 0.0)))
        waits.append(float(trip.get("waitingTime", 0.0)))
        losses.append(float(trip.get("timeLoss", 0.0)))
        em = trip.find("emissions")
        if em is not None:
            co2_vals.append(float(em.get("CO2_abs", 0.0)))
            fuel_vals.append(float(em.get("fuel_abs", 0.0)))

    if not durations:
        return {
            "n_trips": 0,
            "mean_travel_time_s": float("nan"),
            "mean_waiting_time_s": float("nan"),
            "mean_time_loss_s": float("nan"),
            "p95_waiting_time_s": float("nan"),
            "total_co2_mg": float("nan"),
            "total_fuel_mg": float("nan"),
        }

    return {
        "n_trips": len(durations),
        "mean_travel_time_s": float(np.mean(durations)),
        "mean_waiting_time_s": float(np.mean(waits)),
        "mean_time_loss_s": float(np.mean(losses)),
        "p95_waiting_time_s": float(np.percentile(waits, 95)),
        # emissions are post-hoc only (never in the reward); NaN when the SUMO
        # emissions device was not active for this run.
        "total_co2_mg": float(np.sum(co2_vals)) if co2_vals else float("nan"),
        "total_fuel_mg": float(np.sum(fuel_vals)) if fuel_vals else float("nan"),
    }


__all__ = ["parse_tripinfo_means"]
