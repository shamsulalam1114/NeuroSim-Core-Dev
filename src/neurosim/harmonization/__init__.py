"""
neurosim.harmonization — Data Harmonization via Blind neuroCombat.

This module implements the harmonization strategy described in Module A of the
NeuroSim proposal. The Blind neuroCombat approach preserves biological variance
(disease signal) while removing scanner-induced site effects.

Pipeline:
    1. Load a reference cohort (e.g., HCP Healthy Controls).
    2. Fit ComBat parameters on the reference cohort ONLY (blind to pathology).
    3. Apply fitted parameters to clinical cohorts (ADNI, AUD, Epilepsy).

This ensures that the scanner effect model is learned exclusively from healthy
subjects, preventing pathological signal from contaminating the harmonization.
"""

from neurosim.harmonization.combat import (
    fit_combat,
    apply_combat,
    blind_harmonize,
)

__all__ = [
    "fit_combat",
    "apply_combat",
    "blind_harmonize",
]
