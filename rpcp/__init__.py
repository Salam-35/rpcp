"""R-PCP: Reliability-Weighted Concept Learning from Noisy Class-Level Priors.

Reference implementation of the paper plan:

* :mod:`rpcp.datasets` -- datasets, class-level prior tables, controlled corruption;
* :mod:`rpcp.models` -- backbone, concept predictor, reliability module;
* :mod:`rpcp.losses` -- ``L_cls``, ``L_match``, ``L_prior``, ``L_ent``, ``R(r)``;
* :mod:`rpcp.training` -- phased trainer, cross-fitting, class means;
* :mod:`rpcp.evaluation` -- concept / class / reliability metrics, ``Delta``;
* :mod:`rpcp.plotting` -- Figures 1-5.

Quick start::

    from rpcp.config import load_config
    from rpcp.training import run_experiment

    result = run_experiment(load_config("configs/synthetic.yaml"))
    print(result.summary())
"""

from __future__ import annotations

from rpcp.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "__version__", "load_config"]

__version__ = "0.1.0"
