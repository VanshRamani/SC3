"""
External baselines: methods integrated into the SC³ SDK that follow the
``BaseMethod`` interface defined in ``base.py``.

These are baselines whose training procedure is bespoke enough that we keep
each implementation self-contained rather than folding it into the unified
``train.py`` dispatcher used for tree / descriptor-NN / GNN methods.

A method here is registered in :mod:`sc3_bench.registry` under
``model_type='external'``. The dispatcher in :mod:`sc3_bench.train` then
constructs the method via the symbol exported below and follows the
``BaseMethod.fit`` / ``BaseMethod.predict`` protocol.

Methods exposed:
    * ``chemprop`` — D-MPNN baseline (CLI wrapper around chemprop v1.x);
                     functional API, see :mod:`chemprop_method`.
    * ``soltrannet`` — Molecule-Attention-Transformer (SolTranNet) dual encoder.
    * ``unimol`` — Uni-Mol2 frozen-feature MLP head.
    * ``unimol_catboost`` — Uni-Mol2 frozen features + CatBoost head.
    * ``unifac`` — Group-contribution UNIFAC residual + CatBoost correction.
    * ``solvaformer`` — SE(3)-equivariant graph transformer (PaiNN-style backbone).
    * ``rilood`` — RIL-OOD model (relational invariant learning for OOD).

For ``solvaformer``, ``rilood``, and ``chemprop`` the implementations expose
torch model classes / functional drivers but no ``BaseMethod`` wrapper —
those are intended to be invoked directly from a method-specific training
script (see SDK README).
"""

from .base import BaseMethod
from .soltrannet import SolTranNetMethod
from .unimol_method import UniMolMethod, UniMolCatBoostMethod
from .unifac_method import UNIFACMLModel

__all__ = [
    "BaseMethod",
    "SolTranNetMethod",
    "UniMolMethod",
    "UniMolCatBoostMethod",
    "UNIFACMLModel",
]
