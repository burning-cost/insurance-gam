"""
insurance-gam — interpretable GAM toolkit for insurance pricing.

Three subpackages, one install:

    from insurance_gam.ebm import InsuranceEBM        # interpretML EBM wrapper
    from insurance_gam.anam import ANAM               # Actuarial Neural Additive Model
    from insurance_gam.pin import PINModel            # Pairwise Interaction Networks

Each subpackage is independent. You can use any one without importing the others.
Heavy dependencies (torch, interpret) are only loaded when the subpackage is imported.
"""

from insurance_gam import ebm, anam, pin

__version__ = "0.1.0"
__all__ = ["ebm", "anam", "pin"]
