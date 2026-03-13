"""
insurance-gam — interpretable GAM toolkit for insurance pricing.

Three subpackages, one install:

    from insurance_gam.ebm import InsuranceEBM        # interpretML EBM wrapper
    from insurance_gam.anam import ANAM               # Actuarial Neural Additive Model
    from insurance_gam.pin import PINModel            # Pairwise Interaction Networks

Each subpackage is independent. Import only the one you need.
Heavy dependencies (torch, interpret) are only loaded when the subpackage is imported.

Do NOT import this top-level package expecting all subpackages to be available —
use subpackage imports directly. The top-level package exposes only the version.
"""

__version__ = "0.1.0"
__all__ = ["ebm", "anam", "pin", "__version__"]
