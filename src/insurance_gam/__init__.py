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

Missing extras
--------------
If you see an ImportError when importing a subpackage, install the relevant extra:

    pip install insurance-gam[ebm]     # for insurance_gam.ebm (requires interpret)
    pip install insurance-gam[neural]  # for insurance_gam.anam and insurance_gam.pin (requires torch)
    pip install insurance-gam[all]     # everything
"""

__version__ = "0.1.6"
__all__ = ["ebm", "anam", "pin", "__version__"]
