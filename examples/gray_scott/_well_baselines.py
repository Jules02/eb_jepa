"""Shared helper for importing The Well baseline models.

``the_well.benchmark.models.__init__`` eagerly imports every model, including
AFNO/AViT/DilatedResNet/ReFNO which pull in heavy deps (timm, ...). The
Gray-Scott scripts only use FNO/TFNO/UNet*, so we register lightweight stub
modules for the ones we don't need before touching ``the_well``.
"""
import sys
import types

# Models we don't use -> (module path, class name) to stub out.
_STUBS = {
    "the_well.benchmark.models.afno": "AFNO",
    "the_well.benchmark.models.avit": "AViT",
    "the_well.benchmark.models.dilated_resnet": "DilatedResNet",
    "the_well.benchmark.models.refno": "ReFNO",
}


def stub_heavy_well_models():
    """Register stub modules for the unused (heavy-dep) The Well models.

    Idempotent and safe to call before importing FNO/TFNO/UNet* from
    ``the_well.benchmark.models``.
    """
    for mname, cls in _STUBS.items():
        if mname in sys.modules:
            continue
        m = types.ModuleType(mname)
        setattr(m, cls, type(cls, (), {}))
        sys.modules[mname] = m
