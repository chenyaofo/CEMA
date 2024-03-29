from .register import TTADAPTER

from .base import CloudEdgeTestTimeAdaptation
from .bn_adapt import BNAdaptation
from .ptn import PredictionBatchNorm
from .tbr import TBR
from .tema import TEMA
from .tent import Tent
from .distillation_tent import DistillationTent
from .distillation_tbr import DistillationTBR
from .selective_tent import SelectiveTent
from .pl import PL
from .cotta import CoTTA
from .lame import LAME
from .eata import EATA
from.eata_plus import EATAPlus
from .cetta import CETTA
from .cetta_clip import CETTA_CLIP

TTADAPTER.register(CloudEdgeTestTimeAdaptation)
TTADAPTER.register(BNAdaptation)
TTADAPTER.register(PredictionBatchNorm)
TTADAPTER.register(TBR)
TTADAPTER.register(TEMA)
TTADAPTER.register(Tent)
TTADAPTER.register(DistillationTent)
TTADAPTER.register(DistillationTBR)
TTADAPTER.register(SelectiveTent)
TTADAPTER.register(PL)
TTADAPTER.register(CoTTA)
TTADAPTER.register(LAME)
TTADAPTER.register(EATA)
TTADAPTER.register(EATAPlus)
TTADAPTER.register(CETTA)
TTADAPTER.register(CETTA_CLIP)
