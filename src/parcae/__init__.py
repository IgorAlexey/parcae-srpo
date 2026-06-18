"""
Recurrent-Depth Gemma 4; retrofit a pretrained N-layer model
into a middle-looped recurrent-depth transformer (Parcae, 2026).

Tested configuration: Gemma 4 E2B (35 layers → 12+11+12 split).

Architecture:
    Prelude (first 12 layers) → run once
    Recurrent (middle 11 layers) → looped T times with LTI-stable injection
    Coda (last 12 layers) → run once

Reference papers:
    - Parcae: Scaling Laws For Stable Looped Language Models (arxiv 2604.12946)
    - OpenMythos: Reverse-engineered Mythos-class recurrent-depth (Kye Gomez, 2026)
    - Kohli et al., Loop, Think, & Generalize (arxiv 2604.07822)
"""

from .injection import LTIInjection
from .model import RecurrentDepthGemma, RecurrentDepthConfig

__all__ = ["LTIInjection", "RecurrentDepthGemma", "RecurrentDepthConfig"]
