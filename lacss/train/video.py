from __future__ import annotations

import jax
import optax
from ml_collections import ConfigDict

from ..losses import binary_focal_crossentropy
from ..ops import match_and_replace, distance_similarity
from ..typing import ArrayLike
from ..utils import deep_update

def train_fn(
    module,
    image: ArrayLike,
    gt_locations: ArrayLike,
    image_mask: ArrayLike|None = None,
    config: ConfigDict = ConfigDict(),
):
    pass
