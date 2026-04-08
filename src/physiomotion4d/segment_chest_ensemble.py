"""Module for ensemble chest CT segmentation.

Currently delegates to TotalSegmentator.  The ensemble class is retained as a
public API entry point so that callers do not need to change their imports.
"""

import logging

from physiomotion4d.segment_chest_total_segmentator import SegmentChestTotalSegmentator


class SegmentChestEnsemble(SegmentChestTotalSegmentator):
    """Ensemble chest CT segmentation.

    Inherits from :class:`SegmentChestTotalSegmentator` and currently delegates
    all segmentation to that backend.  The class is kept as a stable public
    name so that downstream code depending on ``SegmentChestEnsemble`` continues
    to work without modification.

    Args:
        log_level: Logging level (default: ``logging.INFO``).

    Example:
        >>> segmenter = SegmentChestEnsemble()
        >>> result = segmenter.segment(ct_image, contrast_enhanced_study=False)
        >>> labelmap = result['labelmap']
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        super().__init__(log_level=log_level)
