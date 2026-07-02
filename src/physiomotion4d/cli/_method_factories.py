"""Shared string-to-instance factories for CLI segmentation/registration flags.

CLI scripts expose segmentation/registration backends as string choices for
usability, then build the corresponding instance via these factories before
passing it to the library's instance-based workflow API.
"""

from physiomotion4d import (
    RegisterImagesBase,
    RegisterImagesGreedy,
    RegisterImagesGreedyICON,
    RegisterImagesICON,
    SegmentAnatomyBase,
    SegmentChestTotalSegmentator,
    SegmentHeartSimpleware,
    SegmentHeartSimplewareTrimmedBranches,
)

#: Segmentation backend string choices exposed by CLI flags.
SEGMENTATION_METHODS: tuple[str, ...] = (
    "ChestTotalSegmentator",
    "HeartSimpleware",
    "HeartSimplewareTrimmedBranches",
)

#: Registration backend string choices exposed by CLI flags.
REGISTRATION_METHODS: tuple[str, ...] = ("Greedy", "ICON", "Greedy_ICON")


def build_segmentation_method(name: str) -> SegmentAnatomyBase:
    """Build a SegmentAnatomyBase instance for a CLI --segmentation-method choice.

    Args:
        name: One of SEGMENTATION_METHODS.

    Returns:
        A new, unconfigured segmentation backend instance.

    Raises:
        ValueError: If name is not one of SEGMENTATION_METHODS.
    """
    if name == "ChestTotalSegmentator":
        return SegmentChestTotalSegmentator()
    if name == "HeartSimpleware":
        return SegmentHeartSimpleware()
    if name == "HeartSimplewareTrimmedBranches":
        return SegmentHeartSimplewareTrimmedBranches()
    raise ValueError(
        f"Unknown segmentation method: {name}. "
        f"Must be one of: {', '.join(SEGMENTATION_METHODS)}."
    )


def build_registration_method(name: str) -> RegisterImagesBase:
    """Build a RegisterImagesBase instance for a CLI --registration-method choice.

    Args:
        name: One of REGISTRATION_METHODS.

    Returns:
        A new, unconfigured registration backend instance.

    Raises:
        ValueError: If name is not one of REGISTRATION_METHODS.
    """
    if name == "Greedy":
        return RegisterImagesGreedy()
    if name == "ICON":
        return RegisterImagesICON()
    if name == "Greedy_ICON":
        return RegisterImagesGreedyICON()
    raise ValueError(
        f"Unknown registration method: {name}. "
        f"Must be one of: {', '.join(REGISTRATION_METHODS)}."
    )
