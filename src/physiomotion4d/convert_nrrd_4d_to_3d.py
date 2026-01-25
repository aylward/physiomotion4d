import logging
from typing import Any, Optional

import itk
import nrrd
import numpy as np

from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase


class ConvertNRRD4DTo3D(PhysioMotion4DBase):
    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the NRRD 4D to 3D converter.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.nrrd_4d: Optional[tuple[Any, dict[str, Any]]] = None
        self.img_3d: list[Any] = []

    def load_nrrd_3d(self, filenames: list[str]) -> None:
        self.img_3d = []
        for filename in filenames:
            img_3d = itk.imread(filename)
            self.img_3d.append(img_3d)

    def load_nrrd_4d(self, filename: str) -> None:
        # The nrrd sequence is a 4D file which is not supported by itk, babbel, or other readers.
        # We must use the pynrrd reader.
        self.nrrd_4d = nrrd.read(filename)
        header = self.nrrd_4d[1]

        # The pynrrd reader returns a list of tuples, where the first element is the image data and the second element is the header.

        # Extract the origin, spacing, and direction from the header
        origin = np.array(header["space origin"])
        spacing = np.array(
            [abs(header["space directions"][x + 1][x]) for x in range(3)]
        )
        direction = np.array([header["measurement frame"][x] for x in range(3)])
        if "right" in header["space"]:
            direction[0][0] = -1 * direction[0][0]
        if "anterior" in header["space"]:
            direction[1][1] = -1 * direction[1][1]
        if "inferior" in header["space"]:
            direction[2][2] = -1 * direction[2][2]

        self.img_3d = []
        for t in range(self.nrrd_4d[0].shape[0]):
            img4d_arr = np.array(self.nrrd_4d[0])

            # The pynrrd reader returns the image in the order (t,x,y,z)
            # We need to convert it to (z,y,x) for the itk writer
            img3d_arr = img4d_arr[t, :, :, :].transpose(2, 1, 0)

            self.img_3d.append(itk.image_from_array(img3d_arr))
            self.img_3d[-1].SetOrigin(origin)
            self.img_3d[-1].SetSpacing(spacing)
            self.img_3d[-1].SetDirection(direction)

    def get_3d_image(self, index: int) -> Any:
        return self.img_3d[index]

    def get_number_of_3d_images(self) -> int:
        return len(self.img_3d)

    def save_3d_images(self, basename: str) -> None:
        for i in range(self.get_number_of_3d_images()):
            itk.imwrite(self.img_3d[i], f"{basename}_{i:03d}.mha", compression=True)
