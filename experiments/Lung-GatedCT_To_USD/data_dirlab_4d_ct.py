"""
class pmDataDirLab4dCT:
This module contains the pmDataDirLab4DCT class, which is used to store the
data for the DirLab 4DCT dataset.
"""
import itk


class DataDirLab4DCT:
    """
    This class is used to store the data for the DirLab 4DCT dataset.
    """

    def __init__(self):
        """ Define the variables specific to DirLab data"""
        self.case_names = [
            "Case1Pack",
            "Case2Pack",
            "Case3Pack",
            "Case4Pack",
            "Case5Pack",
            "Case6Pack",
            "Case7Pack",
            "Case8Deploy",
            "Case9Pack",
            "Case10Pack",
        ]

    def get_case_names(self) -> list[str]:
        """ Get the case names """
        return self.case_names

    def fix_image(self, input_image: itk.image) -> itk.image:
        """ Fix DirLab_4DCT intensities to conform to HU """
        input_image_arr = itk.GetArrayViewFromImage(input_image)
        input_image_arr -= 1024
        input_image_arr = input_image_arr.clip(-1024, 3071)
        new_image = itk.GetImageFromArray(input_image_arr)
        new_image.CopyInformation(input_image)

        return new_image