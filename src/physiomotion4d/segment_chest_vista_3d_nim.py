"""Module for segmenting chest CT images using VISTA3D."""

# Please start vista3d docker:
#    docker run --rm -it --name vista3d --runtime=nvidia
#      -e CUDA_VISIBLE_DEVICES=0
#      -e NGC_API_KEY=$NGC_API_KEY
#      --shm-size=8G -p 8000:8000
#      -v /tmp/data:/home/aylward/tmp/data nvcr.io/nim/nvidia/vista3d:latest

import argparse
import io
import os
import tempfile
import zipfile

import itk
import numpy as np
import requests

from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D


class SegmentChestVista3DNIM(SegmentChestVista3D):
    """
    A class that inherits from physioSegmentChest and implements the
    segmentation method using VISTA3D.
    """

    def __init__(self):
        """Initialize the vista3d class."""
        super().__init__()

        self.invoke_url = "http://localhost:8000/v1/vista3d/inference"
        self.wsl_docker_tmp_file = (
            "//wsl.localhost/Ubuntu/home/saylward/tmp/data/tmp.nii.gz"
        )
        self.docker_tmp_file = "/tmp/data/tmp.nii.gz"

    def segmentation_method(self, preprocessed_image: itk.image) -> itk.image:
        """
        Run VISTA3D on the preprocessed image using the NIM and return result.

        Args:
            preprocessed_image (itk.image): The preprocessed image to segment.

        Returns:
            the segmented image.
        """

        # Post the image to file.io and get the link
        itk.imwrite(preprocessed_image, self.wsl_docker_tmp_file, compression=True)

        payload = {"image": self.docker_tmp_file, "prompts": {}}

        # Call the API
        session = requests.Session()
        response = session.post(self.invoke_url, json=payload)

        # Check the response
        response.raise_for_status()

        # Get the result
        labelmap_image = None
        with tempfile.TemporaryDirectory() as temp_dir:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(temp_dir)
            file_list = os.listdir(temp_dir)
            for filename in file_list:
                print(filename)
                filepath = os.path.join(temp_dir, filename)
                if os.path.isfile(filepath) and filename.endswith(".nii.gz"):
                    # SUCCESS: Return the results
                    labelmap_image = itk.imread(filepath, pixel_type=itk.SS)
                    break

        if labelmap_image is None:
            raise Exception("Failed to get labelmap image from VISTA3D")

        # HERE
        itk.imwrite(labelmap_image, "vista3d_labelmap.nii.gz", compression=True)

        # Include Soft Tissue
        labelmap_image = self.segment_soft_tissue(preprocessed_image, labelmap_image)

        return labelmap_image


def parse_args():
    """
    Parse command line arguments for Vista3D.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = SegmentChestVista3DNIM().segment(itk.imread(args.fixed_image))
    itk.imwrite(result["labelmap"], args.output_image, compression=True)
