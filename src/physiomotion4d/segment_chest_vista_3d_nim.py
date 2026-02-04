"""Module for segmenting chest CT images using VISTA3D."""

# Please start vista3d docker:
#    docker run --rm -it --name vista3d --runtime=nvidia
#      -e CUDA_VISIBLE_DEVICES=0
#      -e NGC_API_KEY=$NGC_API_KEY
#      --shm-size=8G -p 8000:8000
#      -v /tmp/data:/home/aylward/tmp/data nvcr.io/nim/nvidia/vista3d:latest

import io
import json
import logging
import os
import socket
import tempfile
import zipfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import itk

from physiomotion4d.segment_chest_vista_3d import SegmentChestVista3D


class SegmentChestVista3DNIM(SegmentChestVista3D):
    """
    A class that inherits from physioSegmentChest and implements the
    segmentation method using VISTA3D.
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the vista3d class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(log_level=log_level)

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

        # Call the API (stdlib HTTP client; avoids needing requests stubs)
        payload_bytes = json.dumps(payload).encode("utf-8")
        req = Request(
            self.invoke_url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # Use timeout to prevent indefinite hanging (300s = 5 minutes)
            with urlopen(req, timeout=300) as resp:
                response_content = resp.read()
        except socket.timeout as e:
            raise RuntimeError("VISTA3D NIM request timed out after 300 seconds") from e
        except HTTPError as e:
            raise RuntimeError(
                f"VISTA3D NIM request failed: HTTP {e.code} {e.reason}"
            ) from e
        except URLError as e:
            raise RuntimeError(f"VISTA3D NIM request failed: {e.reason}") from e

        # Get the result
        labelmap_image = None
        with tempfile.TemporaryDirectory() as temp_dir:
            z = zipfile.ZipFile(io.BytesIO(response_content))
            z.extractall(temp_dir)
            file_list = os.listdir(temp_dir)
            for filename in file_list:
                self.log_debug("Found file: %s", filename)
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
