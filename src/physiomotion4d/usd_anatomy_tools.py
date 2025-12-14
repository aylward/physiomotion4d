"""
This module contains the USDAnatomyTools class, which is used to enhance
the anatomy meshes in a USD file.
"""

import argparse
import os

from pxr import Sdf, Usd, UsdGeom, UsdShade


class USDAnatomyTools:
    """
    This class is used to enhance the appearance of anatomy meshes in a USD
    file.
    """

    def __init__(self, stage):
        self.stage = stage

        self.heart_params = {
            "name": "Heart",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.25,
            "diffuse_reflection_color": (0.2, 0.01, 0.01),
            "diffuse_reflection_roughness": 0.5,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.8, 0.4, 0.4),
            "subsurface_scattering_color": (0.8, 0.4, 0.4),
            "subsurface_weight": 0.1,
            "subsurface_scale": 2.0,
            "coat_weight": 0.0,
        }
        self.lung_params = {
            "name": "Lung",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.125,
            "diffuse_reflection_color": (0.34, 0.0, 0.0),
            "diffuse_reflection_roughness": 0.6,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.9, 0.7, 0.7),
            "subsurface_scattering_color": (0.9, 0.7, 0.7),
            "subsurface_weight": 0.1,
            "subsurface_scale": 0.2,
            "coat_weight": 0.0,
        }
        self.bone_params = {
            "name": "Bone",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.2,
            "diffuse_reflection_color": (0.8, 0.8, 0.9),
            "diffuse_reflection_roughness": 0.3,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.95, 0.9, 0.7),
            "subsurface_scattering_color": (0.95, 0.9, 0.7),
            "subsurface_weight": 0.03,
            "subsurface_scale": 0.05,
            "coat_weight": 0.0,
        }
        self.major_vessels_params = {
            "name": "Major_Vessels",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.35,
            "diffuse_reflection_color": (0.2, 0.01, 0.01),
            "diffuse_reflection_roughness": 0.3,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.7, 0.15, 0.18),
            "subsurface_scattering_color": (0.7, 0.15, 0.18),
            "subsurface_weight": 0.09,
            "subsurface_scale": 0.22,
            "coat_weight": 0.12,
        }
        self.contrast_params = {
            "name": "Contrast",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.25,
            "diffuse_reflection_color": (0.2, 0.01, 0.01),
            "diffuse_reflection_roughness": 0.5,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.9, 0.4, 0.4),
            "subsurface_scattering_color": (0.9, 0.4, 0.4),
            "subsurface_weight": 0.1,
            "subsurface_scale": 2.0,
            "coat_weight": 0.0,
        }
        self.soft_tissue_params = {
            "name": "Soft_Tissue",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.17,
            "diffuse_reflection_color": (0.7, 0.5, 0.4),
            "diffuse_reflection_roughness": 0.5,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.95, 0.9, 0.9),
            "subsurface_scattering_color": (0.95, 0.9, 0.9),
            "subsurface_weight": 0.08,
            "subsurface_scale": 0.3,
            "coat_weight": 0.12,
        }
        self.other_params = {
            "name": "Other",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.1,
            "diffuse_reflection_color": (0.7, 0.5, 0.4),
            "diffuse_reflection_roughness": 0.5,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.95, 0.5, 0.4),
            "subsurface_scattering_color": (0.95, 0.5, 0.4),
            "subsurface_weight": 0.1,
            "subsurface_scale": 0.3,
            "coat_weight": 0.12,
        }
        self.liver_params = {
            "name": "Liver",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.1,
            "diffuse_reflection_color": (0.16, 0.01, 0.01),
            "diffuse_reflection_roughness": 0.4,
            "metalness": 0.0,
            "specular_reflection_weight": 0.01,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.7, 0.2, 0.15),
            "subsurface_scattering_color": (0.7, 0.2, 0.15),
            "subsurface_weight": 0.08,
            "subsurface_scale": 0.15,
            "coat_weight": 0.01,
        }
        self.spleen_params = {
            "name": "Spleen",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.1,
            "diffuse_reflection_color": (0.45, 0.08, 0.15),
            "diffuse_reflection_roughness": 0.4,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.6, 0.15, 0.22),
            "subsurface_scattering_color": (0.6, 0.15, 0.22),
            "subsurface_weight": 0.08,
            "subsurface_scale": 0.15,
            "coat_weight": 0.1,
        }
        self.kidney_params = {
            "name": "Kidney",
            "enable_diffuse_transmission": True,
            "diffuse_reflection_weight": 0.1,
            "diffuse_reflection_color": (0.45, 0.13, 0.12),
            "diffuse_reflection_roughness": 0.35,
            "metalness": 0.0,
            "specular_reflection_weight": 0.5,
            "specular_reflection_roughness": 0.5,
            "subsurface_transmission_color": (0.6, 0.18, 0.15),
            "subsurface_scattering_color": (0.6, 0.18, 0.15),
            "subsurface_weight": 0.085,
            "subsurface_scale": 0.18,
            "coat_weight": 0.1,
        }

    def _apply_surgical_materials(self, prim, material_params):
        """Corrected material application with Omniverse-specific fixes"""

        # 1. Unique material path using prim's full path hierarchy
        material_path = Sdf.Path(f"/World/Looks/OmniSurface_{material_params['name']}")
        shader_path = material_path.AppendPath("Shader")

        material = UsdShade.Material.Get(self.stage, material_path)
        if not material or not material.GetPrim().IsValid():
            # prim.CreateDisplayColorAttr().Set(material_params["diffuse_reflection_color"])

            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, shader_path)

            # 2. MDL-Context Shader Definition (REQUIRED for Omniverse)
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset("OmniSurface.mdl", "mdl")
            shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")

            # 3. Set the parameters
            shader.CreateInput(
                "enable_diffuse_transmission", Sdf.ValueTypeNames.Bool
            ).Set(material_params["enable_diffuse_transmission"])
            shader.CreateInput(
                "diffuse_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_weight"])
            shader.CreateInput(
                "diffuse_reflection_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["diffuse_reflection_color"])
            shader.CreateInput(
                "diffuse_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["diffuse_reflection_roughness"])
            shader.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(
                material_params["metalness"]
            )
            shader.CreateInput(
                "specular_reflection_weight", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_weight"])
            shader.CreateInput(
                "specular_reflection_roughness", Sdf.ValueTypeNames.Float
            ).Set(material_params["specular_reflection_roughness"])
            shader.CreateInput(
                "subsurface_transmission_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_transmission_color"])
            shader.CreateInput(
                "subsurface_scattering_color", Sdf.ValueTypeNames.Color3f
            ).Set(material_params["subsurface_scattering_color"])
            shader.CreateInput("subsurface_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_weight"]
            )
            shader.CreateInput("subsurface_scale", Sdf.ValueTypeNames.Float).Set(
                material_params["subsurface_scale"]
            )
            shader.CreateInput("coat_weight", Sdf.ValueTypeNames.Float).Set(
                material_params["coat_weight"]
            )

            # 4. Connect the shader's output to the material's surface output for the MDL render context.
            material.CreateSurfaceOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )
            material.CreateDisplacementOutput("mdl").ConnectToSource(
                shader.ConnectableAPI(), "out"
            )

        # 5. Bind the material to the mesh primitive.
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material)

    def enhance_meshes(self, segmentator):
        """Find and enhance all heart meshes"""

        heart_mask_ids = list(segmentator.heart_mask_ids.values())
        major_vessels_mask_ids = list(segmentator.major_vessels_mask_ids.values())
        lung_mask_ids = list(segmentator.lung_mask_ids.values())
        bone_mask_ids = list(segmentator.bone_mask_ids.values())
        contrast_mask_ids = list(segmentator.contrast_mask_ids.values())
        soft_tissue_mask_ids = list(segmentator.soft_tissue_mask_ids.values())
        other_mask_ids = list(segmentator.other_mask_ids.values())

        liver_mask_ids = ["liver"]
        spleen_mask_ids = ["spleen"]
        kidney_mask_ids = ["kidney_right", "kidney_left"]

        # Safely remove items if they exist
        for item in ["liver", "spleen", "kidney_right", "kidney_left"]:
            if item in soft_tissue_mask_ids:
                soft_tissue_mask_ids.remove(item)

        list_of_mask_ids = [
            heart_mask_ids,
            major_vessels_mask_ids,
            lung_mask_ids,
            bone_mask_ids,
            contrast_mask_ids,
            soft_tissue_mask_ids,
            other_mask_ids,
            liver_mask_ids,
            spleen_mask_ids,
            kidney_mask_ids,
        ]
        list_of_params = [
            self.heart_params,
            self.major_vessels_params,
            self.lung_params,
            self.bone_params,
            self.contrast_params,
            self.soft_tissue_params,
            self.other_params,
            self.liver_params,
            self.spleen_params,
            self.kidney_params,
        ]

        editor = Usd.NamespaceEditor(self.stage)
        prims = []
        for prim in self.stage.Traverse():
            prims.append(prim)
        for prim in prims:
            prim_name = str(prim.GetPrimPath())
            prim_name = prim_name.split("/")[-1]
            prim_sub_name = "_".join(prim_name.split("_")[1:])

            anatomy_params = None
            anatomy_prim_found = False
            for mask_ids, params in zip(list_of_mask_ids, list_of_params):
                if prim_name in mask_ids or prim_sub_name in mask_ids:
                    anatomy_params = params
                    anatomy_prim_found = True
                    break

            if anatomy_prim_found:
                mesh_prim = UsdGeom.Mesh(prim)
                transform_prim = UsdGeom.Xform(prim)
                if transform_prim and not mesh_prim:
                    current_prim_path = str(prim.GetPrimPath())
                    root_prim_path = "/".join(current_prim_path.split("/")[:-1])
                    print(f"Root prim path: {root_prim_path}")
                    anatomy_prim_path = "/".join([root_prim_path, "Anatomy"])
                    print(f"   Anatomy prim path: {anatomy_prim_path}")
                    if not self.stage.GetPrimAtPath(anatomy_prim_path):
                        UsdGeom.Xform.Define(self.stage, anatomy_prim_path)
                    anatomy_prim_path = "/".join(
                        [root_prim_path, "Anatomy", f"Group_{anatomy_params['name']}"]
                    )
                    if not self.stage.GetPrimAtPath(anatomy_prim_path):
                        UsdGeom.Xform.Define(self.stage, anatomy_prim_path)
                    remaining_prim_path = prim.GetName()
                    anatomy_prim_path = "/".join(
                        [
                            root_prim_path,
                            "Anatomy",
                            f"Group_{anatomy_params['name']}",
                            remaining_prim_path,
                        ]
                    )
                    anatomy_prim_path = Sdf.Path(anatomy_prim_path)
                    print(f"   Current prim path: {current_prim_path}")
                    print(f"   Anatomy prim path: {anatomy_prim_path}")
                    editor.MovePrimAtPath(
                        current_prim_path,
                        anatomy_prim_path,
                    )
                    editor.ApplyEdits()

        for prim in self.stage.Traverse():
            prim_name = str(prim.GetPrimPath())
            prim_name = prim_name.split("/")[-1]
            prim_sub_name = "_".join(prim_name.split("_")[1:])

            anatomy_params = None
            anatomy_prim_found = False
            for mask_ids, params in zip(list_of_mask_ids, list_of_params):
                if prim_name in mask_ids or prim_sub_name in mask_ids:
                    anatomy_params = params
                    anatomy_prim_found = True
                    break

            if anatomy_prim_found:
                mesh_prim = UsdGeom.Mesh(prim)
                if mesh_prim:
                    self._apply_surgical_materials(prim, anatomy_params)


def parse_args():
    parser = argparse.ArgumentParser(description="Paint anatomy meshes in a USD file")
    parser.add_argument("usd_file_path", type=str, help="Path to the USD file")
    parser.add_argument(
        "mask_ids", type=int, nargs="+", help="IDs of the heart meshes to paint"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stage = Usd.Stage.Open(args.usd_file_path)
    painter = USDAnatomyTools(stage)
    painter.enhance_meshes(seg)
    filename = os.path.basename(args.usd_file_path).split(".")[0]
    stage.GetRootLayer().Save(f"{filename}_painted.usda")

