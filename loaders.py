import argparse
import hashlib
import os
import sys
import re

import folder_paths

def add_search_paths():
    # Set the configs folder
    custom_nodes_folders, _ = folder_paths.folder_names_and_paths["custom_nodes"]

    cartoon_seg_folders = {}

    # custom_nodes folder is an array.  This code assumes that there be only one instance
    custom_node_folder = custom_nodes_folders[0].rstrip("/").rstrip("\\")
    cartoon_segmentation_root = f"{custom_node_folder}/ComfyUI_CartoonSegmentation/CartoonSegmentation"
    cartoon_seg_config_path = f"{cartoon_segmentation_root}/configs"
    model_path_ani_inst_ckpt = f"{cartoon_segmentation_root}/models/AnimeInstanceSegmentation"
    model_path_ani_seg_ckpt = f"{cartoon_segmentation_root}/models/anime-seg"
    leres_path = f"{cartoon_segmentation_root}/models/leres"

    if "cartoon_segmentation" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["cartoon_segmentation"] = cartoon_seg_folders
    else:
        cartoon_seg_folders = folder_paths.folder_names_and_paths["cartoon_segs_configs"]

    # End Goal is to have three new entries for Paths.  Configs and models in ComfyUI searchable structures and
    # a third key that contains a dictionary for folders that have hard references

    def check_report_path(path, variable_target):
        if path_exists := not os.path.exists(path):
            print(f"Warning Excepted Path {variable_target} "
                  f"does not exist when loading ComfyUI_CartoonSegmentation\n"
                  f"Path::{path}")
        return path_exists

    if not check_report_path(custom_node_folder, "custom node root folder"):
        cartoon_seg_folders["custom_node_root"] = custom_node_folder

    if not check_report_path(cartoon_seg_config_path, "Cartoon Segmentation Configs"):
        folder_paths.folder_names_and_paths["cartoon_seg_configs"] = ([custom_node_folder], {'.yaml'})

    if not check_report_path(model_path_ani_inst_ckpt, "Cartoon Segmentation checkpoints AnimeInstanceSegmentation"):
        folder_paths.folder_names_and_paths["cartoon_seg_ckpt"] = ([model_path_ani_inst_ckpt], {'.ckpt', '.safetensors'})
        cartoon_seg_folders["anime_instance_path"] = model_path_ani_inst_ckpt
        # Add Models to the sys.path.  A call to Config.loadString in AnimeInsSeg external lib mmengine needs help
        sys.path.append(model_path_ani_inst_ckpt)

    if not check_report_path(model_path_ani_seg_ckpt, "Cartoon Segmentation anime-seg checkpoints"):
        cartoon_seg_folders["anime_seg_checkpoints"] = model_path_ani_seg_ckpt

    if not check_report_path(leres_path, "leres_path"):
        cartoon_seg_folders["leres_path"] = leres_path


    cartoon_segmentation_root_split = re.split(r"/|\\", cartoon_segmentation_root)
    # There is a call in net_tools that does not full path, but a relative path for import
    comfy_root_length = len(re.split(r"/|\\", folder_paths.base_path))
    depth_lres_base = ".".join(cartoon_segmentation_root_split[comfy_root_length:])

    cartoon_seg_folders["depth_leres"] = f"{depth_lres_base}.depth_modules.leres"


add_search_paths()

class KenBurnsConfigLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config_file": (folder_paths.get_filename_list("cartoon_seg_configs"), ),
            "model": (folder_paths.get_filename_list("cartoon_seg_ckpt"), ),
            "image_path": ("image_path", {"display": "Input Image File Name"}),
            "verbose": (["Yes", "No"], {"display": "Verbose Logging", "default": "No"}),
            },
        }

    RETURN_TYPES = ('kb_config', )
    FUNCTION = 'load_kb_config'
    CATEGORY = "CartoonSegmentation"

    def load_kb_config(self, config_file, model, image_path, verbose):
        cfg_full_path = os.path.join(folder_paths.folder_names_and_paths["cartoon_seg_configs"][0][0], config_file)
        model_full_path = os.path.join(folder_paths.folder_names_and_paths["cartoon_seg_ckpt"][0][0], model)
        return_value = cfg_full_path, model_full_path, image_path, verbose == "Yes"
        return return_value,

class LoadImageFilename:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("image_path", )
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        return (image_path, )

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

