"""
@author: Nels Larsen
@title: ComfyUI_CartoonSegmentation
@nickname: CfyCS
@description: This extension offers a front end to the Cartoon Segmentation Project (https://github.com/CartoonSegmentation/CartoonSegmentation)
"""

from custom_nodes.ComfyUI_CartoonSegmentation.loaders import KenBurnsConfigLoader, LoadImageFilename
from custom_nodes.ComfyUI_CartoonSegmentation.segmentation import KenBurns_Processor, AnimeSegmentation

# from CartoonSegmentation.utils.constants import COLOR_PALETTE
# from CartoonSegmentation.animeinsseg import AnimeInsSeg, AnimeInstances

print(f"### Loading: ComfyUI_CartoonSegmentation ##")

NODE_CLASS_MAPPINGS = {
    "KenBurns_Processor": KenBurns_Processor,
    "KenBurnsConfigLoader": KenBurnsConfigLoader,
    "LoadImageFilename": LoadImageFilename,
    "AnimeSegmentation": AnimeSegmentation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KenBurns_Processor": "3d Kenburns Processor",
    "KenBurnsConfigLoader": "KenBurns Config Loader",
    "LoadImageFileName": "Load Image Filename",
    "AnimeSegmentation": "Anime Segmentation"
}

try:
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS)

except:
    pass

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
