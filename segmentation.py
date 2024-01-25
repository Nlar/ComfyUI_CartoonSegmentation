import os

import PIL
import numpy as np
import torch
from PIL import ImageSequence, ImageOps
from PIL.Image import Image

import mmcv
import cv2

import folder_paths as folder_paths

from custom_nodes.ComfyUI_CartoonSegmentation.CartoonSegmentation.anime_3dkenburns.kenburns_effect import build_kenburns_cfg, KenBurnsPipeline, npyframes2video
from custom_nodes.ComfyUI_CartoonSegmentation.CartoonSegmentation.animeinsseg import AnimeInsSeg, AnimeInstances
from custom_nodes.ComfyUI_CartoonSegmentation.CartoonSegmentation.utils.constants import get_color


def pil_image_to_image(input_image):
    output_images = []
    for i in ImageSequence.Iterator(input_image):
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
    else:
        output_image = output_images[0]

    return output_image

def nparray_to_image(nparr):
    pil_img = PIL.Image.fromarray(nparr)
    # pil_img.save("output/testing_image.png")

    return pil_image_to_image(pil_img)



def get_drawed(drawed, img, instances, target_mask_count):
    im_h, im_w = img.shape[:2]

    mask_array = None
    target_mask = None
    counter = 1
    try:
        for ii, (xywh, mask) in enumerate(zip(instances.bboxes, instances.masks)):
            color = get_color(ii)

            mask_alpha = 0.5
            linewidth = max(round(sum(img.shape) / 2 * 0.003), 2)

            # draw bbox
            p1, p2 = (int(xywh[0]), int(xywh[1])), (int(xywh[2] + xywh[0]), int(xywh[3] + xywh[1]))
            cv2.rectangle(drawed, p1, p2, color, thickness=linewidth, lineType=cv2.LINE_AA)

            # draw mask
            p = mask.astype(np.float32)
            mask_tensor = torch.from_numpy(p)
            img_tensor = mask_tensor.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            if mask_array is None:
                mask_array = img_tensor
            else:
                mask_array = torch.cat((mask_array, img_tensor), 0)
            if counter == target_mask_count:
                target_mask = img_tensor
            counter += 1
            # torch.from_numpy(p).resize(1, im_h, im_w))
            blend_mask = np.full((im_h, im_w, 3), color, dtype=np.float32)
            alpha_msk = (mask_alpha * p)[..., None]
            alpha_ori = 1 - alpha_msk
            drawed = drawed * alpha_ori + alpha_msk * blend_mask

        drawed = drawed.astype(np.uint8)
    except Exception as e:
        print("Nothing found to segment")

    return drawed, mask_array, target_mask






class AnimeSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (folder_paths.get_filename_list("cartoon_seg_ckpt"), {"default": "rtmdetl_e60.ckpt"},),
            "image": ("IMAGE", {}),
            "mask_thres": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 4.0, "step": 0.01}),
            "instance_thres": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 4.0, "step": 0.01}),
            "character_output": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
        }

    RETURN_TYPES = ('IMAGE', "IMAGE", "IMAGE")
    # OUTPUT_IS_LIST = (False, False, True,)
    RETURN_NAMES = ('display image', 'character image mask', 'all image masks')
    FUNCTION = 'ani_segmentation'
    CATEGORY = "CartoonSegmentation"

    def ani_segmentation(self, model, image, mask_thres, instance_thres, character_output):
        ckpt = os.path.join(folder_paths.folder_names_and_paths["cartoon_seg_ckpt"][0][0], model)
        refine_kwargs = {'refine_method': 'refinenet_isnet'}  # set to None if not using refinenet

        num_img = np.flip((image.squeeze(0).numpy() * 255), axis=2)

        net = AnimeInsSeg(ckpt, mask_thr=mask_thres, refine_kwargs=refine_kwargs)
        instances: AnimeInstances = net.infer(
            num_img,
            output_type='numpy',
            pred_score_thr=instance_thres
        )

        drawed, mask_images, character_mask = get_drawed(num_img.copy(), num_img, instances, character_output)
        # img = PIL.Image.fromarray(drawed[..., ::-1])

        if drawed is None:
            drawed = image

        seg_img = pil_image_to_image(PIL.Image.fromarray(drawed[..., ::-1]))

        if mask_images is None:
            mask_images = seg_img
            targeted_image = seg_img

        return seg_img, character_mask, mask_images

class KenBurns_Processor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kb_config": ("kb_config", {}),
                "output_video_name": ("STRING", {"default": "kenburns.mp4"}),
            }
        }

    FUNCTION = "perform_3dkb"
    CATEGORY = "CartoonSegmentation"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAME = ("Output Preview", )
    OUTPUT_NODE = True



    def perform_3dkb(self, kb_config, output_video_name):
        (cfg_full_path, det_ckpt, image_path, verbose, ) = kb_config
        video_path_output = f"{folder_paths.output_directory.rstrip('/')}/{output_video_name}"

        kbc = build_kenburns_cfg(cfg_full_path)
        kbc.det_ckpt = det_ckpt

        kpipe = KenBurnsPipeline(kbc)

        img = mmcv.imread(image_path)

        kcfg = kpipe.generate_kenburns_config(img, verbose=verbose, savep=video_path_output)

        if verbose:
            stage_instance_vis = kcfg.instances.draw_instances(img, draw_tags=False)
            cv2.imwrite('tmp_stage_instance.png', stage_instance_vis)

            cv2.imwrite('tmp_stage_depth_coarse.png', kcfg.stage_depth_coarse)
            cv2.imwrite('tmp_stage_depth_adjusted.png', kcfg.stage_depth_adjusted)
            cv2.imwrite('tmp_stage_depth_final.png', kcfg.stage_depth_final)

        npy_frame_list = kpipe.autozoom(kcfg, verbose=verbose)

        if verbose:
            for ii, inpainted_img in enumerate(kcfg.stage_inpainted_imgs):
                cv2.imwrite(f'tmp_stage_inpaint_{ii}.png', inpainted_img)
            for ii, mask in enumerate(kcfg.stage_inpainted_masks):
                cv2.imwrite(f'tmp_stage_inpaint_mask_{ii}.png', mask)

        npyframes2video(npy_frame_list, video_path_output)
        np_arry_sample = npy_frame_list[int(len(npy_frame_list) / 2)]

        img = nparray_to_image(np_arry_sample)

        return (img, )
