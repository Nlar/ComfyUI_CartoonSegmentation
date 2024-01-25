# ComfyUI_CartoonSegmentation

Front end ComfyUI nodes for CartoonSegmentation
based upon the work of the [CartoonSegmentation](https://github.com/CartoonSegmentation/CartoonSegmentation) repository this project will provide a front end to some of the features.  For notation purposes this project will be referred to as ComfyUI_CartoonSegmentation of briefly as "CfyCS" and the non UI code that provides the power to this project will be referred to as "CartoonSegmentation". 

## Run Segmentation
This will create segmentations for an input image.  An example of final anime segmentation result is [workflows and outputs](examples/workflows_and_outputs/cartoon_segmentation.png) folder.  The image source used is in the [[examples/source_image/standing_in_rain.png]]

## Run 3d Kenburns
This will create a short video of a zoom in and out from a single image.  There are two finished output examples show in the [[examples/workflows_and_outputs/]] folder based on the images warrior and warrior_blue in the [example source image](examples/source_image/) folder.  This workflow includes a required image creation at the end of workflow to pass ComfyUI validation.  The image is a screenshot from the half way through the video. 

## Install
### Initial Setup
```bash
# Create a venv and active it
# The root of the environment will be noted as [venv_path]
# Start in ComfyUI Root Directory
cd custom_nodes
git clone https://github.com/Nlar/ComfyUI_CartoonSegmentation.git
cd ComfyUI_CartoonSegmentation
# This is inteded as a superset of the requirments for CartoonSegmentation
pip install -r requirement.txt
```
### Addtional Files
#### libpatchmatch
Follow the [instructions](https://github.com/CartoonSegmentation/CartoonSegmentation#compile-patchmatch) from the Cartoon Segmentation project.

With the nodes new directory structure the file should be placed in 
./custom_nodes/ComfyUI_CartoonSegmentation/CartoonSegmentation/data/libs/libpatchmatch_inpaint.so

#### Models
Follow the [instructions](https://github.com/CartoonSegmentation/CartoonSegmentation#download-models) to download the models from Hugging Face after changing the local directory.  
```bash
# The ckpt files should be in the directory [comfy_root]/custom_nodes/ComfyUI_CartoonSegmentation/CartoonSegmentation/models/AnimeInstanceSegmentation
cd custom_nodes/ComfyUI_CartoonSegmentation/CartoonSegmentation
# Follow the instructions from CartoonSegmentation repository
```
**Currently only the rtmdetl_e60.ckpt model is supported.**  

#### res101.pth
The original hosting provider no longer stores this annotator.  It can be found on [HuggingFace](https://huggingface.co/lllyasviel/Annotators/blob/af19c34529d974eb965a00250f7b743431d56047/res101.pth) and should be placed in the "[comfy_root]/custom_nodes/ComfyUI_CartoonSegmentation/CartoonSegmentation/models/leres" directory.
```bash
cd [comfy_root]/custom_nodes/ComfyUI_CartoonSegmentation/CartoonSegmentation/models
mkdir leres
# Place file in the leres directory 
```

### Troubleshooting

#### ModuleNotFoundError: No module named 'mmcv._ext'
The requirements for mmcv are fairly specific and this message indicates that will have to manually install the package. The current version of mmcv (2.1.0) can be used with this module.  The best case install can be completed by downloading mmcv from the [mmcv release](https://github.com/open-mmlab/mmcv/releases/tag/v2.1.0) and installed through pip.

```bash
pip install [extracted_directory]/mmcv
```

#### OSError: [venv_path]/lib/python3.11/site-packages/torch/lib/libgomp-a34b3233.so.1: version `GOMP_5.0' not found (required by /usr/lib/libvtkCommonCore.so.1)
The installed version of gcc-libs (13.2.1-3) provided a file from /usr/lib/libgomp.so.  This file was able to resolve this issue.
```bash
cd [venv_path]/lib/python3.1*/site-packages/torch/lib/
mv libgomp-a34b3233.so.1 libgomp-a34b3233.so.1.bak
# Symlink below can be replaced with a copy command. 
ln -s /usr/lib/libgomp.so libgomp-a34b3233.so.1
```


## Roadmap
- ~~Correct output folder issue~~
- ~~Verify functionality beyond KenBurns Config Loader~~
- ~~Ouptut masks for the input image during anime segmentation.~~  
