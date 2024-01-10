# ComfyUI_CartoonSegmentation

Front end ComfyUI nodes for CartoonSegmentation
Based upon the work of the [CartoonSegmentation]([https://github.com/CartoonSegmentation/CartoonSegmentation] ) repository this project will provide a front end to some of the features.

## Run Segmentation
This will create segmentations for an input image.  An example of final anime segregation result is [[examples/workflows_and_outputs/ani_seg_standing_rain.png ]]folder.  The image source used is in the [[examples/source_image/standing_in_rain.png]]

## Run 3d Kenburns
This will create a short video of a zoom in and out from a single image.  There are two finished output examples show in the [[examples/workflows_and_outputs/]] folder based on the images warrior and warrior_blue in the [[examples/source_image/]] folder.

## Roadmap
- ~~Produce demo examples in ComfyUI~~
- Publish Code
	- Add installation instructions as requirements from the  [CartoonSegmentation]([https://github.com/CartoonSegmentation/CartoonSegmentation] ) repository are using conda.
	- Add pip versions to requirements.txt
	- Upload source code
- Fine tune model selection as some models currently crash
- Output segs for the segmentation in addition to an image
