# Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

## Results and Models
<div align="center">
  <b> Results from Pix2Pix trained by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/114269080-4ff0ec00-9a37-11eb-92c4-1525864e0307.PNG" width="800"/>
</div>
We use `FID` and `IS` metrics to evaluate the generation performance of pix2pix.

`FID` evaluation:

| Dataset  | [facades](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py) | [maps-a2b](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pix2pix/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps.py) | [maps-b2a](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pix2pix/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps.py) | [edges2shoes](https://github.com/open-mmlab/mmgeneration/tree/master/configs/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes.py) |   average    |
| :------: | :--------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------: |
| official |                                                         **119.135**                                                          |                                                              149.731                                                              |                                                              102.072                                                              |                                                                       **75.774**                                                                       |   111.678    |
|   ours   |                                                           124.372                                                            |                                                            **122.691**                                                            |                                                            **88.378**                                                             |                                                                         85.144                                                                         | **105.1463** |

`IS` evaluation:

| Dataset  |  facades  | maps-a2b  | maps-b2a  | edges2shoes |  average  |
| :------: | :-------: | :-------: | :-------: | :---------: | :-------: |
| official |   1.650   |   2.529   |   3.552   |    2.766    |   2.624   |
|   ours   | **1.665** | **3.337** | **3.585** |  **2.797**  | **2.846** |

Model and log downloads:

| Dataset  |                                                                                                                        facades                                                                                                                        |                                                                  maps-a2b                                                                  |                                                                  maps-b2a                                                                  |                                                                         edges2shoes                                                                          |
| :------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| download | [model](https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py_20210410_174537-36d956f1.pth) \| [log](https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades_20210317_172625.log.json) | [model](https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_convert-bgr_20210410_173329-3ec2ed64.pth) | [model](https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_convert-bgr_20210410_173757-e262060f.pth) | [model](https://download.openmmlab.com/mmgen/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210410_174116-aaaa3687.pth) |

Note: we strictly follow the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) setting in Section 3.3: "*At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization using the statistics of
the test batch, rather than aggregated statistics of the training batch.*" (i.e., use model.train() mode), thus may lead to slightly different inference results every time.
