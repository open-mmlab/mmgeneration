# A Style-Based Generator Architecture for Generative Adversarial Networks (CVPR'2019)

## Introduction
<!-- [ALGORITHM] -->
```latex
@inproceedings{karras2019style,
  title={A style-based generator architecture for generative adversarial networks},
  author={Karras, Tero and Laine, Samuli and Aila, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4401--4410},
  year={2019}
}
```

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGANv1 trained by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113845642-4f9ee980-97c8-11eb-85c7-49d6d21dd46b.png" width="800"/>
</div>

|        Model         | FID50k |  P&R50k_full  |                                                        Config                                                         |                                                      Download                                                       |
| :------------------: | :----: | :-----------: | :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
| styleganv1_ffhq_256  | 6.090  | 70.228/27.050 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv1/styleganv1_ffhq_256_g8_25Mimg.py)  | [model](https://download.openmmlab.com/mmgen/styleganv1/styleganv1_ffhq_256_g8_25Mimg_20210407_161748-0094da86.pth)  |
| styleganv1_ffhq_1024 | 4.056  | 70.302/36.869 | [config](https://github.com/open-mmlab/mmgeneration/tree/master/configs/styleganv1/styleganv1_ffhq_1024_g8_25Mimg.py) | [model](https://download.openmmlab.com/mmgen/styleganv1/styleganv1_ffhq_1024_g8_25Mimg_20210407_161627-850a7234.pth) |
