# Analyzing and Improving the Image Quality of Stylegan (CVPR'2020)

## Abstract
We observe that despite their hierarchical convolutional nature, the synthesis
process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to
be glued to image coordinates instead of the surfaces of depicted objects. We trace
the root cause to careless signal processing that causes aliasing in the generator
network. Interpreting all signals in the network as continuous, we derive generally
applicable, small architectural changes that guarantee that unwanted information
cannot leak into the hierarchical synthesis process. The resulting networks match
the FID of StyleGAN2 but differ dramatically in their internal representations, and
they are fully equivariant to translation and rotation even at subpixel scales. Our
results pave the way for generative models better suited for video and animation.


<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/22982797/150353023-8f7eeaea-8783-4ed4-98d5-67a226e00cff.png"/>
</div>

## Citation

<!-- [ALGORITHM] -->

<summary align="right"><a href="https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf">StyleGANv3 (NeurIPS'2021)</a></summary>

```latex
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}
```

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGAN3 config-T converted by MMGeneration</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/150450502-c182834f-796f-4397-bd38-df1efe4a8a47.png" width="800"/>
</div>

|                Model                |     Comment     | FID50k |    EQ-T     | EQ-R     |                                                            Config                                                             |                                                                 Download                                                                 |
| :---------------------------------: | :-------------: | :----: | :-----------: | :-----------: |:---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
|    stylegan3_config-T_afhqv2_512    | official weight | 4.04 | 60.15 | 13.51   |  [config](configs/styleganv3/stylegan3_t_afhqv2_512_b4x8_official.py)       |  [model]()  |
|    stylegan3_config-T_ffhqu_256    | official weight | 4.62 | 63.01 | 13.12   |  [config](configs/styleganv3/stylegan3_t_ffhqu_256_b4x8_official.py)       |  [model]()  |
|    stylegan3_config-R_afhqv2_512     | official weight |4.40    |64.89  | 40.34   |  [config](configs/styleganv3/stylegan3_r_afhqv2_512_b4x8_official.py)       |  [model]()  |
|    stylegan3_config-R_ffhqu_256    | official weight |  4.50| 66.65 |  40.48  |  [config](configs/styleganv3/stylegan3_r_ffhqu_256_b4x8_official.py)       |  [model]()  |



## Interpolation

https://user-images.githubusercontent.com/22982797/150354820-638ce279-b548-492f-8a5e-e3faf5170a8a.mp4

https://user-images.githubusercontent.com/22982797/150354922-f5612775-f617-4ed2-8562-a06bbef0fbab.mp4

## Equivarience Visualization && Evaluation

https://user-images.githubusercontent.com/22982797/150293816-23a9ac23-ce07-487b-8fea-9303fab05658.mp4

https://user-images.githubusercontent.com/22982797/150293909-0887dd5a-18f3-423b-a06a-39a940e03b0a.mp4

https://user-images.githubusercontent.com/22982797/150294018-cf9b151d-7b76-4cfd-9c05-f86f0a324ff5.mp4

https://user-images.githubusercontent.com/22982797/150294058-8a444653-1416-4997-bc16-61509f32e33f.mp4

https://user-images.githubusercontent.com/22982797/150294118-0af818b8-7ad7-4ea6-a0eb-6e5f0d920b2e.mp4

https://user-images.githubusercontent.com/22982797/150294479-a0e33233-a16d-4521-bacd-2fffb68ef3d7.mp4

## Visualize average 2D power spectra
