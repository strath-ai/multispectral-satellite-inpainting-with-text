# multispectral-satellite-inpainting-with-text
<a href="https://arxiv.org/abs/2311.03008/" class="text-decoration-none site-link"><img alt="Arxiv Link" src="https://img.shields.io/badge/Open_Access-arxiv:2311.03008-b31b1b"></a> <a href="https://ieeexplore.ieee.org/document/10445344" class="text-decoration-none site-link"><img alt="IEEE Xplore Link" src="https://img.shields.io/badge/Access-IEEE-00629B"></a>

Official code for the paper "[*Exploring the Capability of Text-to-Image Diffusion Models With Structural Edge Guidance for Multispectral Satellite Image Inpainting*](https://ieeexplore.ieee.org/document/10445344/)" published in IEEE Geoscience and Remote Sensing Letters:
```latex
@ARTICLE{10445344,
  author={Czerkawski, Mikolaj and Tachtatzis, Christos},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Exploring the Capability of Text-to-Image Diffusion Models With Structural Edge Guidance for Multispectral Satellite Image Inpainting}, 
  year={2024},
  volume={21},
  pages={1-5},
  keywords={Satellite images;Image edge detection;Data models;Standards;Noise reduction;Task analysis;Process control;Generative models;image completion;image inpainting},
  doi={10.1109/LGRS.2024.3370212}
}
```

## Method
There two components to the considered system:

### RGB-based Inpainting Model
The method explores the capabilities of existing off-the-shelf inpainting models, in this case, StableDiffusion 1.5 and ControlNet conditioning.
![diffusion-edgeguided-infer](https://github.com/strath-ai/multispectral-satellite-inpainting-with-text/assets/13435425/c8ca8429-38e7-4b74-ab9a-8b9ec89eadfe)


> Based on the [ControlNetInpaint](https://github.com/mikonvergence/ControlNetInpaint) [![GitHub Repo stars](https://img.shields.io/github/stars/mikonvergence/ControlNetInpaint)](https://github.com/mikonvergence/ControlNetInpaint) repository. Check it out for other types of conditioned inpainting!

### Channel-wise Inpainting
The output of an existing model can be used to propagate the inpainting information across channels. Here, Deep Image Prior is used to do this in an internal learning regime (no pre-training necessary).
![dip-diagram](https://github.com/strath-ai/multispectral-satellite-inpainting-with-text/assets/13435425/8b8e2a03-388f-47a9-8f4a-bafd28a87712)

## Variants

| Variant | Details |
| ------- | ------- |
| Direct-DIP | Direct use of Deep Image Prior to inpaint an image (With an optional conditioning signal, such as historical) |
| SD-Inpainting | Conventional use of StableDiffusion 1.5 Inpainting Model |
| Edge-Guided Inpainting | StableDiffusion 1.5 inpainting with the use of edge-guided conditioning via ControlNet |

Example use:
```python
model = InpainterMSI(type='EG')

out = model(current,
            mask,
            condition=hist,
            #prompt='Your custom text prompt',
            text_guidance_scale = 7.5, # influence of text prompt
            edge_guidance_scale = 0.5, # influence of historical edge
            num_inference_steps=20, # number of diffusion RGB inpainting steps
            num_DIP_steps=4000 # number of DIP optimisation steps for RGB-to-MSI
           )
```
