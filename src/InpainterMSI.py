from .ControlNetInpaint.src.pipeline_stable_diffusion_controlnet_inpaint import *
from .DIP import *
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torchvision.transforms as T
from torch import nn
import torch

toTensor=T.ToTensor()
toPIL = T.ToPILImage()

def mask2pil(mask, shape=(512,512)):
    mask=torch.stack(3*[mask],0)
    return toPIL(mask).resize(shape)

def s2pil(s2_image, shape=(512,512)):
    rgb_image=s2_image.squeeze()[(3,2,1),:,:].clip(0,1)
    return toPIL(rgb_image).resize(shape)

class InpainterMSI(nn.Module):

    def __init__(self,
                 type = 'EG', # EG - Edge-Guided StableDiffsion; SD - Regular Stable Diffusion; DIP - Deep Image Prior
                ):

        super().__init__()
        self.type = type
        self.dip_model=DIP(sigmoid_output=False)
        
        if self.type == 'EG':
            # EG Inpainting
            self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
            self.eg_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
              "runwayml/stable-diffusion-inpainting", controlnet=self.controlnet, torch_dtype=torch.float16).to("cuda")
            # speed up diffusion process with faster scheduler and memory optimization
            self.eg_pipe.scheduler = UniPCMultistepScheduler.from_config(self.eg_pipe.scheduler.config)
            from controlnet_aux import HEDdetector
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        elif self.type == 'SD':
            # SD Inpainting
            self.sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",
                                                                          torch_dtype=torch.float16).to("cuda")
            self.sd_pipe.scheduler = UniPCMultistepScheduler.from_config(self.sd_pipe.scheduler.config)
        elif self.type == 'DIP':
            self.text_model=None
        else:
            raise NotImplemented
    
    def forward(self,
                input,
                mask,
                condition=None,
                hist_fill=True, # only relevant for 'EG' and 'SD'
                prompt="a cloud-free satellite image",
                text_guidance_scale = 7.5,
                edge_guidance_scale = 0.5,
                num_inference_steps=20,
                num_DIP_steps=4000,
                DIP_lr = 2e-2,
                generator=torch.manual_seed(0),
                msi=True,
                rgb_channels=(3,2,1) # indicate the indices of RGB from the MSI representation
               ):

        if hist_fill:
            source=input*mask[None,:,:] + condition*(1-mask[None,:,:]) # hist-fill
        else:
            source=input*mask[None,:,:] # black-fill
        
        input_img=s2pil(source)
        input_mask=mask2pil(1-mask)

        if self.type == 'DIP':
            output = source
        else:
            print('STEP I: RGB-based Inpainting')
            if self.type == 'SD':
                image = self.sd_pipe(prompt=prompt,
                                     num_inference_steps=num_inference_steps,
                                     guidance_scale=text_guidance_scale,
                                     generator=generator,
                                     image=input_img,
                                     mask_image=input_mask
                                     ).images[0]
            elif self.type == 'EG':
                hed_hist=self.hed(255*condition[rgb_channels,:,:].permute(1,2,0).clip(0,1))
                image = self.eg_pipe(prompt=prompt,
                                     num_inference_steps=num_inference_steps,
                                     guidance_scale=text_guidance_scale,
                                     controlnet_conditioning_scale = edge_guidance_scale,
                                     generator=generator,
                                     image=input_img,
                                     control_image=hed_hist,
                                     mask_image=input_mask).images[0]
            # update image and mask
            output = source
            mask = mask.repeat(output.shape[-3],1,1)
            output[...,rgb_channels,:,:] = toTensor(image.resize(output.shape[-2:]))
            mask[...,rgb_channels,:,:] = 1

        # reconstruct
        if msi and output.shape[-3] != 3:
            self.dip_model=DIP(sigmoid_output=False,
                               lr=DIP_lr,
                               epoch_steps=num_DIP_steps)

            if condition is not None and self.type == 'DIP':
                targets = [output.unsqueeze(0), condition.unsqueeze(0)]
                masks = [mask, torch.ones_like(mask)]
            else:
                targets = [output.unsqueeze(0)]
                masks = [mask]
            # used DIP model
            self.dip_model.set_target(*targets)
            self.dip_model.set_mask(*masks)
                
            print('STEP II: MSI Expansion')
            trainer = pl.Trainer(
                max_epochs = 1,    
                logger=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                #enable_progress_bar = False,
                devices = [0]
            )
            trainer.fit(self.dip_model)
            result=self.dip_model.output()[0]

            # mix prediction and known
            final=(mask*output + (mask.logical_not())*result)
        else:
            final = output

        return final
