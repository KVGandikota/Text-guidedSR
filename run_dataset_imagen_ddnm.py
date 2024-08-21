import sys
import argparse
import random
from pipeline_if_all import IFPipeline
from pipeline_if_superresolution_all import IFSuperResolutionPipeline
from diffusers.utils import pt_to_pil
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import get_Afuncs

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import utils
import logging
import importlib
import pandas as pd
import numpy as np



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Text guided super resolution')
    parser.add_argument('--log_dir', type=str, default='SR', help='Experiment root')
    parser.add_argument('--scale', type=int, default=8, help='super-resolution scale options 8,16')
    parser.add_argument('--count', type=int, default=200, help='number of samples for testing')
    parser.add_argument('--dec_steps', type=int, default=100, help='number of decoder inference steps')
    parser.add_argument('--sr_steps', type=int, default=50, help='number of 4xSR inference steps')
    parser.add_argument('--start_time', type=int, default=None, help='start step for diffusion')
    parser.add_argument('-g1','--guidance_scale_stage1', type=float, default=7.0, help= 'classifier free guidance in 1st stage imagen diffusion if g1>1')
    parser.add_argument('-g2','--guidance_scale_stage2', type=float, default=4.0, help= 'classifier free guidance in 2nd stage (SR stage) in imagen if g2>1')
    parser.add_argument('--run', type=int, default=1, help='number of runs')
    return parser.parse_args()


def main (args):
    stage_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
    #stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_1.enable_model_cpu_offload()
    # stage 2
    stage_2 = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    )
    #stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
    stage_2.enable_model_cpu_offload()


    #Dataset
    obj = pd.read_pickle(r'/data/work_data/multi_mod_celebahq/filenames.pickle')

    image_dir = '/data/work_data/multi_mod_celebahq/CelebAMask-HQ/CelebA-HQ-img/'

    #Resize ground truth currently supporting GT and output resolution 256x256
    H = 256
    W = 256
    improcess =  transforms.Compose([transforms.Resize((H,W), interpolation=InterpolationMode.BICUBIC)])
   


    #Setting up Consistency enforcement 
    scale = args.scale
    
    Av=get_Afuncs('sr_bicubic',sizes=256,sr_scale=scale) 
    A = Av.A
    Ap = Av.A_pinv

    exp_dir = f'{args.log_dir}_{args.scale}/'
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    #Keep track of Quantitative evaluation
    LR_psnr=[]
        
    for restart in range(args.run):
            exp_dir1 = exp_dir
            exp_dir1 = f'{exp_dir}/restart{str(restart)}'
            if not os.path.isdir(exp_dir1):
                os.mkdir(exp_dir1)
            for i in range(args.count):
                generator = torch.manual_seed(restart)
                #get image and caption
                file_num = obj[i]
                image_name = f'{image_dir}{file_num}.jpg'
                gt_t = Image.open(image_name)
                gt_t = utils.from_pil_image(improcess(gt_t))[:3,:,:]
                prompt = 'A professional realistic high-res portrait face photograph' 
                negative_prompt = 'disfigured,blurred, ugly, bad, immature, bad teeth, caricature, unnatural, wierd lighting, fake'
                #embeddings
                prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt = prompt, negative_prompt = negative_prompt)
                #bicubic downsampling, generate LR observation
                lrf = A(gt_t.detach().unsqueeze(0).cuda()).reshape(1,3,256//scale,256//scale).half()
                
                # Text guided restoration
                
                #Within pipe stage 1
                image = stage_1.sup_res(lr = lrf.detach(),prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, num_inference_steps=args.dec_steps, guidance_scale = args.guidance_scale_stage1, generator=generator, output_type="pt", sr_scale=scale//4, algo='ddnm')
                
                #pipe stage 2                
                image = stage_2.sup_res(image=image,lr = lrf.detach(),sr_scale=scale//4, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,  num_inference_steps=args.sr_steps, guidance_scale = args.guidance_scale_stage2, generator=generator, output_type="pt", algo='ddnm').images.clamp_(-1,1)
                
                image =  0.5+0.5*image
                # Quantitative evaluation
                lroutfin = A(image.reshape(image.size(0),-1).float()).reshape(1,3,256//scale,256//scale)
                LR_psnr.append(-10*torch.log10(F.mse_loss(lroutfin, 0.5+0.5*lrf))) 
                res = (image[0].float().cpu()).permute(1,2,0).clip(0,1.0).numpy()
                plt.imsave(f'{exp_dir1}/{i}_{restart}.jpg',res)
                lrsave =F.interpolate( 0.5+0.5*lrf.float(), size=(256,256), mode='nearest')[0].detach().cpu().permute(1,2,0).clip(0,1.0).numpy()
                plt.imsave(f'{exp_dir1}/{i}_LRx{args.scale}.jpg',lrsave)

                
            
            np.savetxt(f'{exp_dir1}/LR_Psnr.txt', torch.Tensor(LR_psnr).numpy())
            
            stats = 'mean LR_PSNR is ' + str(torch.Tensor(LR_psnr).mean().numpy())
            
            text_file = open(f'{exp_dir1}/stats.txt', "a")
            n = text_file.write(stats)
            text_file.close()

                        
if __name__ == '__main__':
    args = parse_args()
    main(args)
