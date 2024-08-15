# Text-guidedSR
Code for the CVPR 2024 [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Gandikota_Text-guided_Explorable_Image_Super-resolution_CVPR_2024_paper.html) Text-guided Explorable Image Super-resolution.

## Environment settings and libraries we used in our experiments

This project is tested on a machine with
- OS: Ubuntu 22.04.4 
- GPU: NVIDIA GeForce RTX 3090 

Comprehensive list of packages used in the environment used for experiments are provided in requirements.txt

## Acknowledgement
The codes are based on [deep-floyd IF](https://github.com/deep-floyd/IF), [karlo unCLIP](https://github.com/kakaobrain/karlo), using [Huggingface Diffusers](https://github.com/huggingface/diffusers). Three zero-shot methods [DDNM](https://github.com/wyhuai/DDNM), [DPS](https://github.com/DPS2022/diffusion-posterior-sampling), [PiGDM](https://github.com/NVlabs/RED-diff) are included for text guided super-resolution.
We thank the authors and contributors of these repositories for making their code public!

## Coming Soon
- **T2I DPS and T2I PiGDM**.
- **unCLIP DDNM**.

## Note
We corrected a bug in our PSNR computation using Deepfloyd IF (Imagen) for SR.The LR PSNRs are now better than reported values in the paper.

As mentioned in the discussion section, even with high LR PSNR, results may not always be perceptually high quality. If the result is not satisfactory try running with different random seeds.

## Evaluation on dataset
Download multimodal CelebA HQ to /data/work_data/multi_mod_celebahq

running command for testing 8x SR
```python
python run_dataset_imagen_ddnm.py --count 200 --scale 16 --g1 7 --g2 4 --run 3\
```

running command for testing 16x SR
```python
python run_dataset_imagen_ddnm.py --count 200 --scale 16 --g1 7 --g2 4 --run 3\
```


## References
If you find our workuseful for your research, please consider citing
```bib
@InProceedings{Gandikota_2024_CVPR,
    author    = {Gandikota, Kanchana Vaishnavi and Chandramouli, Paramanand},
    title     = {Text-guided Explorable Image Super-resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {25900-25911}
}

```


