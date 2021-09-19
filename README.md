# ICE-GAN
**\[our code for model design is released]**

ICE-GAN: ICE-GAN: Identity-aware and Capsule-Enhanced GAN with Graph-based Reasoning for Micro-Expression Recognition and Synthesis. ([Arxiv version](https://arxiv.org/pdf/2005.04370.pdf))

Author: Jianhui Yu, Chaoyi Zhang, Yang Song, Weidong Cai

## Model Architecture
![model architecture](/images/model_overview.jpg)

## Requirements
* Python >=3.6
* Pytorch >= 1.4
* Packages: tqdm, sklearn

## Dataset
* The links for the data we use are provided below:
    1. [SMIC dataset](https://www.oulu.fi/cmvs/node/41319)
    2. [CASME II dataset](http://fu.psych.ac.cn/CASME/casme2-en.php)
    3. [SAMM dataset](http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php)

## Code
This repo contains Pytorch implementation of the following modules:
- [x] training log file for all splits;
- [x] model/discriminator.py: model architecture for capsule-based discriminator;
- [x] model/generator.py: model architecture for Unet-based generator with graph reasoning module (GRM);
- [ ] ...

## Performance
![Model performance](/images/performance.jpg)

## Acknowledgement
Our code borrows a lot from:
* [pix2pix](https://github.com/phillipi/pix2pix)
* [Pytorch-CapsuleNet](https://github.com/jindongwang/Pytorch-CapsuleNet)