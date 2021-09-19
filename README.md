# ICE-GAN
**\[our code will be coming soon]**

ICE-GAN: Identity-aware and Capsule-Enhanced GAN for Micro-Expression Recognition and Synthesis. ([Arxiv version](https://arxiv.org/pdf/2005.04370.pdf))

Author: Jianhui Yu, Chaoyi Zhang, Yang Song, Weidong Cai
## Model Architecture
![model architecture](/images/model_overview.jpg)

# Installation
The code is tested with Python 3.6, CUDA 10.1, Pytorch 1.4 on Ubuntu 18.04.


# Usage
## Dataset
* The links for the data we use are provided below:
    1. [SMIC dataset](https://www.oulu.fi/cmvs/node/41319)
    2. [CASME II dataset](http://fu.psych.ac.cn/CASME/casme2-en.php)
    3. [SAMM dataset](http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php)

## Classifcation
To train the model from scratch, use the following code:
```python
python main.py
```
# Performance
![Model performance](/images/performance.png)
# License
