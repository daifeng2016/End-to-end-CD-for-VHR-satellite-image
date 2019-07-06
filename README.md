# End-to-end-CD-for-VHR-satellite-image
The project aims to contribute to geoscience community.
<br>
## Paper

End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++ [https://www.mdpi.com/2072-4292/11/11/1382] 
## Introduction

Change detection (CD) is essential to the accurate understanding of land surface changes using available Earth observation data. Due to the great advantages in deep feature representation and nonlinear problem modeling, deep learning is becoming increasingly popular to solve CD tasks in remote-sensing community. However, most existing deep learning-based CD methods are implemented by either generating difference images using deep features or learning change relations between pixel patches, which leads to error accumulation problems since many intermediate processing steps are needed to obtain final change maps. To address the above-mentioned issues, a novel end-to-end CD method is proposed based on an effective encoder-decoder architecture for semantic segmentation named UNet++, where change maps could be learned from scratch using available annotated datasets. Firstly, co-registered image pairs are concatenated as an input for the improved UNet++ network, where both global and fine-grained information can be utilized to generate feature maps with high spatial accuracy. Then, the fusion strategy of multiple side outputs is adopted to combine change maps from different semantic levels, thereby generating a final change map with high accuracy. The effectiveness and reliability of our proposed CD method are verified on very-high-resolution (VHR) satellite image datasets. Extensive experimental results have shown that our proposed approach outperforms the other state-of-the-art CD methods


## Citation
Please cite our paper if you find it useful for your research.
```
@article{peng2019end,
  title={End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++},
  author={Peng, Daifeng and Zhang, Yongjun and Guan, Haiyan},
  journal={Remote Sensing},
  volume={11},
  number={11},
  pages={1382},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
