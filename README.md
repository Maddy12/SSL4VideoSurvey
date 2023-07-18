# SSL4VideoSurvey
A collection of works on self-supervised, deep-learning learning for video. The papers listed here refers to our survey:

[**Self-Supervised Learning for Videos: A Survey**](https://dl.acm.org/doi/abs/10.1145/3577925)

[*Madeline Chantry Schiappa*](https://www.linkedin.com/in/madelineschiappa/),
[*Yogesh Singh Rawat*](https://www.crcv.ucf.edu/person/rawat/),
[*Mubarak Shah*](https://www.crcv.ucf.edu/person/mubarak-shah/)

## Summary 
In this survey, we provide a review of existing approaches on self-supervised learning focusing on the video domain. We summarize these methods into four different categories based on their learning objectives: 1) *pretext tasks*, 2) *generative learning*, 3) *contrastive learning*, and 4) *cross-modal agreement*. We further introduce the commonly used datasets, downstream evaluation tasks, insights into the limitations of existing works, and the potential future directions in this area.

![Overview of publications](Figures/FullLandscape.png)
*Statistics of self-supervised (SSL) video representation learning research in recent years. From left to
right we show a) the total number of SSL related papers published in top conference venues, b) categorical
breakdown of the main research topics studied in SSL, and (c) modality breakdown of the main modalities
used in SSL. The year 2022 remains incomplete because a majority of the conferences occur later in the year.*

![Overview of publications related to Action Recognition](Figures/ActionRecongitionOverTime.png)
*Action recognition performance of models over time for different self-supervised strategies including
different modalities: video-only (V), video-text (V+T), video-audio (V+A), video-text-audio (V+T+A). More
recently, contrastive learning has become the most popular strategy.*

# Training Tasks
## Pre-Text Learning
### Action Recognition
*Downstream evaluation of action recognition on pretext self-supervised learning measured by
prediction accuracy. Top scores are in **bold**. Playback speed related tasks typically perform the best.*

| Model                                     | Subcategory                               | Visual Backbone | Pre-Train                    | UCF101            | HMDB51            |
|-------------------------------------------|-------------------------------------------|-----------------|------------------------------|-------------------|-------------------|
| [Geometry](https://ieeexplore.ieee.org/document/8578684)          | Appearance                                | AlexNet         | UCF101/HMDB51                | 54.10             | 22.60             |
| [Wang et al.](https://arxiv.org/abs/1904.03597)| Appearance                                | C3D             | UCF101                       | 61.20             | 33.40             |
| [3D RotNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf)    | Appearance                                | 3D R-18         | [MT](https://pubmed.ncbi.nlm.nih.gov/30802849/) | 62.90             | 33.70             |
| [VideoJigsaw](https://ieeexplore.ieee.org/document/8659002)       | Jigsaw                                    | CaffeNet        | Kinetics                     | 54.70             | 27.00             |
| [3D ST-puzzle](https://dl.acm.org/doi/10.1609/aaai.v33i01.33018545) | Jigsaw                                    | C3D             | Kinetics                     | 65.80             | 33.70             |
| [CSJ](https://www.ijcai.org/proceedings/2021/104)                     | Jigsaw                                    | R(2+3)D         | Kinetics+UCF101+HMDB51       | 79.50             | <u>52.60</u> |
| [PRP](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yao_Video_Playback_Rate_Perception_for_Self-Supervised_Spatio-Temporal_Representation_Learning_CVPR_2020_paper.pdf)                  | Speed                                     | R3D             | Kinetics                     | 72.10             | 35.00             |
| [SpeedNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.pdf)         | Speed                                     | S3D-G           | Kinetics                     | 81.10             | 48.80             |
| [Jenni et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730426.pdf)        | Speed                                     | R(2+1)D         | UCF101                       | <u>87.10</u> | 49.80             |
| [PacePred](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620494.pdf)         | Speed                                     | S3D-G           | UCF101                       | **87.10**    | **52.60**    |
| [ShuffleLearn](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_32) | Temporal Order  | AlexNet                      | UCF101            | 50.90             | 19.80 |
| [OPN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Unsupervised_Representation_Learning_ICCV_2017_paper.pdf)                   | Temporal Order                            | VGG-M           | UCF101                       | 59.80             | 23.80             |
| [O3N](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fernando_Self-Supervised_Video_Representation_CVPR_2017_paper.pdf)            | Temporal Order                            | AlexNet         | UCF101                       | 60.30             | 32.50             |
| [ClipOrder](https://ieeexplore.ieee.org/document/8953292)         | Temporal Order                            | R3D             | UCF101                       | 72.40             | 30.90             |

### Video Retreival
*Performance for the downstream video retrieval task with top scores for each category in **bold**. K/U/H indicates using all three datasets for pre-training, i.e. Kinetics, UCF101, and HMDB51.*

| Model                            | Category     | Subcategory          | Visual Backbone | Pre-train           | UCF101 R@5 | HMDB51 R@5 |
|----------------------------------|--------------|----------------------|-----------------|---------------------|-----------|-----------|
| [SpeedNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.pdf)   | Pretext      | Speed                | S3D-G           | Kinetics            | 28.10     | --        |
| [ClipOrder](https://ieeexplore.ieee.org/document/8953292)     | Pretext      | Temporal Order       | R3D             | UCF101              | 30.30     | 22.90     |
| [OPN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Unsupervised_Representation_Learning_ICCV_2017_paper.pdf)          | Pretext      | Temporal Order       | CaffeNet        | UCF101              | 28.70     | --        |
| [CSJ](https://www.ijcai.org/proceedings/2021/104)              | Pretext      | Jigsaw               | R(2+3)D         | K/U/H               | 40.50     | --        |
| [PRP](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yao_Video_Playback_Rate_Perception_for_Self-Supervised_Spatio-Temporal_Representation_Learning_CVPR_2020_paper.pdf)          | Pretext      | Speed                | R3D             | Kinetics            | 38.50     | 27.20     |
| [Jenni et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730426.pdf)| Pretext      | Speed                | 3D R-18         | Kinetics            | 48.50     | --        |
| [PacePred](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620494.pdf)  | Pretext      | Speed                | R(2+1)D         | UCF101              | **49.70**     | **32.20**     |

## Generative Learning

### Action Recognition
*Downstream action recognition evaluation for models that use a generative self-supervised pre-training approach. Top scores are in **bold***
| Model                              | Subcategory       | Visual Backbone         | Pre-train        | UCF101 | HMDB51 |
|------------------------------------|-------------------|-------------------------|------------------|--------|--------|
| [Mathieu et al.](https://arxiv.org/abs/1511.05440)  | Frame Prediction  | C3D                     | Sports1M          | 52.10  | --     |
| [VideoGan](https://dl.acm.org/doi/pdf/10.5555/3157096.3157165) | Reconstruction    | VAE                     | Flickr            | 52.90  | --     |
| [Liang et al.](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liang_Dual_Motion_GAN_ICCV_2017_paper.pdf)          | Frame Prediction  | LSTM                    | UCF101            | 55.10  | --     |
| [VideoMoCo](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_VideoMoCo_Contrastive_Video_Representation_Learning_With_Temporally_Adversarial_Examples_CVPR_2021_paper.pdf)          | Frame Prediction  | R(2+1)D                 | Kinetics          | 78.70  | 49.20  |
| [MemDPC-Dual](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480324.pdf)             | Frame Prediction  | R(2+3)D                 | Kinetics          | 86.10  | 54.50  |
| [Tian et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590069.pdf)          | Reconstruction    | 3D R-101                | Kinetics          | 88.10  | 59.00  |
| [VideoMAE](https://arxiv.org/abs/2203.12602)     | [MAE](https://arxiv.org/abs/2205.09113)               | [ViT-L](https://arxiv.org/abs/2010.11929) | ImageNet  | 91.3   | 62.6   |
| [MotionMAE](https://arxiv.org/abs/2210.04154)          | [MAE](https://arxiv.org/abs/2205.09113)               | [ViT-B](https://arxiv.org/abs/2010.11929) | Kinetics  | 96.3   | --     |


*Downstream evaluation of action recognition on  self-supervised learning measured by prediction accuracy for [Something-Something (SS)](https://developer.qualcomm.com/software/ai-datasets/something-something) and [Kinetics400 (Kinetics)](https://www.deepmind.com/open-source/kinetics). SS is a more temporally relevant dataset and therefore is more challenging.  Top scores for each category are in **bold** and second best scores \underline{underlined}.*

| Model                                          | Category     | Subcategory | Visual Backbone      | Pre-Train                           | SS   | Kinetics |
|-----------------------------------------------|--------------|-------------|---------------------|-------------------------------------|------|----------|
| [BEVT](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_BEVT_BERT_Pretraining_of_Video_Transformers_CVPR_2022_paper.pdf)                   | Generative   | [MAE](https://arxiv.org/abs/2205.09113)         | [SWIN-B](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)      | Kinetics+[ImageNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 71.4 | 81.1     |
| [MAE](https://arxiv.org/abs/2205.09113)           | Generative   | [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-H](https://arxiv.org/abs/2010.11929)    | Kinetics                            | 74.1 | 81.1     |
| [MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf)                | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [MViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf)       | Kinetics                            | 74.4 | **86.7**     |
| [VideoMAE](https://arxiv.org/abs/2203.12602)              | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-L](https://arxiv.org/abs/2010.11929)      | ImageNet                            | 75.3 | 85.1     |
| [MotionMAE](https://arxiv.org/abs/2210.04154)                     | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-B](https://arxiv.org/abs/2010.11929)     | Kinetics                            | **75.5** | 81.7     |

### Video Retreival
*Performance for the downstream video retrieval task with top scores for each category in **bold**. K/U/H indicates using all three datasets for pre-training, i.e. Kinetics, UCF101, and HMDB51.*
| Model                            | Category     | Subcategory          | Visual Backbone | Pre-train           | UCF101 R@5 | HMDB51 R@5 |
|----------------------------------|--------------|----------------------|-----------------|---------------------|-----------|-----------|
| [MemDPC-RGP](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480324.pdf)         | Generative   | Frame Prediction     | R(2+3)D         | Kinetics            | 40.40     | 25.70     |
| [MemDPC-Flow](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480324.pdf)         | Generative   | Frame Prediction     | R(2+3)D         | Kinetics            | **63.20**     | **37.60**     |

## Contrastive Learning
### Action Recognition
*Downstream evaluation of action recognition on  self-supervised learning measured by prediction accuracy for [Something-Something (SS)](https://developer.qualcomm.com/software/ai-datasets/something-something) and [Kinetics400 (Kinetics)](https://www.deepmind.com/open-source/kinetics). SS is a more temporally relevant dataset and therefore is more challenging.  Top scores for each category are in **bold** and second best scores \underline{underlined}.*

| Model                                          | Category     | Subcategory | Visual Backbone      | Pre-Train                           | SS   | Kinetics |
|-----------------------------------------------|--------------|-------------|---------------------|-------------------------------------|------|----------|
| [pSwaV](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)          | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 51.7 | 62.7     |
| [pSimCLR](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)      | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 52.0 | 62.0     |
| [pMoCo](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)         | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 54.4 | 69.0     |
| [pBYOL](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)          | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | **55.8** | **71.5**     |

# Evaluation Tasks
## Action Recognition (Kinetics and SS)
*Downstream evaluation of action recognition on  self-supervised learning measured by prediction accuracy for [Something-Something (SS)](https://developer.qualcomm.com/software/ai-datasets/something-something) and [Kinetics400 (Kinetics)](https://www.deepmind.com/open-source/kinetics). SS is a more temporally relevant dataset and therefore is more challenging.  Top scores for each category are in **bold** and second best scores \underline{underlined}.*

| Model                                          | Category     | Subcategory | Visual Backbone      | Pre-Train                           | SS   | Kinetics |
|-----------------------------------------------|--------------|-------------|---------------------|-------------------------------------|------|----------|
| [pSwaV](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)          | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 51.7 | 62.7     |
| [pSimCLR](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)      | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 52.0 | 62.0     |
| [pMoCo](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)         | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 54.4 | 69.0     |
| [pBYOL](https://openaccess.thecvf.com/content/CVPR2021/papers/Feichtenhofer_A_Large-Scale_Study_on_Unsupervised_Spatiotemporal_Representation_Learning_CVPR_2021_paper.pdf)          | Contrastive  | View Aug.   | [R-50](https://arxiv.org/abs/1512.03385)                | Kinetics                            | 55.8 | 71.5     |
| [BEVT](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_BEVT_BERT_Pretraining_of_Video_Transformers_CVPR_2022_paper.pdf)                   | Generative   | [MAE](https://arxiv.org/abs/2205.09113)         | [SWIN-B](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)      | Kinetics+[ImageNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 71.4 | 81.1     |
| [MAE](https://arxiv.org/abs/2205.09113)           | Generative   | [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-H](https://arxiv.org/abs/2010.11929)    | Kinetics                            | 74.1 | 81.1     |
| [MaskFeat](https://openaccess.thecvf.com/content/CVPR2022/papers/Wei_Masked_Feature_Prediction_for_Self-Supervised_Visual_Pre-Training_CVPR_2022_paper.pdf)                | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [MViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf)       | Kinetics                            | 74.4 | 86.7     |
| [VideoMAE](https://arxiv.org/abs/2203.12602)              | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-L](https://arxiv.org/abs/2010.11929)      | ImageNet                            | 75.3 | 85.1     |
| [MotionMAE](https://arxiv.org/abs/2210.04154)                     | Generative   |  [MAE](https://arxiv.org/abs/2205.09113)         | [ViT-B](https://arxiv.org/abs/2010.11929)     | Kinetics                            | 75.5 | 81.7     |


## Video Retrieval 
*Performance for the downstream video retrieval task with top scores for each category in **bold**. K/U/H indicates using all three datasets for pre-training, i.e. Kinetics, UCF101, and HMDB51.*
| Model                            | Category     | Subcategory          | Visual Backbone | Pre-train           | UCF101 R@5 | HMDB51 R@5 |
|----------------------------------|--------------|----------------------|-----------------|---------------------|-----------|-----------|
| [SpeedNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Benaim_SpeedNet_Learning_the_Speediness_in_Videos_CVPR_2020_paper.pdf)   | Pretext      | Speed                | S3D-G           | Kinetics            | 28.10     | --        |
| [ClipOrder](https://ieeexplore.ieee.org/document/8953292)     | Pretext      | Temporal Order       | R3D             | UCF101              | 30.30     | 22.90     |
| [OPN](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Unsupervised_Representation_Learning_ICCV_2017_paper.pdf)          | Pretext      | Temporal Order       | CaffeNet        | UCF101              | 28.70     | --        |
| [CSJ](https://www.ijcai.org/proceedings/2021/104)              | Pretext      | Jigsaw               | R(2+3)D         | K/U/H               | 40.50     | --        |
| [PRP](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yao_Video_Playback_Rate_Perception_for_Self-Supervised_Spatio-Temporal_Representation_Learning_CVPR_2020_paper.pdf)          | Pretext      | Speed                | R3D             | Kinetics            | 38.50     | 27.20     |
| [Jenni et al.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730426.pdf)| Pretext      | Speed                | 3D R-18         | Kinetics            | 48.50     | --        |
| [PacePred](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620494.pdf)  | Pretext      | Speed                | R(2+1)D         | UCF101              | **49.70**     | **32.20**     |
| [MemDPC-RGP](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480324.pdf)         | Generative   | Frame Prediction     | R(2+3)D         | Kinetics            | 40.40     | 25.70     |
| [MemDPC-Flow](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480324.pdf)         | Generative   | Frame Prediction     | R(2+3)D         | Kinetics            | **63.20**     | **37.60**     |
| [DSM](https://arxiv.org/abs/2009.05757)               | Contrastive  | Spatio-Temporal      | I3D             | Kinetics            | 35.20     | 25.90     |
| [IIC](https://arxiv.org/abs/2010.15464)        | Contrastive  | Spatio-Temporal      | R-18            | UCF101              | 60.90     | 42.90     |
| [SeLaVi](https://arxiv.org/pdf/2006.13662.pdf)           | Cross-Modal  | Video+Audio          | R(2+1)D         | Kinetics            | 68.60     | 47.60     |
| [CoCLR](https://proceedings.neurips.cc/paper/2020/file/3def184ad8f4755ff269862ea77393dd-Paper.pdf)   | Contrastive  | View Augmentation    | S3D-G           | UCF101              | 70.80     | 45.80     |
| [GDT](https://arxiv.org/abs/2003.04298)            | Cross-Modal  | Video+Audio          | R(2+1)D         | Kinetics            | **79.00**    | **51.70**     |

## Citation
```
@article{10.1145/3577925,
author = {Schiappa, Madeline C. and Rawat, Yogesh S. and Shah, Mubarak},
title = {Self-Supervised Learning for Videos: A Survey},
year = {2023},
issue_date = {December 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {55},
number = {13s},
issn = {0360-0300},
url = {https://doi.org/10.1145/3577925},
doi = {10.1145/3577925},
journal = {ACM Comput. Surv.},
month = {jul},
articleno = {288},
numpages = {37},
}

```
