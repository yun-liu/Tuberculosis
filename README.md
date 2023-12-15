## [TPAMI23] Revisiting Computer-Aided Tuberculosis Diagnosis

This is the official repository for Revisiting Computer-Aided Tuberculosis Diagnosis. 

Related links:
[[Official PDF Download]](https://mftp.mmcheng.net/Papers/23PAMI-Tuberculosis.pdf)
[[中译版全文]](https://mftp.mmcheng.net/Papers/23PAMI-Tuberculosis-cn.pdf)

### Requirements:

* torch==1.9.0
* torchvision==0.10.0
* mmcv==1.3.12

Run `pip install -v -e .` to install this repository.

### Introduction

"Revisiting Computer-Aided Tuberculosis Diagnosis" focuses on addressing challenges in tuberculosis (TB) diagnosis using deep learning. We introduces a large-scale dataset, TBX11K, which significantly enhances computer-aided TB diagnosis (CTD) capabilities. TBX11K contains 11,200 chest X-ray images with detailed annotations for TB infection areas. A novel framework, SymFormer, is proposed, leveraging bilateral symmetry in X-ray images for effective TB detection. SymFormer integrates Symmetric Search Attention and Symmetric Positional Encoding, significantly improving TB detection and classification. We also presents a benchmark for CTD, incorporating various evaluation metrics and baseline models, and launches an online challenge to facilitate future research in the field.

<center>
<img src="resources/teaser.jpg" width="500">
</center>

### Visualization
<center>
<img src="resources/vis.png" width="1000">
</center>

### TBX11K Dataset

#### Summary of publicly available TB datasets.
| Datasets                      | Pub. Year | \#Classes | Annotations  | \#Samples |
|-------------------------------|-----------|-----------|--------------|-----------|
| MC \cite{jaeger2014two}       | 2014      | 2         | Image-level  | 138       |
| Shenzhen \cite{jaeger2014two} | 2014      | 2         | Image-level  | 662       |
| DA \cite{chauhan2014role}     | 2014      | 2         | Image-level  | 156       |
| DB \cite{chauhan2014role}     | 2014      | 2         | Image-level  | 150       |
| TBX11K (Ours)                 | 2020&2023         | 4         | Bounding box | 11,200    |

#### Split for the TBX11K dataset

<table>
    <tr>
        <td></td>
        <td>Classes</td>
        <td>Train</td>
        <td>Val</td>
        <td>Test</td>
        <td>Total</td>
    </tr>
    <tr>
        <td rowspan="2">Non-TB</td>
        <td>Healthy</td>
        <td>3,000</td>
        <td>800</td>
        <td>1,200</td>
        <td>5,000</td>
    </tr>
    <tr>
        <td>Sick&Non-TB</td>
        <td>3,000</td>
        <td>800</td>
        <td>1,200</td>
        <td>5,000</td>
    </tr>
    <tr>
        <td rowspan="4">TB</td>
        <td>Active TB</td>
        <td>473</td>
        <td>157</td>
        <td>294</td>
        <td>924</td>
    </tr>
    <tr>
        <td>Latent TB</td>
        <td>104</td>
        <td>36</td>
        <td>72</td>
        <td>212</td>
    </tr>
    <tr>
        <td>Active&Latent TB</td>
        <td>23</td>
        <td>7</td>
        <td>24</td>
        <td>54</td>
    </tr>
    <tr>
        <td>Uncertain TB</td>
        <td>0</td>
        <td>0</td>
        <td>10</td>
        <td>10</td>
    </tr>
    <tr>
        <td colspan="2">Total</td>
        <td>6,600</td>
        <td>1,800</td>
        <td>2,800</td>
        <td>11,200</td>
    </tr>
</table>

Download TBX11K dataset: [TBX11K](https://mmcheng.net/tb/).

### SymFormer

#### CXR image classification results on the TBX11K test data.

| Methods                                  | Backbones        | Accuracy      | AUC (TB)      | Sensitivity   | Specificity   | Ave. Prec. (AP) | Ave. Rec. (AR) | Result  |
|------------------------------------------|------------------|---------------|---------------|---------------|---------------|-----------------|----------------|----------------|
| Deformable DETR | ResNet-50 w/ FPN | 91.3          | 97.6          | 89.2          | 95.3          | 89.8            | 91.0           | [[JSON]]('resources/result_files/Deformable.json')[[TXT]]('resources/result_files/Deformable.txt') |
| SymFormer w/ Deformable DETR             | ResNet-50 w/ FPN | 94.3          | 98.5          | 87.3          | 97.3 | 93.2            | 93.2           | [[JSON]]('resources/result_files/SymFormer_Deformable.json')[[TXT]]('resources/result_files/SymFormer_Deformable.txt') |
| SymFormer w/ RetinaNet                   | ResNet-50 w/ FPN | 94.5          | 98.9          | 91.0          | 96.8          | 93.3            | 94.0           | [[JSON]]('resources/result_files/SymFormer_RetinaNet_R50.json')[[TXT]]('resources/result_files/SymFormer_RetinaNet_R50.txt') |
| SymFormer w/ RetinaNet                   | P2T-Small w/ FPN | 94.6 | 99.1 | 92.1 | 96.7          | 93.4   | 94.2  | [[JSON]]('resources/result_files/SymFormer_RetinaNet_P2T.json')[[TXT]]('resources/result_files/SymFormer_RetinaNet_P2T.txt') |


| Methods                                  | Backbones        | \#FLOPs | \#Params | FPS  | $F_1$ score $\uparrow$ | TP/\#Total $\uparrow$ | TN/\#Total $\uparrow$ | FP/\#Total $\downarrow$ | FN/\#Total $\downarrow$ |
|------------------------------------------|------------------|---------|----------|------|------------------------|-----------------------|-----------------------|-------------------------|-------------------------|
| Deformable DETR | ResNet-50 w/ FPN | 54.07   | 52.67    | 23.0 | 85.6                   | 17.5                  | 76.6                  | 3.8                     | 2.1                     |
| SymFormer w/ Deformable DETR             | ResNet-50 w/ FPN | 54.08   | 52.69    | 22.5 | 87.9                   | 17.1                  | 78.2        | 2.2            | 2.5                     |
| SymFormer w/ RetinaNet                   | ResNet-50 w/ FPN | 59.14   | 50.03    | 24.3 | 89.0                   | 17.8                  | 77.8                  | 2.6                     | 1.8                     |
| SymFormer w/ RetinaNet                   | P2T-Small w/ FPN | 55.46   | 45.10    | 17.9 | 89.6          | 18.1         | 77.7                  | 2.7                     | 1.5            |


#### TB infection area detection results on our TBX11K test set.

<table>
    <tr>
        <td rowspan="2">Methods</td>
        <td rowspan="2">Test Data</td>
        <td rowspan="2">Backbones</td>
        <td colspan="2">Category-agnostic TB</td>
        <td colspan="2">Active TB</td>
        <td colspan="2">Latent TB</td>
    </tr>
    <tr>
        <td>AP<sup>50</sup><sub style="margin-left:-13px">bb</sub></td>
        <td>AP<sub>bb</sub></td>
        <td>AP<sup>50</sup><sub style="margin-left:-13px">bb</sub></td>
        <td>AP<sub>bb</sub></td>
        <td>AP<sup>50</sup><sub style="margin-left:-13px">bb</sub></td>
        <td>AP<sub>bb</sub></td>
    </tr>
    <tr>
        <td>Deformable DETR</td>
        <td rowspan="4">ALL</td>
        <td>ResNet-50 w/ FPN</td>
        <td>51.7</td>
        <td>22.0</td>
        <td>48.9</td>
        <td>21.2</td>
        <td>7.1</td>
        <td>1.9</td>
    </tr>
    <tr>
        <td>SymFormer w/ Deformable DETR</td>
        <td>ResNet-50 w/ FPN</td>
        <td>57.0</td>
        <td>23.3</td>
        <td>52.1</td>
        <td>22.7</td>
        <td>7.1</td>
        <td>2.0</td>
    </tr>
    <tr>
        <td>SymFormer w/ RetinaNet</td>
        <td>ResNet-50 w/ FPN</td>
        <td>68.0</td>
        <td>29.5</td>
        <td>62.0</td>
        <td>27.3</td>
        <td>13.3</td>
        <td>4.4</td>
    </tr>
    <tr>
        <td>SymFormer w/ RetinaNet</td>
        <td>P2T-Small w/ FPN</td>
        <td>70.4</td>
        <td>30.0</td>
        <td>63.6</td>
        <td>26.9</td>
        <td>11.4</td>
        <td>4.3</td>
    </tr>
    <tr>
        <td>Deformable DETR</td>
        <td rowspan="4">Only TB</td>
        <td>ResNet-50 w/ FPN</td>
        <td>57.4</td>
        <td>24.2</td>
        <td>54.5</td>
        <td>23.5</td>
        <td>7.6</td>
        <td>2.3</td>
    </tr>
    <tr>
        <td>SymFormer w/ Deformable DETR</td>
        <td>ResNet-50 w/ FPN</td>
        <td>60.8</td>
        <td>24.5</td>
        <td>55.2</td>
        <td>23.8</td>
        <td>9.2</td>
        <td>2.6</td>
    </tr>
    <tr>
        <td>SymFormer w/ RetinaNet</td>
        <td>ResNet-50 w/ FPN</td>
        <td>73.4</td>
        <td>31.5</td>
        <td>67.1</td>
        <td>29.2</td>
        <td>14.7</td>
        <td>4.8</td>
    </tr>
    <tr>
        <td>SymFormer w/ RetinaNet</td>
        <td>P2T-Small w/ FPN</td>
        <td>75.7</td>
        <td>32.1</td>
        <td>68.9</td>
        <td>28.9</td>
        <td>13.0</td>
        <td>4.7</td>
    </tr>
</table>


### Train
Download pretrained model: [P2T_small](https://drive.google.com/file/d/1FlwhyVKw0zqj2mux248gIQFQ8DGPi8rS/view?usp=sharing).

Use the following commands to train `SymFormer`:

```bash
# train detection
CUDA_VISIBLE_DEVICE=0 python tools/train.py \
    configs/symformer/symformer_retinanet_p2t_fpn_2x_TBX11K.py \
    --work-dir work_dirs/symformer_retinanet_p2t/ \
    --no-validate

# train classification
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py \
    --work-dir work_dirs/symformer_retinanet_p2t_cls/ \
    --no-validate
```

### Test

Use the following commands to generate results on TBX11K test dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py \
    configs/symformer/symformer_retinanet_p2t_cls_fpn_1x_TBX11K.py \
    work_dirs/symformer_retinanet_p2t_cls/latest.pth \
    --out work_dirs/symformer_retinanet_p2t_cls/result/result.pkl \
    --format-only --cls-filter True \
    --options "jsonfile_prefix=work_dirs/symformer_retinanet_p2t_cls/result/bbox_result" \
    --txt work_dirs/symformer_retinanet_p2t_cls/result/cls_result.txt
```

### Online Challenge
We only release the training and validation sets of the proposed TBX11K dataset. The test set is retained as an online challenge for simultaneous TB X-ray classification and TB area detection in a single system (e.g., a convolutional neural network). To participate this challenge, you need to create an account on [CodaLab](https://codalab.lisn.upsaclay.fr/) and register for the [TBX11K Tuberculosis Classification and Detection Challenge](https://codalab.lisn.upsaclay.fr/competitions/7916). Please refer to this [webpage](https://codalab.lisn.upsaclay.fr/competitions/7916#learn_the_details-evaluation) or our paper to see the evaluation metrics. Then, open the “Participate” tab to read the [submission guidelines](https://codalab.lisn.upsaclay.fr/competitions/7916#participate) carefully. Next, you can upload your submission. Once uploaded, your submissions will be evaluated automatically.


### Citation

If you are using the code/model/data provided here in a publication, please consider citing our works:

````
@inproceedings{liu2020rethinking,
  title={Rethinking computer-aided tuberculosis diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Ban, Yunfeng and Wang, Huifang and Cheng, Ming-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2646--2655},
  year={2020}
}

@article{wu2022p2t,
  title={P2T: Pyramid pooling transformer for scene understanding},
  author={Wu, Yu-Huan and Liu, Yun and Zhan, Xin and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@article{liu2023revisiting,
  title={Revisiting Computer-Aided Tuberculosis Diagnosis},
  author={Liu, Yun and Wu, Yu-Huan and Zhang, Shi-Chen and Liu, Li and Wu, Min and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}
````

### License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Non-Commercial use only. Any commercial use should get formal permission first.
