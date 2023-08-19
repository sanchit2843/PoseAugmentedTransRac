## Introduction
This repository extends the original TransRac model by introducing a human pose-based stream for action recognition. The additional stream leverages the PoseC3D architecture, designed for action recognition using heatmaps of human poses as input. To enhance computational efficiency, the MoviNet backbone is incorporated. An extra branch is also introduced for action classification, aiding in dynamically adjusting the sampling rate and addressing issues related to sampling rate in untrimmed videos or live stream. This project was undertaken as the final assignment for CMSC733 at the University of Maryland. The complete report can be accessed here.

## RepCount Dataset   
The Homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html) is available now. 
## Usage  
### Install 
Please refer to [install.md](https://github.com/SvipRepetitionCounting/TransRAC/blob/main/install.md) for installation.

### Data preparation
Since we utilize human poses in additional stream, these human poses are extracted automatically using Yolov7 and no manual annotation was performed. Thus the first step for dataset preparation is preparing human pose estimation. 

1. Estimate the human poses for each frame in video and save a numpy array with frames, this numpy will have keys imgs and pose for images and pose heatmaps. 

```
cd data_preprocess
python data_preprocess_merged.py --video_directory <path to all videos> --output_directory <path where jsons will be saved>
```

Data directory should be in format 
```
├── root_directory
│   ├── train
│   ├── train.csv
│   ├── test
│   ├── test.csv
```


### Train   

Through experiments we discovered that it is necessary to first train the pose branch to get reasonable improvements, thus skeleton branch is first trained independently, these weights are later fine tuned in final multistream model. 
1. Train skeleton based model

`
python pretrain_skeleton.py --root_path <path where dataset exists>
`

2. Train the final model

` 
python train.py --root_ath <path where dataset is stored> --pose_checkpoint <path to the final trained model received from previous step>
`    

### Model Zoo
##### RepCount Dataset 

## Acknowledgements

```
@inproceedings{hu2022transrac,
  title={TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting},
  author={Hu, Huazhang and Dong, Sixun and Zhao, Yiqun and Lian, Dongze and Li, Zhengxin and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19013--19022},
  year={2022}
}
```
```
@inproceedings{duan2022revisiting,
  title={Revisiting skeleton-based action recognition},
  author={Duan, Haodong and Zhao, Yue and Chen, Kai and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2969--2978},
  year={2022}
}
```

```
@inproceedings{kondratyuk2021movinets,
  title={Movinets: Mobile video networks for efficient video recognition},
  author={Kondratyuk, Dan and Yuan, Liangzhe and Li, Yandong and Zhang, Li and Tan, Mingxing and Brown, Matthew and Gong, Boqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16020--16030},
  year={2021}
}
```

