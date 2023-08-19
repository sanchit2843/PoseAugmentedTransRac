## This is temp readme, change it to md

### Data preparation
Since we utilize human poses in additional stream, these human poses are extracted automatically using Yolov7 and no manual annotation was performed. Thus the first step for dataset preparation is preparing human pose estimation. 

1. Estimate the human poses for each frame in video and save a numpy array with frames, this numpy will have keys imgs and pose for images and pose heatmaps. 

`
`
bash
cd data_preprocess
python data_preprocess_merged.py --video_directory <path to all videos> --output_directory <path where jsons will be saved>
`

