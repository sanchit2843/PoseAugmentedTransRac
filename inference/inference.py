import torch
import os
import cv2
from models.TransRac_multistream import TransferModelMultiStream
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import (
    non_max_suppression_kpt,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import (
    output_to_keypoint,
    plot_skeleton_kpts,
    colors,
    plot_one_box_kpt,
)
from torchvision import transforms
import numpy as np
from einops import rearrange
import json

### TODO: Complete final inference code.


def extract_keypoint_image(image, model, device="cuda"):
    image_ = image.copy()
    image = letterbox(image, (256, 256), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)  # convert image data to device
    image = image.float()
    with torch.no_grad():  # get predictions
        output_data, _ = model(image)

    output_data = non_max_suppression_kpt(
        output_data.cpu(),  # Apply non max suppression
        # Conf. Threshold.
        0.25,
        0.65,  # IoU Threshold.
        # Number of classes.
        nc=model.yaml["nc"],
        # Number of keypoints.
        nkpt=model.yaml["nkpt"],
        kpt_label=True,
    )

    output = output_to_keypoint(output_data)
    if len(output):
        output = convert_keypoint_size(image_.shape, output[0][7:], image.shape[2:])
    return output


def convert_keypoint_size(image_size, keypoint, letterbox_size):
    keypoint = np.array(keypoint)
    keypoint = keypoint.reshape(-1, 3)
    keypoint[:, 0] = keypoint[:, 0] * image_size[1] / letterbox_size[1]
    keypoint[:, 1] = keypoint[:, 1] * image_size[0] / letterbox_size[0]
    keypoint = keypoint.reshape(-1)
    return list(keypoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="/workspace/movinet.pt")
    parser.add_argument("--video_path", type=str, default="/workspace/movinet.mp4")
    parser.add_argument("--yolo_pose_path", type=str, default="./yolov7-w6-pose.pt")
    args = parser.parse_args()
    my_model = TransferModelMultiStream(
        config=None,
        checkpoint=None,
        num_frames=64,
        scales=[8],
        OPEN=False,
        num_classes=None,
        training_flag=False,
    )
    my_model.load_state_dict(
        torch.load(
            args.weight_path,
        )["state_dict"]
    )
    my_model.eval()
    yolov7 = attempt_load(args.yolo_pose_path, map_location="cuda")
    _ = yolov7.eval()

    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_no = 0
    initial_sampling_rate = 4
    sampling_class_mapping = json.load(open("./stride_class_mapping.json"))
    videos_first_batch_flag = True
    sampling_rate = initial_sampling_rate

    ## TODO: handle the last batch of the video in case the number of frames are less than 64
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % sampling_rate == 0:
            frames.append(frame)
        frame_no += 1
        if len(frames) == 64 and videos_first_batch_flag:
            output = extract_keypoint_image(frames, yolov7)
            print(output.shape)
            videos_first_batch_flag = False
            break
