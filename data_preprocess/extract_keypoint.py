import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
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
from icecream import ic
import mmcv
import os


def extract_keypoint_image_model(image, device="cuda"):
    device = select_device("0")
    model = attempt_load("./yolov7-w6-pose.pt", map_location=device)
    _ = model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    image_ = image.copy()
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
    return output


def extract_keypoint_image(image, model, device="cuda"):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    image_ = image.copy()
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
    return output


def video_to_arr(video_path, image_size, sampled_indexes):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if success is False:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, image_size)  # format width, height
            frames.append(frame_rgb)
    cap.release()
    updated_frames = []
    for i in sampled_indexes:
        updated_frames.append(frames[i])
    return np.array(updated_frames).transpose(0, 3, 1, 2)


def write_json_coco(base_json, output, image_size, output_directory):
    base_json = mmcv.load(base_json)
    ann_id = 0
    for key in output.keys():
        image_dict = {"image_id": key, "width": image_size[1], "height": image_size[0]}
        base_json["images"].append(image_dict)
        for keypoint in output[key]:
            keypoint = convert_keypoint_size(image_size, keypoint[7:])
            keypoint_dict = {"image_id": key, "id": ann_id, "keypoints": keypoint}
            base_json["annotations"].append(keypoint_dict)
            ann_id += 1
    mmcv.dump(base_json, output_directory)


def frame_extract(video_path):
    v = mmcv.VideoReader(video_path)
    return v


def convert_keypoint_size(image_size, keypoint):
    keypoint = np.array(keypoint)
    keypoint = keypoint.reshape(-1, 3)
    keypoint[:, 0] = keypoint[:, 0] * image_size[1] / 640
    keypoint[:, 1] = keypoint[:, 1] * image_size[0] / 384
    keypoint = keypoint.reshape(-1)
    return list(keypoint)


def process_videos(model, video_path, output_path, base_json, device="cuda"):
    v = frame_extract(video_path)
    outputs = {}
    for idx, frame in enumerate(v):
        frame_ = cv2.resize(frame, (640, 384))
        output = extract_keypoint_image(frame_, model, device)
        outputs[idx] = output
    write_json_coco(base_json, outputs, frame.shape, output_path)
    return


# except:
#     ic("Error in processing video: ", video_path)


def main(arg):
    process_videos(*arg)


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./yolov7-w6-pose.pt")
    parser.add_argument(
        "--video_directory",
        type=str,
        default="/mnt/workspace/UMD/CMSC733/CMSC733_project/test_video1",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/mnt/workspace/UMD/CMSC733/CMSC733_project/test_video1",
    )
    parser.add_argument("--base_json", type=str, default="./base.json")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")
    device = select_device("0")
    model = attempt_load(args.model_path, map_location=device)
    _ = model.eval()

    task_list = []
    from tqdm import tqdm

    for split in ["valid"]:
        os.makedirs(os.path.join(args.output_directory, split + "_json"), exist_ok=True)
        for i in os.listdir(os.path.join(args.video_directory, split)):
            if i.endswith(".mp4") or i.endswith(".avi"):
                task_list.append(
                    (
                        model,
                        os.path.join(args.video_directory, split, i),
                        os.path.join(
                            args.output_directory,
                            split + "_json",
                            i.replace(i[-3:], "json"),
                        ),
                        args.base_json,
                        "cuda",
                    )
                )
    print(len(task_list))
    mmcv.track_parallel_progress(main, task_list, nproc=2)
