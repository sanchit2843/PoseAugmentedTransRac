## This is merged script to preprocess videos for training. The steps followed are written below
"""
1. Convert videos to npz files for training and testing folders. 
2. Pose estimation for each of the frame, selected in previous step 
3. Convert pose to heatmap and add it to npz file. 
"""
import argparse
import numpy as np
import os
import cv2
import multiprocessing as mp
import torch
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


def keypoint_to_heatmap(arr, pose, image_size, sigma=0.6, heatmap_size=(56, 56)):
    """_summary_

    Args:
        pose (_type_): _description_
        image_size (_type_): _description_
        sigma (int, optional): _description_. Defaults to 1.
        heatmap_size (tuple, optional): _description_. Defaults to (56, 56).
    """
    EPS = 1e-3
    centers = pose.reshape(-1, 3)
    max_values = centers[:, 2]
    centers[:, 0] = centers[:, 0] * heatmap_size[2] / image_size[1]
    centers[:, 1] = centers[:, 1] * heatmap_size[1] / image_size[0]
    centers = centers.astype(np.int)
    for idx, (center, max_value) in enumerate(zip(centers, max_values)):
        if max_value < EPS:
            continue

        mu_x, mu_y = center[0], center[1]
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, heatmap_size[2])
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, heatmap_size[1])
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)
        if not (len(x) and len(y)):
            continue
        y = y[:, None]
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
        patch = patch * max_value
        arr[idx, st_y:ed_y, st_x:ed_x] = np.maximum(
            arr[idx, st_y:ed_y, st_x:ed_x], patch
        )
    return arr


def convert_keypoint_size(image_size, keypoint, letterbox_size):
    keypoint = np.array(keypoint)
    keypoint = keypoint.reshape(-1, 3)
    keypoint[:, 0] = keypoint[:, 0] * image_size[1] / letterbox_size[1]
    keypoint[:, 1] = keypoint[:, 1] * image_size[0] / letterbox_size[0]
    keypoint = keypoint.reshape(-1)
    return list(keypoint)


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


def video_to_arr(video_path, image_size, num_frames):
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
            ## TODO: Add code to directly create frames from sampled index to save ram.
    cap.release()
    sampled_indexes = consistent_sample(len(frames), num_frames)
    updated_frames = []
    for i in sampled_indexes:
        updated_frames.append(frames[i])
    return np.array(updated_frames).transpose(0, 3, 1, 2), sampled_indexes


def consistent_sample(video_length, num_frames):
    indexes = []
    for i in range(1, num_frames + 1):
        indexes.append(i * video_length // num_frames - 1)
    return indexes


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    # Plot the skeleton and keypointsfor coco datatset
    palette = np.array(
        [
            [255, 128, 0],
            [255, 153, 51],
            [255, 178, 102],
            [230, 230, 0],
            [255, 153, 255],
            [153, 204, 255],
            [255, 102, 255],
            [255, 51, 255],
            [102, 178, 255],
            [51, 153, 255],
            [255, 153, 153],
            [255, 102, 102],
            [255, 51, 51],
            [153, 255, 153],
            [102, 255, 102],
            [51, 255, 51],
            [0, 255, 0],
            [0, 0, 255],
            [255, 0, 0],
            [255, 255, 255],
        ]
    )

    skeleton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    pose_limb_color = palette[
        [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]
    ]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                # if conf < 0.5:
                #     continue
            cv2.circle(
                im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1
            )

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        if steps == 3:
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            # if conf1 < 0.5 or conf2 < 0.5:
            #     continue
        if pos1[0] % 640 == 0 or pos1[1] % 640 == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def main(
    video_path, output_path, image_size, num_frames, heatmap_size, nopose, model, sanity
):
    video_length = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    image_array, sampled_indexes = video_to_arr(
        video_path, (image_size, image_size), num_frames
    )

    if nopose:
        np.savez(
            output_path,
            imgs=image_array,
            fps=fps,
            sampled_indexes=np.array(sampled_indexes),
        )
        return

    keypoints = []

    heatmaps = []
    for image in image_array:
        heatmap = np.zeros((17, heatmap_size, heatmap_size), dtype=np.float32)
        keypoint = extract_keypoint_image(
            image.transpose(1, 2, 0), model, device="cuda"
        )

        ### Uncomment next 3 lines to plot the keypoints and save each image, currently supports saving with same name, make name dynamic to save all frames.
        # frame = image.transpose(1, 2, 0).copy()
        # plot_skeleton_kpts(frame, keypoint, 3)
        # cv2.imwrite("frame.jpg", frame)

        if len(keypoint) == 0:
            heatmaps.append(heatmap)
            continue

        heatmap = keypoint_to_heatmap(
            heatmap,
            np.array(keypoint),
            (image_size, image_size),
            heatmap_size=heatmap.shape,
        )
        heatmaps.append(heatmap)
    np.savez(
        output_path,
        pose=np.array(heatmaps),
        imgs=image_array,
        fps=fps,
        sampled_indexes=np.array(sampled_indexes),
    )
    if sanity:
        save_arrays_to_mp4(
            image_array,
            np.array(heatmaps),
            output_path.replace(output_path.split("/")[-2], "sanity"),
        )


def save_arrays_to_mp4(video, heatmap, output_path):
    video_writer = cv2.VideoWriter(
        output_path.replace(".npz", "_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (video.shape[2], video.shape[3]),
    )
    heatmap_writer = cv2.VideoWriter(
        output_path.replace(".npz", "_heatmap.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (heatmap.shape[2], heatmap.shape[3]),
    )
    for i in video:
        video_writer.write(i.transpose(1, 2, 0)[:, :, ::-1])
    for i in heatmap:
        i = cv2.cvtColor((np.sum(i, axis=0) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        heatmap_writer.write(i)
    video_writer.release()
    heatmap_writer.release()
    ic("saved", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_directory",
        type=str,
        default="/mnt/workspace/UMD/CMSC733/CMSC733_project/LLSP/video",
    )  ## This will be path to the directory where with subdirectories of videos
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/mnt/current_working_datasets/LLSP_npz/Repcount_npz_64_increased",
    )

    parser.add_argument("--nopose", action="store_true")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--heatmap_size", type=int, default=56)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--num_worker", type=int, default=4
    )  ## Decide based on number of vcpu and vram in gpu if using pose

    parser.add_argument(
        "--sanity", action="store_true"
    )  ## Will save mp4 of npz video and heatmap for sanity check

    ## This is the directory where the videos
    args = parser.parse_args()
    task_list = []
    if args.nopose is False:
        torch.multiprocessing.set_start_method("spawn")
        model = attempt_load("./yolov7-w6-pose.pt", map_location="cuda")
        _ = model.eval()

    for split in os.listdir(args.video_directory):
        if os.path.isdir(os.path.join(args.video_directory, split)) is False:
            continue
        os.makedirs(os.path.join(args.output_directory, split), exist_ok=True)
        if args.sanity:
            os.makedirs(os.path.join(args.output_directory, "sanity"), exist_ok=True)
        for video_name in os.listdir(os.path.join(args.video_directory, split)):
            if os.path.exists(
                os.path.join(
                    args.output_directory, split, video_name.replace(".mp4", ".npz")
                )
            ):
                continue
            task_list.append(
                (
                    os.path.join(args.video_directory, split, video_name),
                    os.path.join(
                        args.output_directory, split, video_name.replace(".mp4", ".npz")
                    ),
                    args.image_size,
                    args.num_frames,
                    args.heatmap_size,
                    args.nopose,
                    model,
                    args.sanity,
                )
            )
    ic(len(task_list))
    # pool = mp.Pool(args.num_worker)
    # pool.starmap(main, task_list)
    # pool.close()
    # pool.join()
    for task in task_list:
        print(task[0])
        main(*task)
