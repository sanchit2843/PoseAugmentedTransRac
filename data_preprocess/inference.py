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


label_map_updated = {
    0: "jump_jack",
    1: "battle_rope",
    2: "pommelhorse",
    3: "bench_pressing",
    4: "frontraise",
    5: "push_up",
    6: "squat",
    7: "situp",
    8: "pullups",
    9: "others",
    10: "None",
}


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
    heatmaps = []
    ## TODO: handle the last batch of the video in case the number of frames are less than 64
    current_count = 0
    class_prediction = 10
    while True:
        ret, frame = cap.read()
        if videos_first_batch_flag:
            out_video = cv2.VideoWriter(
                "./{}.mp4".format(video_path.split("/")[-1].split(".")[0]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap.get(cv2.CAP_PROP_FPS),
                (frame.shape[1], frame.shape[0]),
            )

        if not ret:
            break
        if frame_no % sampling_rate == 0:
            frames.append(frame)
            keypoint = extract_keypoint_image(frame, yolov7)
            heatmap = np.zeros((17, 56, 56))
            if len(keypoint) == 0:
                heatmaps.append(heatmap)
            else:
                heatmap = keypoint_to_heatmap(
                    heatmap,
                    np.array(keypoint),
                    (224, 224),
                    heatmap_size=heatmap.shape,
                )
                heatmaps.append(heatmap)

        frame_no += 1
        if len(frames) == 64:
            ## The sampling rate is dynamically adjusted based on the predicted class, since we dont have any class at t=0, we will run the prediction with default sampling rate, we will later change the sampling rate based on the predicted class. This technique works well as tested, but will fail miserably in case the predicted class is wrong.
            frames = (np.array(frames).transpose(0, 3, 1, 2) - 127.5) / 127.5
            frames = torch.tensor(frames).float().cuda()
            heatmap = torch.tensor(np.array(heatmaps)).float().cuda()
            videos_first_batch_flag = False

            ## TODO:
            # 1. get transrac prediction
            # 2. get class prediction
            # 3. adjust sampling rate based on class prediction, use stride_class_mapping.json
            # 4. Take the global count and plot this on the video, write this video to a new file.

            pred, _, class_prediction, _ = my_model(
                frames.unsqueeze(0), heatmap.unsqueeze(0)
            )
            predict_count = torch.sum(pred, dim=1).round()

            class_prediction = torch.argmax(class_prediction, dim=1)
            sampling_rate = sampling_class_mapping[str(class_prediction.item())]
            current_count += predict_count.item()
        output_txt = "Current Count: {}, Current predicted action {}".format(
            current_count, label_map_updated[class_prediction.item()]
        )
        cv2.putText(
            frame, output_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        out_video.write(frame)

    cap.release()
    out_video.release()
