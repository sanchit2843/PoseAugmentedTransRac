""" 
Repcount data loader from fixed frames file(.npz) which will be uploaded soon.
if you don't pre-process the data file,for example,your raw file is .mp4,
you can use the *RepCountA_raw_Loader.py*(slowly).
or
you can use 'tools.video2npz.py' to transform .mp4 tp .npz
"""
import csv
import os
import os.path as osp
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
import torch

try:
    from .label_norm import normalize_label
except:
    from label_norm import normalize_label

from icecream import ic

label_map = {
    "jumpjacks": 0,
    "battle_rope": 1,
    "pommelhorse": 2,
    "jump_jack": 0,
    "bench_pressing": 3,
    "frontraise": 4,
    "push_up": 5,
    "squat": 6,
    "front_raise": 4,
    "pushups": 5,
    "situp": 7,
    "pullups": 8,
    "benchpressing": 3,
    "squant": 6,
    "pull_up": 8,
    "others": 9,
}


class MyData(Dataset):
    def __init__(
        self, root_path, video_path, label_path, num_frame, num_classes, pose=True
    ):
        """
        :param root_path: root path
        :param video_path: video child path (folder)
        :param label_path: label child path(.csv)
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(self.video_path)
        self.label_dict, self.class_label_dict = get_labels_dict(
            self.label_path
        )  # get all labels
        self.num_classes = num_classes
        self.pose = pose
        self.num_frame = num_frame

    def __getitem__(self, inx):
        """get data item
        :param  video_tensor, label
        """
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        get_frame_return_tuple = get_frames(file_path, self.pose)
        if self.pose:
            (
                video_tensor,
                pose_tensor,
                video_frame_length,
                sampled_indexes,
            ) = get_frame_return_tuple
        else:
            video_tensor, video_frame_length, sampled_indexes = get_frame_return_tuple

        if video_tensor is None:
            ic("video_tensor is None", video_file_name)
        # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        video_tensor = video_tensor.transpose(0, 1)
        if video_file_name in self.label_dict.keys():
            time_points = self.label_dict[video_file_name]
            if self.num_classes:
                class_label = label_map[self.class_label_dict[video_file_name]]
            else:
                class_label = 1
            label = preprocess(
                video_frame_length,
                time_points,
                num_frames=self.num_frame,
                sampled_indexes=sampled_indexes,
            )
            label = torch.tensor(label).type(torch.float32)
            return_dict = {
                "video": video_tensor,
                "label": label,
                "class_label": class_label,
            }
            if self.pose:
                # ic(inx)
                # ic(pose_tensor.shape, video_tensor.shape, label.shape, class_label, inx)
                return_dict["pose"] = pose_tensor

            return return_dict
        else:
            print(video_file_name, "not exist")
            return

    def __len__(self):
        """:return the number of video"""
        return len(self.video_dir)


def get_frames(npz_path, pose):
    # get frames from .npz files

    with np.load(npz_path) as data:
        sampled_indexes = data["sampled_indexes"]
        frames_length = data["fps"].item()
        frames = data["imgs"]  # numpy.narray [64, 3, 224, 224]
        # the raw video(.mp4) total frames number
        frames = torch.FloatTensor(frames)
        frames -= 127.5
        frames /= 127.5

        if pose:
            heatmap = data["pose"].transpose(1, 0, 2, 3)
            return frames, heatmap, frames_length, sampled_indexes
        return frames, frames_length, sampled_indexes


def get_labels_dict(path):
    # read label.csv to RAM
    labels_dict = {}
    class_label_dict = {}
    check_file_exist(path)
    with open(path, encoding="utf-8") as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            for k, v in row.items():
                if row[k] == None:
                    row[k] = ""
            cycle = [
                int(float(row[key]))
                for key in row.keys()
                if "L" in key and row[key] != ""
            ]
            class_ = row["type"]
            if not row["count"]:
                print(row["name"] + "error")
            else:
                labels_dict[row["name"].split(".")[0] + str(".npz")] = cycle
                class_label_dict[row["name"].split(".")[0] + str(".npz")] = class_
    return labels_dict, class_label_dict


def preprocess(video_frame_length, time_points, num_frames, sampled_indexes):
    """
    process label(.csv) to density map label
    Args:
        video_frame_length: video total frame number, i.e 1024frames
        time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
        num_frames: 64
    Returns: for example [0.1,0.8,0.1, .....]
    """
    new_crop = []
    for i in range(len(time_points)):  # frame_length -> 64
        item = min(
            math.ceil(
                (float((time_points[i])) / float(video_frame_length)) * num_frames
            ),
            num_frames - 1,
        )
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)
    return label


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


if __name__ == "__main__":

    # # # we pick out the fixed frames from raw video file, and we store them as .npz file
    # # # we currently support 64 or 128 frames
    # data root path
    root_path = "/mnt/current_working_datasets/LLSP_npz/Repcount_npz_64"

    train_video_dir = "valid"
    train_label_dir = "valid.csv"

    # valid_video_dir = "test_heatmap"
    # valid_label_dir = "test.csv"
    NUM_FRAME = 64
    train_dataset = MyData(
        root_path,
        train_video_dir,
        train_label_dir,
        num_frame=NUM_FRAME,
        num_classes=None,
        pose=True,
    )
    for i in range(len(train_dataset)):
        # print(i, train_dataset[i]["video"].shape)
        train_dataset[i]
