""" test of TransRAC """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs, plot_inference, density_map
import cv2
from models.TransRac_multistream import TransferModelMultiStream

torch.manual_seed(1)


def load_npz(path_to_npz):
    npz = np.load(path_to_npz)
    video = npz["imgs"]
    heatmap = npz["pose"].transpose(1, 0, 2, 3)
    frames = torch.FloatTensor(video)
    frames -= 127.5
    frames /= 127.5
    return frames, torch.from_numpy(heatmap)


def inference_loop_npz(
    model,
    path_to_npz,
    inference=True,
    lastckpt=None,
    paint=True,
    device_ids=[0],
):
    device = torch.device(
        "cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    if lastckpt is not None:
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        del checkpoint

    pose = True
    with torch.no_grad():
        batch_idx = 0
        # pbar = tqdm(testloader, total=len(testloader))
        model.eval()
        import time

        density_maps = []
        for npz in os.listdir(path_to_npz):
            # with autocast():
            st = time.time()
            input, heatmap = load_npz(os.path.join(path_to_npz, npz))
            input = input.type(torch.FloatTensor).to(device)
            heatmap = heatmap.type(torch.FloatTensor).to(device)
            output, sim_matrix, class_ = model(
                input.unsqueeze(0).transpose(1, 2), heatmap.unsqueeze(0)
            )
            density_maps.append(output)
            torch.cuda.synchronize()
            print(time.time() - st)
            print(torch.argmax(class_, dim=1))
            predict_count = torch.sum(output, dim=1).round()

            print("predicted count for video {0} is {1}".format(npz, predict_count))
            if paint:
                density_map(output, batch_idx, "pred")
                pred = cv2.imread(
                    "density_map_test/{0}_{1}.png".format("pred", batch_idx)
                )
                paint_smi_matrixs(sim_matrix)
            batch_idx += 1
        torch.save(torch.stack(density_maps), "density_maps.pt")


if __name__ == "__main__":
    my_model = TransferModelMultiStream(
        config=None,
        checkpoint=None,
        num_frames=64,
        scales=[8],
        OPEN=False,
        num_classes=None,
        training_flag=False,
    )
    inference_loop_npz(
        my_model,
        "/mnt/workspace/UMD/CMSC733/CMSC733_project/test_video1/valid_merged",
        lastckpt="./checkpoint/best_142_0.4137580722570419_0.3082051282051282.pt",
        paint=True,
    )
