"""train or valid looping """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from ranger21 import Ranger21
from tools.my_tools import paint_smi_matrixs, density_map, plot_inference
from icecream import ic
import wandb
import random
from testing.test_looping import test_loop

seed = 1
torch.manual_seed(seed)  # random seed. We not yet optimization it.
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(1)

## TODO: add function to show pose obo as well at val time
def train_loop(
    n_epochs,
    model,
    train_set,
    valid_set,
    train=True,
    valid=True,
    inference=False,
    batch_size=1,
    lr=1e-6,
    ckpt_name="ckpt",
    lastckpt=None,
    saveckpt=False,
    log_dir="scalar",
    device_ids=[0],
    mae_error=False,
    num_classes=None,
    pose=True,
):
    device = torch.device(
        "cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu"
    )
    currEpoch = 0
    trainloader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=min(1, batch_size),
        worker_init_fn=seed_worker,
        generator=g,
    )
    validloader = DataLoader(
        valid_set,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=min(1, batch_size),
    )
    # model = nn.DataParallel(model.to(device), device_ids=device_ids)
    model = model.to(device)
    optimizer = Ranger21(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        num_epochs=n_epochs,
        num_batches_per_epoch=len(trainloader),
        use_madgrad=True,
        warmdown_active=False,
    )

    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    # )

    milestones = [i for i in range(0, n_epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.8
    )  # three step decay

    # scaler = GradScaler()

    if lastckpt is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint["epoch"]
        # # # load hyperparameters by pytorch
        # # # if change model
        # net_dict=model.state_dict()
        # state_dict={k: v for k, v in checkpoint.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # model.load_state_dict(net_dict, strict=False)

        # # # or don't change model
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    lossSL1 = nn.SmoothL1Loss()
    loss_classification = nn.CrossEntropyLoss()

    # loss weights
    lambda_classification = 0.1
    lambda_pose = 0

    for epoch in tqdm(range(currEpoch, n_epochs)):
        trainLosses = []
        validLosses = []
        trainLoss1 = []
        validLoss1 = []
        trainOBO = []
        validOBO = []
        poseOBO = []
        poseMae = []
        trainMAE = []
        validMAE = []
        predCount = []
        Count = []

        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0
            num_correct = 0
            num_samples = 0
            model.train()

            for batch_dict in pbar:
                # with autocast():
                input = batch_dict["video"]
                target = batch_dict["label"]
                class_label = batch_dict["class_label"]
                heatmap = None
                if pose:
                    heatmap = batch_dict["pose"]
                    heatmap = heatmap.type(torch.FloatTensor).to(device)

                optimizer.zero_grad()
                acc = 0
                input = input.type(torch.FloatTensor).to(device)
                density = target.type(torch.FloatTensor).to(device)
                class_label = class_label.type(torch.LongTensor).to(device)

                count = (
                    torch.sum(target, dim=1).type(torch.FloatTensor).round().to(device)
                )

                if pose:
                    output, pose_output, matrixs, class_prediction = model(
                        input, heatmap
                    )
                    loss_pose = lossMSE(pose_output, density)

                else:
                    output, matrixs, class_prediction, _ = model(input, heatmap)
                    loss_pose = 0
                predict_count = (
                    torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                )

                predict_density = output

                loss1 = lossMSE(predict_density, density)
                loss2 = lossSL1(predict_count, count)

                # loss2 = lossMSE(predict_count, count)
                loss3 = (
                    torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1))
                    / predict_count.flatten().shape[0]
                )  # mae

                loss = loss1 + lambda_pose * loss_pose
                if mae_error:
                    loss += loss3

                if num_classes:
                    # this is for the classification of exercise being performed.

                    loss_classification_value = loss_classification(
                        class_prediction, class_label
                    )
                    loss += lambda_classification * loss_classification_value

                class_prediction = torch.argmax(class_prediction.detach(), dim=1)

                num_correct += (class_prediction == class_label).sum()
                num_samples += class_prediction.size(0)
                # calculate MAE or OBO

                gaps = (
                    torch.sub(predict_count, count)
                    .reshape(-1)
                    .cpu()
                    .detach()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )
                for item in gaps:
                    if abs(item) <= 1:
                        acc += 1
                OBO = acc / predict_count.flatten().shape[0]

                trainOBO.append(OBO)
                MAE = loss3.item()
                trainMAE.append(MAE)
                train_loss = loss.item()
                train_loss1 = loss1.item()
                trainLosses.append(train_loss)
                trainLoss1.append(train_loss1)

                batch_idx += 1
                pbar.set_postfix(
                    {
                        "Epoch": epoch,
                        "loss_train": train_loss,
                        "Train MAE": MAE,
                        "Train OBO ": OBO,
                        "train accuracy": (num_correct / num_samples).item(),
                    }
                )
                if num_classes == None:
                    assert loss == loss1 + lambda_pose * loss_pose
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
            wandb.log(
                {
                    "Train Loss": np.mean(trainLosses),
                    "Train MAE": np.mean(trainMAE),
                    "Train OBO": np.mean(trainOBO),
                    "Train Accuracy": (num_correct / num_samples).item(),
                }
            )

            # if valid and epoch % 1 == 0:
            # num_correct = 0
            # num_samples = 0
            # best_mae = 100
            # best_obo = 0
            # with torch.no_grad():
            #     batch_idx = 0
            #     pbar = tqdm(validloader, total=len(validloader))
            #     model.eval()
            #     for batch_dict in pbar:
            #         # with autocast():
            #         acc = 0
            #         input = batch_dict["video"]
            #         target = batch_dict["label"]
            #         class_label = batch_dict["class_label"]
            #         input = input.type(torch.FloatTensor).to(device)
            #         density = target.type(torch.FloatTensor).to(device)
            #         class_label = class_label.type(torch.LongTensor).to(device)
            #         heatmap = None
            #         if pose:
            #             heatmap = batch_dict["pose"]
            #             heatmap = heatmap.type(torch.FloatTensor).to(device)
            #         count = (
            #             torch.sum(target, dim=1)
            #             .type(torch.FloatTensor)
            #             .round()
            #             .to(device)
            #         )
            #         ## TODO: Add pose based models obo and mae
            #         if pose:
            #             output, pose_output, matrixs, class_prediction = model(
            #                 input, heatmap
            #             )
            #             loss_pose = lossMSE(pose_output, density)

            #         else:
            #             output, matrixs, class_prediction = model(input, heatmap)
            #             loss_pose = 0

            #         predict_count = (
            #             torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
            #         )
            #         predict_density = output

            #         class_prediction = torch.argmax(class_prediction, dim=1)
            #         num_correct += (class_prediction == class_label).sum()
            #         num_samples += class_prediction.size(0)

            #         loss1 = lossMSE(predict_density, density)

            #         loss2 = lossSL1(predict_count, count)
            #         # loss2 = lossMSE(predict_count, count)
            #         loss3 = (
            #             torch.sum(
            #                 torch.div(torch.abs(predict_count - count), count + 1e-1)
            #             )
            #             / predict_count.flatten().shape[0]
            #         )  # mae
            #         loss = loss1
            #         if mae_error:
            #             loss += loss3
            #         gaps = (
            #             torch.sub(predict_count, count)
            #             .reshape(-1)
            #             .cpu()
            #             .detach()
            #             .numpy()
            #             .reshape(-1)
            #             .tolist()
            #         )
            #         for item in gaps:
            #             if abs(item) <= 1:
            #                 acc += 1
            #         OBO = acc / predict_count.flatten().shape[0]

            #         if pose:
            #             predicted_count_pose = (
            #                 (torch.sum(pose_output, dim=1))
            #                 .type(torch.FloatTensor)
            #                 .to(device)
            #             )
            #             gaps_pose = (
            #                 torch.sub(predicted_count_pose, count)
            #                 .reshape(-1)
            #                 .cpu()
            #                 .detach()
            #                 .numpy()
            #                 .reshape(-1)
            #                 .tolist()
            #             )
            #             acc = 0
            #             for item in gaps_pose:
            #                 if abs(item) <= 1:
            #                     acc += 1
            #             OBO_pose = acc / predicted_count_pose.flatten().shape[0]

            #             mae_pose = (
            #                 torch.sum(
            #                     torch.div(
            #                         torch.abs(predicted_count_pose - count),
            #                         count + 1e-1,
            #                     )
            #                 )
            #                 / predicted_count_pose.flatten().shape[0]
            #             ).item()
            #         else:
            #             OBO_pose = 0
            #             mae_pose = 0
            #         validOBO.append(OBO)
            #         poseOBO.append(OBO_pose)
            #         poseMae.append(mae_pose)
            #         MAE = loss3.item()
            #         validMAE.append(MAE)
            #         train_loss = loss.item()
            #         train_loss1 = loss1.item()
            #         validLosses.append(train_loss)
            #         validLoss1.append(train_loss1)

            #         batch_idx += 1
            #         # density_map_pred = density_map(
            #         #     predict_density[0], batch_idx, "pred")
            #         # density_map_gt = density_map(
            #         #     density[0], batch_idx, "gt")
            #         # merged_density_map = np.vstack(
            #         #     [density_map_pred, density_map_gt])
            #         # wandb.log(
            #         #     {"Density Map": [wandb.Image(merged_density_map)]})

            #     print("###############################################")
            #     print("")
            #     print(
            #         "Epoch",
            #         epoch,
            #         "valid loss: ",
            #         np.mean(validLosses),
            #         "valid MAE: ",
            #         np.mean(validMAE),
            #         "valid OBO: ",
            #         np.mean(validOBO),
            #         "valid accuracy: ",
            #         (num_correct / num_samples).item(),
            #         "pose OBO",
            #         np.mean(poseOBO),
            #         "pose MAE",
            #         np.mean(poseMae),
            #     )

            #     wandb.log(
            #         {
            #             "Valid Loss": np.mean(validLosses),
            #             "Valid MAE": np.mean(validMAE),
            #             "Valid OBO": np.mean(validOBO),
            #             "Valid Accuracy": (num_correct / num_samples).item(),
            #             "pose OBO": np.mean(poseOBO),
            #             "pose MAE": np.mean(poseMae),
            #         }
            #     )
            #     # TODO: Write code to plot density map for valid and gt and save it to wandb
        if not os.path.exists("checkpoint/{0}/".format(ckpt_name)):
            os.mkdir("checkpoint/{0}/".format(ckpt_name))

        if saveckpt:
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
            }

            torch.save(
                checkpoint,
                "checkpoint/{0}/".format(ckpt_name) + "best_{}.pt".format(epoch),
            )
            wandb.save(
                "checkpoint/{0}/".format(ckpt_name) + "best_{}.pt".format(epoch),
            )
        lastckpt = "checkpoint/{0}/".format(ckpt_name) + "best_{}.pt".format(epoch)
        print()
        test_loop(1, model, valid_set, lastckpt=lastckpt)
        scheduler.step()

        # if np.mean(validMAE) < best_mae or np.mean(validOBO) > best_obo:
        #     best_obo = np.mean(validOBO)
        #     best_mae = np.mean(validMAE)

    optimizer.show_schedule()
