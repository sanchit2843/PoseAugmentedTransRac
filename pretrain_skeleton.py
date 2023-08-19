"""train TransRAC model """
from platform import node
import os

# if your data is .mp4 form, please use RepCountA_raw_Loader.py (slowly)
# from dataset.RepCountA_raw_Loader import MyData
# if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from dataset.RepCountA_Loader import MyData

# you can use 'tools.video2npz.py' to transform .mp4 to .npz
# from models.TransRAC import TransferModel
# from models.TransRac_movinet import TransferModelMovinet
# from models.TransRac_singlescale_movinet import TransferModelMovinet_SS
# from models.TransRac_multistream import TransferModelMultiStream
from models.Transrac_pose import TransferModelPose
import wandb
from training.train_looping import train_loop

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/workspace/openset_path")
    args = parser.parse_args()
    # CUDA environment
    N_GPU = 1
    device_ids = [i for i in range(N_GPU)]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # # # we pick out the fixed frames from raw video file, and we store them as .npz file
    # # # we currently support 64 or 128 frames
    # data root path
    root_path = args.root_path

    train_video_dir = "train"
    train_label_dir = "train.csv"
    valid_video_dir = "test"
    valid_label_dir = "test.csv"

    # please make sure the pretrained model path is correct
    checkpoint = "./pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    config = "./configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"

    # TransRAC trained model checkpoint, we will upload soon.
    # "./checkpoint/ours/25_0.6321.pt"
    lastckpt = None  # "/mnt/workspace/UMD/CMSC733/CMSC733_project/ExerciseCountDomainAdapt/Transrac_original/checkpoint/swin-1-_-_-128-224-repcount/15_1.3207.pt"
    NUM_FRAME = 64
    # multi scales(list). we currently support 1,4,8 scale.
    SCALES = [8]
    num_classes = None  # use this if
    train_dataset = MyData(
        root_path,
        train_video_dir,
        train_label_dir,
        num_frame=NUM_FRAME,
        num_classes=num_classes,
        pose=True
    )
    valid_dataset = MyData(
        root_path,
        valid_video_dir,
        valid_label_dir,
        num_frame=NUM_FRAME,
        num_classes=num_classes,
        pose=True
    )
    pose = True

    my_model = TransferModelPose(
        num_frames=NUM_FRAME,
        scales=SCALES,
        OPEN=True,  ## required since we want to retrain the backbone for pose
    )
    my_model.train()
    NUM_EPOCHS = 200
    LR = 1e-4

    BATCH_SIZE = 32
    experiment_name = "multi-finalcorrected-openset"
    os.system("WANDB_MODE=run")
    os.system("wandb login {}".format("wandb_sample_key"))  ## add your wandb key here
    wandb.init(name=experiment_name, project="Repetition counting Multi Stream")

    train_loop(
        NUM_EPOCHS,
        my_model,
        train_dataset,
        valid_dataset,
        train=True,
        valid=True,
        batch_size=BATCH_SIZE,
        lr=LR,
        saveckpt=True,
        ckpt_name=experiment_name,
        log_dir=experiment_name,
        device_ids=device_ids,
        lastckpt=lastckpt,
        mae_error=False,
        num_classes=num_classes,
        pose=pose,
    )
