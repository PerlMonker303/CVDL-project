import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model import UNet
from model_resunet import ResUNet
from model_resunetpp import ResUNetPlusPlus
from utils import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # do not modify

MODEL_PATH = 'saved_models/resunetpp/att10_3.pth.tar'
VIDEO_PATH = 'data_rail/videos/source1.mp4'
OUTPUT_PATH = 'data_rail/videos/1/att10_3.avi'
FRAME_DIM = (1280, 720)  # video dim
IMAGE_DIM = (512, 512)
FPS = 30.0
SEC = 75  # how much to take from the video
FULL_VIDEO = False
CROP = False  # show results only on the center of the image where the rails of interest should be
CROP_SIZE = 0.4
# 0.3 0.4 0.3


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, dsize=IMAGE_DIM, interpolation=cv2.INTER_LINEAR)
    frame = np.array(frame)
    frame = np.expand_dims(frame, 0)
    frame = np.expand_dims(frame, 0)
    frame = torch.from_numpy(frame).float()
    return frame


def postprocess_output(frame, output):
    output = torch.squeeze(output, 0)
    output = torch.squeeze(output, 0)
    output = output.cpu().detach().numpy()
    output = np.where(output > 0.5, 255, 0)
    output = output.astype('uint8')
    output = cv2.resize(output, dsize=FRAME_DIM, interpolation=cv2.INTER_LINEAR)
    # output = np.expand_dims(output, 0)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    if CROP:
        left_area = right_area = (1 - CROP_SIZE) / 2
        h, w, _ = output.shape
        left_start = 0
        left_end = int(w * left_area)
        right_start = int(w / 2 + w * right_area)
        right_end = int(w)
        output[:, left_start:left_end, :] = 0
        output[:, right_start:right_end, :] = 0

    # overlap segmentation over input frame
    output = cv2.addWeighted(frame, 1, output, 0.4, 0)

    # cv2.imshow('image', output)
    # cv2.waitKey()

    return output


if __name__ == "__main__":
    # instantiate the model
    # model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    # model = ResUNet(in_channels=1, out_channels=1).to(DEVICE)
    model = ResUNetPlusPlus(in_channels=1, out_channels=1).to(DEVICE)

    # load the model
    print(torch.cuda.is_available())
    load_checkpoint(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)), model)

    # load video
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    vidwriter = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, FRAME_DIM)  # add 0 param for grayscale results
    success, frame = vidcap.read()
    count = 0
    while success:
        print(count)
        # transform frame to input
        input = preprocess_frame(frame).to(DEVICE)
        output = model(input)
        output = postprocess_output(frame, output)
        vidwriter.write(output)

        # plt.imshow(output, cmap='gray', interpolation='bilinear')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        success, frame = vidcap.read()
        count += 1
        if count == int(FPS * SEC) and not FULL_VIDEO:
            break

    vidcap.release()
    vidwriter.release()