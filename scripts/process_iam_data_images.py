import cv2
import numpy as np
from tqdm import tqdm

from pathlib import Path
import multiprocessing as mp
import math
import os

os.chdir(r"D:\Projects\image_to_text")


def process_image(paths):
    try:
        ip_path, op_path = paths

        # read and remove 4th channel
        img = cv2.imread(str(ip_path))
        img = img[:, :, :3]

        # grayscale and initial cropping
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray[550:2772, 250:]

        # bolden for line detection
        img_gray_inv = 255 - img_gray
        img_blur = cv2.bilateralFilter(img_gray_inv, d=7, sigmaColor=300, sigmaSpace=50)
        dodgeV2 = lambda img_, mask: cv2.divide(img_, 255 - mask, scale=256)
        img_blend = dodgeV2(img_gray, img_blur)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = 255 - cv2.dilate(255 - img_blend, kernel, iterations=1)
        # gray = img_gray

        # line detection
        edges = cv2.Canny(gray, 80, 120)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 2, None, 30, 1)
        # for line in lines[0]:
        #     pt1 = (line[0],line[1])
        #     pt2 = (line[2],line[3])
        #     print(pt1, pt2)
        #     cv2.line(img_gray, pt1, pt2, (0,0,0), 5)

        # line cropping
        if lines is not None:
            line = lines[0][0]
            ly_crop = line[1]
            if ly_crop < int(img_gray.shape[0] / 4):
                img_gray = img_gray[ly_crop:, :]

        # resize and save image
        img_gray = cv2.resize(img_gray, (512, 512), interpolation=cv2.INTER_AREA).astype(np.uint8)
        cv2.imwrite(str(op_path), img_gray)

        return 1

    except Exception as e:
        print(e)
        return 0


if __name__ == "__main__":
    base_ips = [Path("data/IAM/formsA-D"), Path("data/IAM/formsE-H"), Path("data/IAM/formsI-Z")]
    base_op = Path("data/final/images")
    im_tups = []
    top = []
    for base_ip in base_ips:
        files = os.listdir(base_ip)
        for file in files:
            im_tups.append((base_ip / file, base_op / file))
    with mp.Pool(16) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_image, im_tups),
                total=len(im_tups),
                desc="Processing images",
            )
        )

    print(f"{op.count(1)} images processed successfully. {op.count(0)} images failed.")
