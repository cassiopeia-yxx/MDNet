import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
from utils import PSNR, calculate_ssim


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                return_path = entry.path if full_path else osp.relpath(entry.path, root)
                if suffix is None or return_path.endswith(suffix):
                    yield return_path
            elif entry.is_dir() and recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='results/LOLv2', help='Path to enhanced images')
    parser.add_argument('--gt_path', type=str, default='datasets/LOLv2/Real_captured/Test/high', help='Path to ground truth images')
    args = parser.parse_args()

    args.result_path = args.result_path.rstrip('/')
    args.gt_path = args.gt_path.rstrip('/')

    img_out_paths = sorted(list(scandir(args.result_path, suffix=('jpg', 'png'), recursive=True, full_path=True)))
    total_num = len(img_out_paths)

    score_psnr_all, score_ssim_all = [], []

    for i, img_out_path in tqdm(enumerate(img_out_paths), total=total_num, desc='Evaluating'):
        img_name = img_out_path.replace(args.result_path + '/', '')
        try:
            img_gt_path = img_out_path.replace(args.result_path, args.gt_path)

            img_out = cv2.imread(img_out_path).astype(np.float32) / 255.
            img_gt = cv2.imread(img_gt_path).astype(np.float32) / 255.

            # PSNR using utils.PSNR (input range [0, 1])
            psnr = PSNR(img_gt, img_out)
            score_psnr_all.append(psnr)

            # SSIM using utils.calculate_ssim (input range [0, 255])
            ssim = calculate_ssim((img_gt * 255).astype(np.uint8), (img_out * 255).astype(np.uint8))
            score_ssim_all.append(ssim)
        except Exception as e:
            print(f"Skip {img_name} due to error: {e}")
            continue

    mean_psnr = np.mean(score_psnr_all)
    mean_ssim = np.mean(score_ssim_all)

    print('\n------------------- Final Scores -------------------')
    print(f'PSNR : {mean_psnr:.4f}')
    print(f'SSIM : {mean_ssim:.4f}')


if __name__ == '__main__':
    main()
