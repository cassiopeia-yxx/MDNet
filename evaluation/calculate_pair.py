import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
import pyiqa


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


def compute_metrics(metrics, img_out, img_gt, device):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(img_out, img_gt).item()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='F:\Python\LLIE&Deblur\\result\LOL\LOL_v2_real\CWNet',
                        help='Path to the model output images')
    parser.add_argument('--gt_path', type=str, default='F:\Python\LLIE&Deblur\\result\LOL\LOL_v2_real\GT',
                        help='Path to the ground truth images')
    parser.add_argument('--metrics', nargs='+', default=['psnr','ssim'],
                        help='List of IQA metrics to compute') #  ,'lpips','mad','vif'
    args = parser.parse_args()

    args.result_path = args.result_path.rstrip('/')
    args.gt_path = args.gt_path.rstrip('/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize selected metrics
    available_metrics = {}
    for metric_name in args.metrics:
        try:
            m = pyiqa.create_metric(metric_name).to(device).eval()
            available_metrics[metric_name] = m
        except Exception as e:
            print(f"Warning: Failed to load metric '{metric_name}': {e}")

    metric_scores = {k: [] for k in available_metrics}

    img_out_paths = sorted(list(scandir(args.result_path, suffix=('jpg', 'png'), recursive=True, full_path=True)))
    total_num = len(img_out_paths)

    for i, img_out_path in tqdm(enumerate(img_out_paths), total=total_num, desc='Calculating metrics'):
        img_name = img_out_path.replace(args.result_path + '/', '')
        try:
            img_out = cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            img_out = torch.from_numpy(np.transpose(img_out, (2, 0, 1))).unsqueeze(0).to(device)

            img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
            img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            img_gt = torch.from_numpy(np.transpose(img_gt, (2, 0, 1))).unsqueeze(0).to(device)

            with torch.no_grad():
                scores = compute_metrics(available_metrics, img_out, img_gt, device)
                for name, score in scores.items():
                    metric_scores[name].append(score)
        except Exception as e:
            print(f"skip: {img_name} due to error: {e}")
            continue

    print('\n------------------- Final Scores -------------------')
    for name, scores in metric_scores.items():
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f'{name.upper():<6}: {mean_score:.4f}')


if __name__ == '__main__':
    main()
