import os
import csv
import argparse
import torch
import numpy as np
import pyiqa
from pyiqa.utils.img_util import imread2tensor
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from tqdm import tqdm


def dict2csv(dic, filename):
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dic.keys()))
        writer.writeheader()
        for i in range(len(dic[next(iter(dic))])):
            row = {k: dic[k][i] for k in dic}
            writer.writerow(row)


def run_nr_metrics(img_dir, metric_names, device, output_csv):
    # 1) 准备图像列表
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    img_list = sorted(
        [fname for fname in os.listdir(img_dir) if fname.lower().endswith(valid_ext)]
    )
    imgs = []
    for name in img_list:
        path = os.path.join(img_dir, name)
        tensor = imread2tensor(path).unsqueeze(0).to(device)
        imgs.append(tensor)
    print(f">>>> Loaded {len(imgs)} images from {img_dir} onto {device}")

    # 2) 筛选出 NR 指标
    nr_metrics = []
    for name in metric_names:
        mode = DEFAULT_CONFIGS.get(name, {}).get("metric_mode", "")
        if mode == "NR":
            nr_metrics.append(name)
    if not nr_metrics:
        raise ValueError("No no-reference (NR) metrics found in your list.")
    print(f">>>> Evaluating NR metrics: {nr_metrics}")

    # 3) 逐指标计算
    all_results = {"image": img_list}
    with torch.no_grad():
        for m in nr_metrics:
            print(f">>>> Computing {m} ...")
            metric_fn = pyiqa.create_metric(m, as_loss=False, device=device)
            scores = []
            for img in tqdm(imgs):
                score = metric_fn(img).squeeze().cpu().item()
                scores.append(score)
            all_results[m] = scores
            avg, std = np.mean(scores), np.std(scores)
            print(f"{m}: avg={avg:.4f}, std={std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute no-reference IQA metrics with pyiqa"
    )
    parser.add_argument(
        "--img_dir",
        "-i",
        default=r"F:\\Python\\LLIE&Deblur\\URWKV\\results\\LOL_blur",
        help="Path to the folder containing test images",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=[
            "niqe",
            "brisque",
            "nrqm",
            "pi",
            "musiq",
            "arniqa",
        ],
        help="List of no-reference metric names to evaluate (default: ['niqe','liqe'])",
    )
    parser.add_argument(
        "--output_csv", "-o", default=None, help="Path to write the CSV results"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force using CPU even if CUDA is available",
    )
    args = parser.parse_args()

    # 如果用户没有指定 --output_csv，就放到 img_dir 下
    if args.output_csv is None:
        args.output_csv = os.path.join(args.img_dir, "nr_iqa_results.csv")

    device = torch.device(
        "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda"
    )

    metrics = args.metrics

    run_nr_metrics(args.img_dir, metrics, device, args.output_csv)
