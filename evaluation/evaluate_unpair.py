import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.utils import img2tensor, scandir
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

# 引入 pyiqa 相关的包
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS


def check_image_size(x, down_factor):
    """确保图像尺寸是下采样因子的整数倍，通过 padding 实现。"""
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


@torch.no_grad()
def inference_images(test_path, model, device, down_factor):
    """
    对指定路径下的所有图像进行模型推理（逐张处理）。
    返回图像名列表和原始的、未经任何转换的 float32 Tensor 结果列表。
    """
    img_paths = sorted(list(scandir(test_path, suffix=('jpg', 'png'), recursive=True, full_path=True)))
    results = []
    names = []

    for img_path in tqdm(img_paths, desc='Running inference'):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_tensor = img2tensor(img / 255., bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        
        H, W = img_tensor.shape[2:]
        img_tensor_padded = check_image_size(img_tensor, down_factor)
        
        out_tensor = model(img_tensor_padded)
        out_tensor = out_tensor[:, :, :H, :W]
        out_tensor = torch.clamp(out_tensor, 0.0, 1.0)
        
        results.append(out_tensor.cpu())
        names.append(os.path.basename(img_path))

    return names, results


@torch.no_grad()
def compute_metrics_with_simulation(raw_image_tensors, image_names, metric_names, device):
    """
    一个高度集成的函数（非批处理版本）：
    在计算每个图像的指标前，对其进行“即时”模拟转换。
    """
    nr_metrics = [m for m in metric_names if DEFAULT_CONFIGS.get(m, {}).get('metric_mode') == 'NR']
    if not nr_metrics:
        raise ValueError("在列表中没有找到有效的 No-Reference (NR) 指标。")

    print(f"\n>>> Evaluating NR metrics with on-the-fly simulation: {nr_metrics}")
    all_results = {}

    for m in nr_metrics:
        print(f"\n>>> Computing {m} ...")
        metric_fn = pyiqa.create_metric(m, as_loss=False, device=device)
        scores = []
        
        # 将两个循环合并为一个：遍历原始的推理结果
        for raw_tensor in tqdm(raw_image_tensors, desc=f'Metric: {m}'):
            
            # --- 步骤 1: 对当前这张图进行“即时”模拟转换 ---
            # 模拟保存：将 float32 Tensor [0,1] 转换为 uint8 NumPy 数组 [0,255]
            uint8_numpy = raw_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            uint8_numpy = (uint8_numpy * 255.0).round().astype(np.uint8)

            # 模拟读取：将 uint8 NumPy 数组转换回 float32 Tensor [0,1]
            reloaded_tensor = torch.from_numpy(uint8_numpy).permute(2, 0, 1).float() / 255.0
            reloaded_tensor = reloaded_tensor.unsqueeze(0)
            
            # --- 步骤 2: 立刻使用转换后的 tensor 计算指标 ---
            tensor_device = reloaded_tensor.to(device)
            score = metric_fn(tensor_device).squeeze().cpu().item()
            scores.append(score)
        
        avg, std = np.mean(scores), np.std(scores)
        all_results[m] = (avg, std)
        print(f"→ {m}: avg = {avg:.4f}, std = {std:.4f}")

    return all_results


if __name__ == '__main__':
    # --- 参数设置 ---
    test_path = '/home/data/dupf/ddd/real_blur' 
    ckpt_path = '/home/data/dupf/ddd/experiments/20250708_193835_bs8_C24_L1FFT_RthreeFFN(2.66)_v05noz_LFFN(2.66)_A800/models/net_g_245000.pth'
    metrics = ['niqe', 'brisque']
    down_factor = 8

    # --- 环境与设备设置 ---
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 加载模型 ---
    print(">>> Loading model...")
    net = ARCH_REGISTRY.get('ReDUNet')(n_feat=32, nums_stages=5).to(device)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)['params']
    net.load_state_dict(checkpoint, strict=True)
    net.eval()
    print(">>> Model loaded.")

    # --- 步骤 1: 运行模型推理 ---
    names, raw_outputs = inference_images(test_path, net, device, down_factor)

    # --- 步骤 2: "即时"模拟转换并计算所有指标 ---
    compute_metrics_with_simulation(raw_outputs, names, metrics, device)

    print("\n>>> All evaluation completed.")