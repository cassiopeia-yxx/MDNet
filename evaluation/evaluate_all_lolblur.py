import os
import glob
import argparse
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from basicsr.utils import scandir, img2tensor
from basicsr.utils.registry import ARCH_REGISTRY
import pyiqa
from tqdm import tqdm

def pad_to_align(x, factor):
    """Pad tensor so that height and width are multiples of factor."""
    _, _, h, w = x.size()
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

def load_model(checkpoint_path, device):
    """Instantiate and load network from checkpoint."""
    # Replace 'ReDUNet' with your network class name
    net = ARCH_REGISTRY.get('ReDUNet')(n_feat=32, nums_stages=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)['params']
    net.load_state_dict(checkpoint, strict=True)
    net.eval()
    return net

def evaluate_one(model, input_dir, gt_dir, metrics, align_factor, device):
    """Run inference and compute metrics for all images in input_dir."""
    calculators = {}
    results = {m: [] for m in metrics}
    for m in metrics:
        calc = pyiqa.create_metric(m if m!='lpips' else 'lpips-vgg').to(device)
        calc.eval()
        calculators[m] = calc

    img_paths = sorted(scandir(input_dir, suffix=('jpg','png'), recursive=True, full_path=True))
    for path in tqdm(img_paths, desc='Images', unit='img'):
        # load low-quality image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        inp = img2tensor(img / 255., bgr2rgb=True, float32=True).unsqueeze(0).to(device)
        H, W = inp.shape[2:]
        inp = pad_to_align(inp, align_factor)

        # inference
        with torch.no_grad():
            out = model(inp)[:, :, :H, :W]
            out = torch.clamp(out, 0, 1)
        # 3. simulate save→read and RGB→BGR via OpenCV
        #    out: [1,C,H,W] float in [0,1]
        out_np = out.squeeze(0).cpu().numpy().transpose(1,2,0)      # → H×W×C RGB float
        out_uint8 = (out_np * 255.0).round().astype('uint8')        # → uint8 RGB
        out_bgr_uint8 = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)  # → uint8 BGR
        out_bgr = out_bgr_uint8.astype('float32') / 255.0           # → float BGR in [0,1]
        out_tensor = torch.from_numpy(out_bgr.transpose(2,0,1))     # → C×H×W
        out_tensor = out_tensor.unsqueeze(0).to(device)             # → 1×C×H×W

        # 4. load & preprocess GT exactly like calculate_pair.py (BGR→CHW)
        gt_path = path.replace(input_dir, gt_dir)
        gt_np = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        gt_tensor = torch.from_numpy(gt_np.transpose(2,0,1)).float().unsqueeze(0).to(device)

        # compute metrics
        for m in metrics:
            score = calculators[m](out_tensor, gt_tensor).item()
            results[m].append(score)

    # average over dataset
    return {m: float(np.mean(results[m])) for m in metrics}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate multiple checkpoints on a test set')
    parser.add_argument('--input_dir', type=str, default='/home/data/dupf/ddd/LOLBlur/test/low_blur_noise', help='Folder with low-quality/test images')
    parser.add_argument('--gt_dir', type=str, default='/home/data/dupf/ddd/LOLBlur/test/high_sharp_scaled', help='Folder with ground-truth images')
    parser.add_argument('--ckpt_dir', type=str, default='weights', help='Directory containing .pth checkpoint files')
    parser.add_argument('--metrics', nargs='+', default=['psnr','ssim','lpips','mad','vif'], help='List of metrics to compute (e.g., psnr ssim lpips)')
    parser.add_argument('--align_factor', type=int, default=8, help='Image size alignment factor for the network')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_list = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pth')))
    # 先写入表头（覆盖旧文件）
    header = ['Checkpoint'] + [m.upper() for m in args.metrics]
    output_file = 'evaluation_results.txt'
    with open(output_file, 'w') as f:
        f.write('  '.join(header) + '\n')

    # 每跑完一个 ckpt，就往文件里追加一行
    for ckpt in ckpt_list:
        name = os.path.basename(ckpt)
        print(f'\n>> Evaluating checkpoint: {name}')
        net = load_model(ckpt, device)
        scores = evaluate_one(net, args.input_dir, args.gt_dir, args.metrics, args.align_factor, device)
        row = [name] + [f'{scores[m]:.4f}' for m in args.metrics]
        print('  '.join([f'{k.upper()}: {v:.4f}' for k,v in scores.items()]))

        # 追加到文件并立即关闭（tmux 里实时可见）
        with open(output_file, 'a') as f:
            f.write('  '.join(row) + '\n')

    print(f'\nResults are being appended to {output_file} as each checkpoint finishes.')