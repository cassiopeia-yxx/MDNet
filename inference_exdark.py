import os
import cv2
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.registry import ARCH_REGISTRY

def tensor_to_image(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor.cpu().squeeze(0))

def save_image(tensor, file_path):
    image = tensor_to_image(tensor)
    image.save(file_path)

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='/home/dupf/pd_code/Transformer/data/ExDark_VOC/ExDark')
    parser.add_argument('--result_path', type=str, default='/home/dupf/pd_code/Transformer/data/ExDark_VOC/JPEGImages')
    parser.add_argument('--ckpt', type=str, default='weights/net_g_500000.pth')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.result_path, exist_ok=True)

    # 初始化网络
    net = ARCH_REGISTRY.get('ReDUNet')(n_feat=32, nums_stages=5).to(device)
    checkpoint = torch.load(args.ckpt)['params']
    net.load_state_dict(checkpoint, strict=True)
    net.eval()
    down_factor = 8

    # 读取图像
    img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG'), recursive=True, full_path=True)))
    print(f"✅ 共找到图像 {len(img_paths)} 张(应为ExDark的7363张)")

    for img_path in tqdm(img_paths, desc='Processing images'):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # 图像预处理
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True).unsqueeze(0).to(device)

        with torch.no_grad():
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, down_factor)
            output_t = net(img_t)
            output_t = output_t[:, :, :H, :W]
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))

        # 保存图像（不创建子文件夹）
        save_restore_path = os.path.join(args.result_path, img_name)
        imwrite(output.astype('uint8'), save_restore_path)

        del output_t
        torch.cuda.empty_cache()

    print(f"\n🎉 所有增强图像已保存至：{args.result_path}")
