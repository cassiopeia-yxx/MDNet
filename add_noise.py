import cv2
import numpy as np
import os

# 添加高斯噪声（使用 OpenCV）
def add_gaussian_noise(image, mean=0, sigma=25):
    """ 添加高斯噪声 """
    row, col, ch = image.shape
    gauss = np.zeros((row, col, ch), dtype=np.float32)
    cv2.randn(gauss, mean, sigma)  # 使用OpenCV生成高斯噪声
    noisy = cv2.add(image.astype(np.float32), gauss)
    noisy = np.clip(noisy, 0, 255)  # 保证值在有效范围内
    return noisy.astype(np.uint8)

# 添加椒盐噪声（优化版：使用 Numpy 矩阵操作提升几百倍速度）
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """ 添加椒盐噪声 """
    output = image.copy()
    row, col, _ = output.shape
    
    # 生成一个与图像宽高相同的随机概率矩阵
    rand_matrix = np.random.rand(row, col)
    
    # 小于 salt_prob 的位置变成白色 (255, 255, 255)
    output[rand_matrix < salt_prob] = [255, 255, 255]
    # 大于 1 - pepper_prob 的位置变成黑色 (0, 0, 0)
    output[rand_matrix > 1 - pepper_prob] = [0, 0, 0]
    
    return output

# 添加运动模糊
def add_motion_blur(image, kernel_size=5):
    """ 添加运动模糊 """
    kernel = np.zeros((kernel_size, kernel_size))
    # 创建水平方向的运动模糊核
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

# 处理图像并保存
def process_images(input_path, output_path, noise_type='gaussian', blur_type='mild'):
    """ 处理图像并保存结果 """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 模糊强度设置
    if blur_type == 'mild':
        blur_size = 5
    elif blur_type == 'severe':
        blur_size = 15
    else:
        raise ValueError("模糊类型无效, 应该是 'mild' 或 'severe'")

    # 噪声类型设置
    if noise_type == 'gaussian':
        noise_function = add_gaussian_noise
    elif noise_type == 'salt_and_pepper':
        noise_function = add_salt_and_pepper_noise
    else:
        raise ValueError("噪声类型无效, 应该是 'gaussian' 或 'salt_and_pepper'")

    # 定义允许读取的图片格式，防止读取隐藏文件导致报错
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

    # 读取图像并处理
    for filename in os.listdir(input_path):
        # 过滤掉非图片文件
        if not filename.lower().endswith(valid_extensions):
            continue

        file_path = os.path.join(input_path, filename)
        image = cv2.imread(file_path)

        # 安全保护：如果图片损坏或读取失败，跳过而不是让程序崩溃
        if image is None:
            print(f"⚠️ 警告: 无法读取图像 {filename}，已跳过。")
            continue

        # 如果是单通道灰度图像，转换为三通道BGR以便统一处理
        if len(image.shape) == 2:  
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 1. 添加噪声
        noisy_image = noise_function(image)

        # 2. 添加运动模糊
        blurred_image = add_motion_blur(noisy_image, kernel_size=blur_size)

        # 3. 保存处理后的图像
        output_filename = os.path.join(output_path, filename)
        cv2.imwrite(output_filename, blurred_image)
        print(f"✅ 成功处理并保存: {filename}")

if __name__ == "__main__":
    input_dir = "F:\BaiduNetdiskDownload\Exdark\JPEGImages\IMGS"  
    output_dir = "F:\BaiduNetdiskDownload\Exdark\JPEGImages\IMGS_with_gaussian_severe"  
    
    # 在这里修改你想要的参数
    selected_noise = 'gaussian'  # 可选: 'gaussian' 或 'salt_and_pepper'
    selected_blur = 'severe'            # 可选: 'mild' 或 'severe'

    print(f"🚀 开始批量处理图片...")
    print(f"👉 噪声类型: {selected_noise} | 模糊程度: {selected_blur}")
    
    # 执行处理
    process_images(input_dir, output_dir, noise_type=selected_noise, blur_type=selected_blur)
    
    print("🎉 所有图片处理完成！")