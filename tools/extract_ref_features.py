import torch
from torchvision import transforms
import os
import tkinter as tk
from PIL import Image, ImageTk
import pickle
from tqdm import tqdm  # 导入 tqdm
from course_match_models.FSRA.make_model import two_view_net
from course_match_models.samplegeo.models import TimmModel


# 显示图片
def show_image(image_path):
    root = tk.Tk()
    root.title("Image Viewer")

    # 打开图像文件
    image = Image.open(image_path)
    # 将PIL图像对象转换为Tkinter图像对象
    tk_image = ImageTk.PhotoImage(image)

    # 创建Label小部件并显示图像
    label = tk.Label(root, image=tk_image)
    label.image = tk_image  # 保持对图像的引用
    label.pack()

    root.mainloop()

def extract_and_save_features(satellite_dir_path, model, transform, output_path, device='cuda:0'):
    """
    提取卫星图像特征并保存到文件
    :param satellite_dir_path: 卫星数据库文件夹路径
    :param model: 提取特征的模型
    :param output_path: 保存特征的路径
    :param device: 设备
    """
    features = {}

    # 使用 tqdm 显示进度条
    files = [file for file in os.listdir(satellite_dir_path) if
             file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
    with tqdm(total=len(files), desc="Extracting Features", unit="file") as pbar:
        for file in files:
            satellite_img_path = os.path.join(satellite_dir_path, file)
            # 读取参考图像
            satellite_img = Image.open(satellite_img_path)

            # 如果参考图是RGBA则转化为RGB格式
            if satellite_img.mode == "RGBA":
                satellite_img = satellite_img.convert("RGB")
            satellite_img = transform(satellite_img).unsqueeze(0).to(device)

            with torch.no_grad():
                _, a = model(None, satellite_img)
                features[file] = a.cpu().numpy()

            # 更新进度条
            pbar.update(1)

    # 保存特征
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"卫星图像特征已保存到 {output_path}")


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 指定图像检索的方法 [FSRA Sample4geo]
retrival_methd = 'Sample4geo'
# 定义粗匹配模型
if retrival_methd == 'FSRA':
    course_model_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/net_119.pth'
    vit_pretrain_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/vit_small_p16_224-15ec54c9.pth'
    coarse_model = two_view_net(class_num=6400, block=1, pretrain_path=vit_pretrain_path)
    coarse_model.load_state_dict(torch.load(course_model_path), strict=False)
    coarse_model = coarse_model.eval()
    coarse_model = coarse_model.to(device)

    # 粗匹配图像预处理
    Transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

elif retrival_methd == 'Sample4geo':
    course_model_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/weights_end.pth'
    coarse_model = TimmModel()
    coarse_model.load_state_dict(torch.load(course_model_path), strict=False)
    coarse_model.to(device).eval()

    # 粗匹配图像预处理
    Transform = transforms.Compose([
    transforms.Resize((384, 384)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

# 设置路径
# satellite_dir_path = "/home/liutao/音乐/demo/Satellite/Visual_Localization/Reference/satellite_img"
# output_path = "/home/liutao/音乐/demo/Satellite/Visual_Localization/Reference/" +retrival_methd + "/Reference_img_feature.pkl"

# satellite_dir_path = "/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/03/satellite_tif_flat"
# output_path = os.path.dirname(satellite_dir_path) +'/'+ retrival_methd + "/Reference_img_feature.pkl"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# device = "cuda:0"
#
# # 提取并保存特征
# extract_and_save_features(satellite_dir_path, coarse_model, Transform, output_path, device)

for scene in ['04', '05', '05', '08', '09', '10', '11']:

    satellite_dir_path = f"/media/liutao/B/论文投稿/两阶段视觉定位/flight/sat_paired/{scene}/satellite_tif_flat"
    output_path = os.path.dirname(satellite_dir_path) +'/'+ retrival_methd + "/Reference_img_feature.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = "cuda:0"

    # 提取并保存特征
    extract_and_save_features(satellite_dir_path, coarse_model, Transform, output_path, device)