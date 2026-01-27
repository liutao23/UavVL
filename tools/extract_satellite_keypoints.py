import os
import pickle
import cv2
import torch
from fine_match_models.fine_matching_model import Matching, frame2tensor
from tqdm import tqdm


# 配置文件路径
superpoint_model_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/superpoint_v1.pth'
superglue_model_path = '/home/liutao/video/demo/Satellite/Visual_Localization/weights/superglue_outdoor.pth'

fine_match_config = {
    'superpoint': {
        'superpoint_weight': superpoint_model_path,
        'nms_radius': 4,  # 非最大抑制半径，用于去除重复的关键点
        'keypoint_threshold': 0.05,  # 关键点置信度阈值，低于此值的关键点将被丢弃
        'max_keypoints': -1  # 最大关键点数，-1 表示保留所有关键点
    },
    'superglue': {
        'superglue_weight': superglue_model_path,
        'sinkhorn_iterations': 20,  # Sinkhorn 算法的迭代次数，用于优化匹配
        'match_threshold': 0.5,  # 匹配的置信度阈值，低于此值的匹配将被丢弃
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化fine-matching模型
fine_matching = Matching(fine_match_config).eval().to(device)

# 卫星图像路径
# satellite_img_path = '/home/linwenhao/Desktop/Visual_Localization/satellite_img'
# satellite_dirs = sorted(os.listdir(satellite_img_path))

satellite_img_path = '/media/liutao/A/UAV_VisLoc_dataset/sat_db_tiles20/01/satellite01'
satellite_dirs = sorted(os.listdir(satellite_img_path))

# 存储提取的特征点数据

# 处理每个卫星图像
for file in tqdm(satellite_dirs):  # 排除最后一个文件，因为代码中是通过索引去除
    file_path = os.path.join(satellite_img_path, file)
    save_path = os.path.join('/media/liutao/A/UAV_VisLoc_dataset/sat_db_tiles20/01/Sample4geo/satellite_keypoints',f'{file[:-4]}.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 读取图像
    image = cv2.imread(file_path)
    if image is None:
        print(f"无法读取图像: {file_path}")
        continue

    # 将图像转换为灰度图像
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调整图像大小
    resize = (800, 800)  # 根据需求调整
    image_resized = cv2.resize(image_gray, resize)

    # 将图像转换为Tensor
    image_tensor = frame2tensor(image_resized, device)

    # 使用fine-matching模型提取关键点特征
    last_data = fine_matching.superpoint({'image': image_tensor})

    # 将关键点数据重命名并保存
    keys = ['keypoints', 'scores', 'descriptors']
    last_data = {k + '1': last_data[k] for k in keys}  # 给每个数据添加一个 0 后缀
    last_data['image1'] = image_tensor
    print()
    # 将提取的特征保存为.pkl文件
    with open(save_path, 'wb') as f:
        pickle.dump(last_data, f)

    # satellite_pkl.append(last_data)

