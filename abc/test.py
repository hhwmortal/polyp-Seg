import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import calculate_metrics
from model import Model
from train import DATASET
import torchvision.transforms as transforms
from PIL import Image
from thop import profile


def load_test_data(path):
    test_names_path = f"{path}/test.txt"
    with open(test_names_path, "r") as f:
        data = f.read().splitlines()

    test_images = []
    test_masks = []
    for name in data:
        image_path = os.path.join(path, "images", name)
        mask_path = os.path.join(path, "masks", name)

        if os.path.exists(image_path):
            test_images.append(image_path)
        else:
            print(f"Warning: Image {image_path} does not exist.")

        if os.path.exists(mask_path):
            test_masks.append(mask_path)
        else:
            print(f"Warning: Mask {mask_path} does not exist.")

    return test_images, test_masks


def save_predictions(predictions, output_dir, image_names):
    os.makedirs(output_dir, exist_ok=True)
    for pred, name in zip(predictions, image_names):
        # 将预测结果转换为numpy数组，并应用sigmoid激活函数
        pred_image = pred.cpu().sigmoid().numpy()  # 假设是二分类，使用sigmoid激活

        # 进行二值化处理
        pred_image = (pred_image > 0.5).astype(np.uint8) * 255  # 将预测结果二值化并转换为uint8

        # 转换为PIL图像并保存
        pred_image = Image.fromarray(pred_image[0])  # 选择通道并转换为图像
        pred_image.save(os.path.join(output_dir, f"{name}"))


def test(model, loader, device, output_dir):
    model.eval()
    epoch_metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "acc": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "f2": 0.0,
        "mae": 0.0,
    }
    predictions = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing", ncols=70):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)

            # 保存预测结果
            predictions.extend(y_pred)

            # 计算每个 batch 的指标
            batch_metrics = {key: [] for key in epoch_metrics.keys()}

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_metrics["dice"].append(score[0])
                batch_metrics["iou"].append(score[1])
                batch_metrics["acc"].append(score[2])
                batch_metrics["recall"].append(score[3])
                batch_metrics["precision"].append(score[4])
                batch_metrics["f2"].append(score[5])
                batch_metrics["mae"].append(score[6])

            for key in epoch_metrics.keys():
                epoch_metrics[key] += np.mean(batch_metrics[key])

    # 计算每个 epoch 的平均值
    for key in epoch_metrics.keys():
        epoch_metrics[key] /= len(loader)

    # 保存所有预测的图像
    save_predictions(predictions, output_dir, loader.dataset.image_names)

    return list(epoch_metrics.values())


if __name__ == "__main__":
    """ Hyperparameters """
    path = "./Kvasir-SEG"
    output_dir = "./predictions"
    image_size = 512
    size = (image_size, image_size)
    batch_size = 8
    checkpoint_path = "files/checkpoint.pth"

    """ Load test data """
    test_x, test_y = load_test_data(path)
    print(f"Total test images: {len(test_x)}")

    """ Dataset and loader """
    test_dataset = DATASET(test_x, test_y, size, transform=None)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    model = model.to(device)

    # Load the model weights
    model.load_state_dict(torch.load(checkpoint_path))

    """ Testing the model """
    test_metrics = test(model, test_loader, device, output_dir)

    # 打印测试结果
    print("Test Metrics:")
    print(
        f"Dice: {test_metrics[0]:.4f} - IoU: {test_metrics[1]:.4f} - Acc: {test_metrics[2]:.4f} - Recall: {test_metrics[3]:.4f} - Precision: {test_metrics[4]:.4f} - F2: {test_metrics[5]:.4f} - MAE: {test_metrics[6]:.4f}")
    print(f"预测结果已保存在文件夹: {output_dir}")


# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from model import Model
# from train import DATASET
# from PIL import Image
# import cv2
#
#
# def load_images(image_dir):
#     """ 加载测试图片 """
#     image_names = os.listdir(image_dir)
#     image_paths = [os.path.join(image_dir, name) for name in image_names if name.endswith(('.png', '.jpg', '.jpeg'))]
#     return image_paths, image_names
#
#
# def save_predictions(predictions, output_dir, image_names, target_size=(1250, 1080)):
#     """ 保存预测结果到指定文件夹 """
#     os.makedirs(output_dir, exist_ok=True)
#     for pred, name in zip(predictions, image_names):
#         # 将预测结果转换为numpy数组，并应用sigmoid激活函数
#         pred_image = pred.cpu().sigmoid().numpy()
#
#         # 进行二值化处理
#         pred_image = (pred_image > 0.5).astype(np.uint8) * 255  # 二值化并转换为uint8
#
#         # 调整尺寸到目标大小
#         pred_image = cv2.resize(pred_image[0], target_size, interpolation=cv2.INTER_NEAREST)
#
#         # 转换为PIL图像并保存
#         pred_image = Image.fromarray(pred_image)  # 单通道预测结果
#         pred_image.save(os.path.join(output_dir, name))
#
#
# def predict(model, loader, device, output_dir, target_size=(1250, 1080)):
#     """ 模型预测函数 """
#     model.eval()
#     predictions = []
#
#     with torch.no_grad():
#         for x in tqdm(loader, desc="Predicting", ncols=70):
#             x = x.to(device, dtype=torch.float32)
#             y_pred = model(x)
#
#             # 保存预测结果
#             predictions.extend(y_pred)
#
#     # 保存所有预测图像
#     save_predictions(predictions, output_dir, loader.dataset.image_names, target_size)
#
#
# if __name__ == "__main__":
#     """ Hyperparameters """
#     image_path = "./Kvasir-SEG/images"  # 输入图像文件夹路径
#     output_dir = "./predictions"  # 保存预测结果的文件夹
#     image_size = 512
#     size = (image_size, image_size)
#     batch_size = 16
#     checkpoint_path = "files/checkpoint.pth"
#
#     """ Load images """
#     test_x, image_names = load_images(image_path)
#     print(f"Total test images: {len(test_x)}")
#
#     """ Dataset and loader """
#     test_dataset = DATASET(test_x, None, size, transform=None)  # 不需要掩膜
#     test_dataset.image_names = image_names  # 附加文件名列表到数据集
#
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=2
#     )
#
#     """ Model """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = Model()
#     model = model.to(device)
#
#     # Load the model weights
#     model.load_state_dict(torch.load(checkpoint_path))
#
#     """ Predicting """
#     predict(model, test_loader, device, output_dir, target_size=(1250, 1080))
#
#     print(f"预测结果已保存在文件夹: {output_dir}")
