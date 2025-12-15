import os
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import Model
from metrics import DiceBCELoss
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def load_names(path, file_path):
    with open(file_path, "r") as f:
        data = f.read().splitlines()
    images = [os.path.join(path, "images", name) for name in data]
    masks = [os.path.join(path, "masks", name) for name in data]
    return images, masks


def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"
    test_names_path = f"{path}/test.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)
    test_x, test_y = load_names(path, test_names_path)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


'''
 有真实标签预测
'''
class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.size = size
        self.n_samples = len(images_path)

        # 保存图像文件名
        self.image_names = [os.path.basename(path) for path in images_path]

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples


'''
 无真实标签预测
'''


# class DATASET(Dataset):
#     def __init__(self, images_path, masks_path=None, size=(512, 512), transform=None):
#         super().__init__()
#
#         self.images_path = images_path
#         self.masks_path = masks_path
#         self.transform = transform
#         self.size = size
#         self.n_samples = len(images_path)
#
#         # 保存图像文件名
#         self.image_names = [os.path.basename(path) for path in images_path]
#
#     def __getitem__(self, index):
#         """ Image """
#         image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
#
#         if self.masks_path:
#             mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
#         else:
#             mask = None  # 如果没有提供掩膜路径，则掩膜为 None
#
#         if self.transform is not None and mask is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]
#         elif self.transform is not None:
#             augmentations = self.transform(image=image)
#             image = augmentations["image"]
#
#         image = cv2.resize(image, self.size)
#         image = np.transpose(image, (2, 0, 1))
#         image = image / 255.0
#
#         if mask is not None:
#             mask = cv2.resize(mask, self.size)
#             mask = np.expand_dims(mask, axis=0)
#             mask = mask / 255.0
#             return image, mask
#         else:
#             return image
#
#     def __len__(self):
#         return self.n_samples


def train(model, loader, optimizer, loss_fn, device, train_log_path):
    model.train()
    epoch_loss = 0.0

    epoch_dice = 0.0
    epoch_iou = 0.0
    epoch_acc = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_f2 = 0.0
    epoch_mae = 0.0

    # 使用 tqdm 包装 loader
    for i, (x, y) in enumerate(tqdm(loader, desc="Training", ncols=70)):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 计算每个 batch 的指标
        batch_dice = []
        batch_iou = []
        batch_acc = []
        batch_recall = []
        batch_precision = []
        batch_f2 = []
        batch_mae = []

        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_dice.append(score[0])  # dice coefficient
            batch_iou.append(score[1])  # IoU
            batch_acc.append(score[2])  # accuracy
            batch_recall.append(score[3])  # recall
            batch_precision.append(score[4])  # precision
            batch_f2.append(score[5])  # F2 score
            batch_mae.append(score[6])  # MAE (mean absolute error)

        epoch_dice += np.mean(batch_dice)
        epoch_iou += np.mean(batch_iou)
        epoch_acc += np.mean(batch_acc)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)
        epoch_f2 += np.mean(batch_f2)
        epoch_mae += np.mean(batch_mae)

    # 在循环外面计算每个 epoch 的平均值
    epoch_dice /= len(loader)
    epoch_iou /= len(loader)
    epoch_acc /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)
    epoch_f2 /= len(loader)
    epoch_mae /= len(loader)

    return epoch_loss, [epoch_dice, epoch_iou, epoch_acc, epoch_recall, epoch_precision, epoch_f2, epoch_mae]


def evaluate(model, loader, loss_fn, device, train_log_path):
    model.eval()
    epoch_loss = 0.0

    epoch_dice = 0.0
    epoch_iou = 0.0
    epoch_acc = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    epoch_f2 = 0.0
    epoch_mae = 0.0

    with torch.no_grad():
        # 使用 tqdm 包装 loader
        for i, (x, y) in enumerate(tqdm(loader, desc="Evaluating", ncols=70)):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # 计算每个 batch 的指标
            batch_dice = []
            batch_iou = []
            batch_acc = []
            batch_recall = []
            batch_precision = []
            batch_f2 = []
            batch_mae = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_dice.append(score[0])  # dice coefficient
                batch_iou.append(score[1])  # IoU
                batch_acc.append(score[2])  # accuracy
                batch_recall.append(score[3])  # recall
                batch_precision.append(score[4])  # precision
                batch_f2.append(score[5])  # F2 score
                batch_mae.append(score[6])  # MAE (mean absolute error)

            epoch_dice += np.mean(batch_dice)
            epoch_iou += np.mean(batch_iou)
            epoch_acc += np.mean(batch_acc)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            epoch_f2 += np.mean(batch_f2)
            epoch_mae += np.mean(batch_mae)

    # 计算每个 epoch 的平均值
    epoch_dice /= len(loader)
    epoch_iou /= len(loader)
    epoch_acc /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)
    epoch_f2 /= len(loader)
    epoch_mae /= len(loader)

    return epoch_loss, [epoch_dice, epoch_iou, epoch_acc, epoch_recall, epoch_precision, epoch_f2, epoch_mae]


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 512
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 300
    lr = 1e-4
    early_stopping_patience = 30  # 早停机制，连续30个epoch中验证性能没有改善，则训练将停止，以防止模型过拟合
    checkpoint_path = "files/checkpoint.pth"
    path = "./Kvasir-SEG"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = Model()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # 添加 train_log_path 参数
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device, train_log_path)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device, train_log_path)

        scheduler.step(valid_loss)

        # 计算 epoch 时间
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 记录每个 epoch 的信息
        data_str = f"Epoch: {epoch + 1}/{num_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Dice: {train_metrics[0]:.4f} - IoU: {train_metrics[1]:.4f} - Acc: {train_metrics[2]:.4f} - Recall: {train_metrics[3]:.4f} - Precision: {train_metrics[4]:.4f} - F2: {train_metrics[5]:.4f} - MAE: {train_metrics[6]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Dice: {valid_metrics[0]:.4f} - IoU: {valid_metrics[1]:.4f} - Acc: {valid_metrics[2]:.4f} - Recall: {valid_metrics[3]:.4f} - Precision: {valid_metrics[4]:.4f} - F2: {valid_metrics[5]:.4f} - MAE: {valid_metrics[6]:.4f}\n"
        print_and_save(train_log_path, data_str)

        # 检查是否需要保存模型
        if valid_metrics[0] > best_valid_metrics:  # Dice (valid_metrics[0] 现在是 Dice)
            data_str = f"Valid Dice improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        # 早停机制
        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continuously.\n"
            print_and_save(train_log_path, data_str)
            break
