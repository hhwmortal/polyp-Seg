import os
import random
import numpy as np
import torch
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, iou_score, accuracy_score, mean_absolute_error

""" Seeding the randomness. """


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Shuffle the dataset. """


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


def calculate_metrics(y_true, y_pred):
    # 将张量转换为numpy数组
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # 将预测值和真实值转换为二进制（0或1）
    y_pred = (y_pred > 0.5).astype(np.uint8).reshape(-1)
    y_true = (y_true > 0.5).astype(np.uint8).reshape(-1)

    # 计算指标
    score_dice = dice_score(y_true, y_pred)  # 计算Dice系数
    score_iou = iou_score(y_true, y_pred)  # 计算IoU（交并比）
    score_acc = accuracy_score(y_true, y_pred)  # 计算准确率
    score_recall = recall(y_true, y_pred)  # 计算召回率
    score_precision = precision(y_true, y_pred)  # 计算精准率
    score_f2 = F2(y_true, y_pred)  # 计算F2分数

    # 将二进制数组转换为PyTorch张量以计算MAE
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

    score_mae = mean_absolute_error(y_true_tensor, y_pred_tensor)  # 计算平均绝对误差（MAE）

    return [score_dice, score_iou, score_acc, score_recall, score_precision, score_f2, score_mae]
