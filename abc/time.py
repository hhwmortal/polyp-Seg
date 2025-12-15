import os
import time
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
        pred_image = pred.cpu().sigmoid().numpy()
        pred_image = (pred_image > 0.5).astype(np.uint8) * 255
        pred_image = Image.fromarray(pred_image[0])
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

            predictions.extend(y_pred)

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

    for key in epoch_metrics.keys():
        epoch_metrics[key] /= len(loader)

    save_predictions(predictions, output_dir, loader.dataset.image_names)

    return list(epoch_metrics.values())


if __name__ == "__main__":
    """ Hyperparameters """
    path = "./Kvasir-SEG"
    output_dir = "./predictions"
    image_size = 512
    size = (image_size, image_size)
    batch_size = 16
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    """ Model Info: FLOPs, Params """
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"[Model Info] FLOPs: {flops / 1e9:.2f} G - Params: {params / 1e6:.2f} M")

    """ Speed Test: FPS """
    model.eval()
    with torch.no_grad():
        repetitions = 100
        start_time = time.time()
        for _ in range(repetitions):
            _ = model(dummy_input)
        end_time = time.time()
        fps = repetitions / (end_time - start_time)
    print(f"[Speed Test] FPS: {fps:.2f} frames/sec")

    """ Testing the model """
    test_metrics = test(model, test_loader, device, output_dir)

    print("Test Metrics:")
    print(
        f"Dice: {test_metrics[0]:.4f} - IoU: {test_metrics[1]:.4f} - Acc: {test_metrics[2]:.4f} "
        f"- Recall: {test_metrics[3]:.4f} - Precision: {test_metrics[4]:.4f} "
        f"- F2: {test_metrics[5]:.4f} - MAE: {test_metrics[6]:.4f}"
    )
    print(f"预测结果已保存在文件夹: {output_dir}")
