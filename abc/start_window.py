import sys
import os
import shutil
import torch
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from model import Model
from utils import calculate_metrics
from PIL import Image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 窗口基本设置
        self.setWindowTitle("结直肠息肉分割系统")
        self.setGeometry(200, 200, 1000, 600)

        # 模型与路径
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img2predict = None

        # 确保目录存在
        os.makedirs("images/tmp", exist_ok=True)
        os.makedirs("predictions", exist_ok=True)

        self.initUI()

    def initUI(self):
        # 主窗口容器
        widget = QWidget()
        self.setCentralWidget(widget)

        # 左边按钮
        self.upload_btn = QPushButton("上传图片")
        self.detect_btn = QPushButton("开始检测")
        self.load_model_btn = QPushButton("加载模型")

        self.upload_btn.clicked.connect(self.upload_img)
        self.detect_btn.clicked.connect(self.detect_img)
        self.load_model_btn.clicked.connect(self.load_model)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.upload_btn)
        left_layout.addWidget(self.detect_btn)
        left_layout.addWidget(self.load_model_btn)
        left_layout.addStretch()

        # 中间显示：原图 / 结果
        self.left_img = QLabel("原始图像")
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img = QLabel("分割结果")
        self.right_img.setAlignment(Qt.AlignCenter)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.left_img)
        img_layout.addWidget(self.right_img)

        # 总体布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(img_layout, 4)

        widget.setLayout(main_layout)

    def load_model(self):
        checkpoint_path = "files/checkpoint.pth"
        if not os.path.exists(checkpoint_path):
            QMessageBox.critical(self, "错误", f"未找到模型权重: {checkpoint_path}")
            return

        self.model = Model().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        QMessageBox.information(self, "成功", f"✅ 模型加载成功: {checkpoint_path}")

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if fileName:
            save_dir = "images/tmp"
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
            save_path = os.path.join(save_dir, "tmp_upload.png")
            shutil.copy(fileName, save_path)

            self.img2predict = save_path
            self.left_img.setPixmap(QPixmap(save_path).scaled(
                self.left_img.width(), self.left_img.height(), Qt.KeepAspectRatio
            ))
            self.right_img.clear()

    def preprocess(self, image_path, size=(512, 512)):
        """读取并预处理图像"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, size)
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        return tensor.to(self.device), img  # 返回原图，方便可视化

    def postprocess(self, pred, orig_img):
        """后处理：输出mask并与原图叠加"""
        pred = torch.sigmoid(pred).cpu().detach().numpy()[0, 0]  # 取第一个batch和通道
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))

        # 转换成三通道
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_img, 0.6, mask_color, 0.4, 0)

        return overlay, mask

    def detect_img(self):
        if self.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型！")
            return
        if self.img2predict is None:
            QMessageBox.warning(self, "提示", "请先上传图片！")
            return

        # 预处理
        tensor, orig_img = self.preprocess(self.img2predict)

        with torch.no_grad():
            pred = self.model(tensor)

        overlay, mask = self.postprocess(pred, orig_img)

        # 保存结果
        save_path = "predictions/result.png"
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 显示到右侧
        self.right_img.setPixmap(QPixmap(save_path).scaled(
            self.right_img.width(), self.right_img.height(), Qt.KeepAspectRatio
        ))

        QMessageBox.information(self, "完成", f"✅ 分割完成，结果已保存: {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
