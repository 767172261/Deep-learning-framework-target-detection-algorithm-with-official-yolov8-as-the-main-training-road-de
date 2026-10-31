基于深度学习的道路缺陷检测系统yolov8
@[toc]

官方yolov8为主干，实现道路缺陷的检测识别，用PyQt5设计了简约的系统UI界面。在界面中，您可以选择自己的视频文件、图片文件进行检测。


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/605314878c6849979f2b8d25f903faf6.png)
要实现功能目标：

 YOLOv8 和 PyQt5 的道路缺陷检测系统的详细实现。系统支持单张图片、视频文件、摄像头实时采集的识别，并提供结果导出功能。

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d91749ca50cf48118f7d00996469b01b.png)


### **1. 安装依赖**

确保安装了必要的库：

```bash
pip install ultralytics opencv-python pyqt5 pandas openpyxl
```

---

### **2. 系统设计与实现**

#### **2.1 数据准备**
- 假设你已经训练了一个 YOLOv8 模型来检测道路缺陷（如裂缝、坑洼等）。
- 将模型权重保存为 `best.pt`，并在代码中加载。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e9abf80c381436a900343296624cd0f.png)

#### **2.2 功能模块**
系统的主要功能包括：
1. 单张图片识别。
2. 遍历文件夹中的图片进行批量识别。
3. 支持视频文件处理。
4. 支持摄像头实时采集和识别。
5. 结果导出为 Excel 或 CSV 文件。

---

### **3. 代码实现**

#### **3.1 主程序逻辑**

```python
from ultralytics import YOLO
import cv2
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class RoadDefectDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("道路缺陷检测系统")
        self.setGeometry(100, 100, 800, 600)

        # 初始化模型
        self.model = YOLO("best.pt")  # 加载训练好的模型

        # UI 元素
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 700, 400)

        self.btn_image = QPushButton("选择图片", self)
        self.btn_folder = QPushButton("遍历文件夹", self)
        self.btn_video = QPushButton("选择视频", self)
        self.btn_camera = QPushButton("打开摄像头", self)
        self.btn_export = QPushButton("导出结果", self)

        self.btn_image.setGeometry(50, 500, 150, 40)
        self.btn_folder.setGeometry(220, 500, 150, 40)
        self.btn_video.setGeometry(390, 500, 150, 40)
        self.btn_camera.setGeometry(560, 500, 150, 40)
        self.btn_export.setGeometry(320, 550, 150, 40)

        # 绑定按钮事件
        self.btn_image.clicked.connect(self.detect_single_image)
        self.btn_folder.clicked.connect(self.detect_folder_images)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_export.clicked.connect(self.export_results)

        # 视频捕获相关
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 存储检测结果
        self.results_data = []

    def detect_single_image(self):
        """检测单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if file_path:
            self.process_image(file_path)

    def detect_folder_images(self):
        """遍历文件夹中的图片进行检测"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(folder_path, file_name)
                    self.process_image(file_path)

    def detect_video(self):
        """检测视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)

    def open_camera(self):
        """打开摄像头进行实时检测"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        """更新视频帧或摄像头捕获的画面"""
        ret, frame = self.cap.read()
        if ret:
            self.process_image(frame=frame, is_video=True)

    def process_image(self, file_path=None, frame=None, is_video=False):
        """处理图片并显示结果"""
        if not is_video:
            frame = cv2.imread(file_path)

        # 使用 YOLOv8 进行检测
        results = self.model(frame)

        # 绘制边界框和标签
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 保存检测结果
                if not is_video:
                    self.results_data.append({
                        "file": file_path,
                        "class": self.model.names[int(cls)],
                        "confidence": conf,
                        "bbox": box.tolist()
                    })

        # 显示结果
        if is_video:
            self.display_frame(frame)
        else:
            self.display_image(frame)

    def display_image(self, frame):
        """显示图片"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def display_frame(self, frame):
        """显示视频帧"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

    def export_results(self):
        """导出检测结果为 Excel 或 CSV 文件"""
        if not self.results_data:
            QMessageBox.warning(self, "警告", "没有检测结果可导出！")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file_path:
            df = pd.DataFrame(self.results_data)
            if file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            elif file_path.endswith(".csv"):
                df.to_csv(file_path, index=False)
            QMessageBox.information(self, "成功", "结果已导出！")

if __name__ == "__main__":
    app = QApplication([])
    window = RoadDefectDetector()
    window.show()
    app.exec_()
```

---



---



