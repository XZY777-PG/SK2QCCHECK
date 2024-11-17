import sys
import os
# sys.path.append(r'C:\Users\xu.z.29\OneDrive - Procter and Gamble\Desktop\SK2 QC Simplification\numpy')
import numpy as np
import cv2
import PyPDF2
import shutil
import fitz
import difflib  # 用于计算相似度
import ctypes
import pdf2image
from pdf2image import convert_from_path
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QDialog, QPushButton, QHBoxLayout, QVBoxLayout,QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import QTimer, QDir, Qt, QThread, pyqtSignal, QEventLoop  # 导入 QTimer
from paddleocr import PaddleOCR
import torch
import torchvision.transforms as transforms
from PIL import Image
# import cv2
from ultralytics import YOLOv10
# import fitz
# import tempfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results
def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results
def remove_substrings(original_string, substrings):
    modified_string = original_string
    for substring in substrings:
        modified_string = modified_string.replace(substring, "")
    return modified_string
# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(3 * 512 * 512, 2)  # 输入大小为512x512的RGB图像

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平
        x = self.fc(x)
        return x

def draw_bounding_boxes(frame, target_locations):
    # 创建一个副本以避免修改原始图像
    annotated_frame = frame.copy()
    # upp = 0
    # downn = 0
    rk = []
    # 遍历目标位置信息列表
    a = 0
    for target in target_locations:
        # 提取目标的四个坐标点
        points = np.array(target, np.int32)
        points = points.reshape((-1, 1, 2))
        
        # 绘制不规则四边形的轮廓线
        cv2.polylines(annotated_frame, [points], isClosed=True, color=(0, 0, 255), thickness=2)

        # 计算四个顶点的平均值作为横坐标
        x_mean = np.mean(points[:, 0, 0])
        # if downn == 0 or x_mean < downn:
        #     downn = x_mean
        # elif upp ==0 or x_mean > upp:
        #     upp = x_mean
        rk.append(x_mean)
        # 在不规则四边形中间绘制垂直线
        cv2.line(annotated_frame, (int(x_mean), 0), (int(x_mean), annotated_frame.shape[0]), (0, 0, 255), thickness=2)
        if rk:
            # print(max(rk) - min(rk))
            a = max(rk) - min(rk)    
    return annotated_frame, a

class ImageWatcherThread(QThread):
    image_added = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        


    def run(self):
        while True:
            self.latest_files = []
            files = os.listdir(self.folder_path)
            new_files = [f for f in files if f not in self.latest_files]

            for new_file in new_files:
                self.latest_files.append(new_file)
                self.image_added.emit(new_file)

            self.sleep(1)  # 每1毫秒检测一次

class ImageLoaderApp(QWidget):
    image_added = pyqtSignal(str)
    # @staticmethod
    def __init__(self):
        super().__init__()
        self.log_file = "operation_log.txt"
        self.image_folder = r'C:\develop\yolov10-main\yolov10-main\testimages'  # 指定图片文件夹路径
        # self.watch_folder = r'C:\MijCtrl\Hot'  # 指定图片文件夹路径
        self.watch_folder = r"C:\MijCtrl\Hot\P04-机"
        self.latest_files = []
        # self.current_index = 0
        self.image_cache = []
        self.current_index = 0  # 当前显示的图片索引
        self.current_index1 = 0
        self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 初始化 PaddleOCR
        self.yolo = YOLOv10(model = r"C:\develop\yolov10-main\yolov10-main\best.pt")
        self.yolo1 = YOLOv10(model = r"C:\develop\yolov10-main\yolov10-main\best 10.pt")
        # self.net = Net()
        # self.net.load_state_dict(torch.load(r"C:\develop\yolov10-main\yolov10-main\model\model_weights.pth"))
        # self.net.eval()
        # 标记变量：用来追踪是否已经显示核验通过或人工确认
        # 初始化日志文件
        # self.log_file = 'ocr_log.txt'
        # self.verification_success = False
        # 初始化计时器
        self.verification_timer = QTimer(self)
        self.verification_timer.setInterval(15000)  # 设置计时器为10秒
        self.verification_timer.timeout.connect(self.show_manual_confirmation)
        self.verification_success = False

        self.initUI()
        self.image_watcher_thread = ImageWatcherThread(self.watch_folder)
        self.image_watcher_thread.image_added.connect(self.on_new_image_added)
        self.image_watcher_thread.start()

        self.pdf_files = self.get_pdf_files(self.image_folder)  # 获取所有图片文件名
        self.image_files = self.get_image_files(self.image_folder)  # 获取所有图片文件名
        # self.image_folder = r"C:\Users\x\Desktop\PGPROJ\PaddleOCR-main\testimages\download"
        # if self.pdf_files:
        #     self.transform_pdf(self.pdf_files[self.current_index])

        if self.image_files:
            self.show_image(self.image_files[self.current_index])

        # 使用 QTimer 定时更新摄像头画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(10)  # 每秒更新一次画面
        
    def initUI(self):
        # 设置窗口标题和尺寸
        self.setWindowTitle('LGDC SKII DTC QC Auto Check')
        self.setGeometry(800, 300, 1200, 800)  # 调整窗口尺寸以适应更多内容

        # 创建一个标签用于显示摄像头画面
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(400, 300)  # 设置摄像头画面的固定大小

        self.log_area = QtWidgets.QPlainTextEdit(self)
        self.log_area.setReadOnly(True)

        # 创建一个标签用于显示摄像头 OCR 结果
        self.camera_ocr_label = QLabel(self)
        self.camera_ocr_label.setFont(QFont('Arial', 12))  # 设置字体大小
        self.camera_ocr_label.setFixedSize(400, 200)  # 设置标签的固定大小

        # # 创建一个文本标签用于显示图片类型结果
        # self.camera_cls_label = QLabel(self)
        # self.camera_cls_label.setFont(QFont('Arial', 12))  # 设置字体大小
        # self.camera_cls_label.setFixedSize(400, 30)  # 设置标签的固定大小

        # 创建一个标签用于显示图片
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(400, 300)  # 设置标签的固定大小

        # 创建一个标签用于显示图片名称
        self.image_name_label = QLabel(self)
        self.image_name_label.setFont(QFont('Arial', 12))  # 设置字体大小
        self.image_name_label.setFixedSize(400, 30)  # 设置标签的固定大小

        # 创建一个文本标签用于显示图片 OCR 结果
        self.ocr_result_label = QLabel(self)
        self.ocr_result_label.setFont(QFont('Arial', 12))  # 设置字体大小
        self.ocr_result_label.setFixedSize(400, 200)  # 设置标签的固定大小

        # 创建一个文本标签用于显示图片类型结果
        self.cls_result_label = QLabel(self)
        self.cls_result_label.setFont(QFont('Arial', 12))  # 设置字体大小
        self.cls_result_label.setFixedSize(400, 30)  # 设置标签的固定大小

        # 创建一个按钮，用于切换图片
        self.next_button = QPushButton('切换图片', self)
        self.next_button.setFont(QFont('Arial', 16))  # 设置按钮的字体大小
        self.next_button.setFixedSize(150, 50)  # 设置按钮的固定大小（更宽和更窄）
        self.next_button.clicked.connect(self.switch_image)

        # 创建一个垂直布局用于放置图片名称和图片 OCR 结果
        name_and_ocr_layout = QVBoxLayout()
        name_and_ocr_layout.addWidget(self.image_name_label)
        name_and_ocr_layout.addWidget(self.ocr_result_label)
        name_and_ocr_layout.addWidget(self.cls_result_label)

        # 创建一个水平布局用于放置图片和上面的垂直布局
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addLayout(name_and_ocr_layout)

        # 创建一个水平布局用于放置摄像头画面和 OCR 结果
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_ocr_label)

        # 创建一个垂直布局，并将控件添加到布局中
        main_layout = QVBoxLayout()
        main_layout.addLayout(camera_layout)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.log_area)
        main_layout.addWidget(self.next_button)

        # 设置窗口的布局
        self.setLayout(main_layout)

    def on_new_image_added(self, image_file):
        src_path = os.path.join(self.watch_folder, image_file)
        dst_path = os.path.join(self.image_folder, image_file)
        if src_path.lower().endswith('.pdf'):
            # 打开 PDF 文件
            pdf_document = fitz.open(src_path)

            # 遍历每一页
            # for page_number in range(len(pdf_document)):
            # 获取页面
            page = pdf_document.load_page(0)
            
            # 将页面转换为图片
            pix = page.get_pixmap()
            dst_path = dst_path[:-4] + '.jpg'
            # 保存为 JPG 文件
            output_path = os.path.join(dst_path)
            # rotated_img = pix.rotate(90, expand=True)  # 逆时针旋转 90 度
            pix.save(output_path)
            with Image.open(output_path) as img:
                # 逆时针旋转 90 度
                rotated_img = img.rotate(90, expand=True)
                # 保存旋转后的图像
                rotated_img.save(output_path)
            print(f'Saved: {output_path}')

            # 关闭 PDF 文档
            pdf_document.close()
            pixmap = QPixmap(dst_path)
            self.image_cache.append(pixmap)
            self.display_image_by_index(len(self.image_cache) - 1)
            self.show_image(image_file[:-4] + '.jpg')        
        else:
            with Image.open(src_path) as img:
                # 逆时针旋转 90 度
                rotated_img = img.rotate(90, expand=True)
                # 保存旋转后的图像
                rotated_img.save(dst_path)
            # shutil.copy(src_path, dst_path)  # 将图片复制到镜像文件夹
            pixmap = QPixmap(dst_path)
            self.image_cache.append(pixmap)
            self.display_image_by_index(len(self.image_cache) - 1)
            self.show_image(image_file)
        # self.verification_success = False

    def log_event(self, result):
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        log_message = f"{timestamp} - {result}\n"
        
        # 在文本框中显示日志
        self.log_area.appendPlainText(log_message)
        
        # 写入文件
        with open(self.log_file, "a") as file:
            file.write(log_message)

    def get_pdf_files(self, folder):
        # 获取指定文件夹中的所有图片文件
        supported_formats = ('.pdf')
        # print([f for f in os.listdir(folder) if f.lower().endswith(supported_formats)])
        return [f for f in os.listdir(folder) if f.lower().endswith(supported_formats)]

    def get_image_files(self, folder):
        # 获取指定文件夹中的所有图片文件
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.pdf')
        # print([f for f in os.listdir(folder) if f.lower().endswith(supported_formats)])
        return [f for f in os.listdir(folder) if f.lower().endswith(supported_formats)]
    
    def display_image_by_index(self, index):
        pixmap = self.image_cache[index]
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        self.current_index = index
    
    def show_image(self, image_name):
        # 加载并显示图片
        image_path = os.path.join(self.image_folder, image_name)
        self.image_name = image_name
        # 使用 OpenCV 读取图片
        image = cv2.imread(image_path)
        
        # 将图像从 BGR 转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result_img, _ = predict_and_detect(self.yolo, image, classes=[], conf=0.7)
        # print(_)
        # print(type(_))
        # print(_[:3])
        if _[0].boxes:
            # print(_[0])
            self.status = 'contain heart pattern'
        else:
            self.status = 'no heart pattern'  

        # 转换为 QImage 并设置到 QLabel
        h, w, ch = result_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.image_label.size(), aspectRatioMode=1))

        # 显示图片文件名，加入固定字段
        self.image_name_label.setText(f"文件名: {image_name}")
        
        # 显示 OCR 结果
        ocr_result, box = self.perform_ocr(image_path)
        self.ocr_result_label.setText(f"OCR 结果:\n{ocr_result}\n图像类别识别结果：\n"+self.status)
        self.current_image_ocr_result = ocr_result
    
        # # 图像预处理
        # transform = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # # 加载并预测新的图像
        # # image = Image.open(r"C:\Users\xu.z.29\OneDrive - Procter and Gamble\Desktop\SK2 QC Simplification\CV\test\P\2.jpg")
        # image = Image.open(image_path)
        # image = transform(image).unsqueeze(0)
        # outputs = self.net(image)
        # _, predicted = torch.max(outputs.data, 1)

        # # 输出预测结果
        # trainset = ImageFolder(root=r"C:\develop\yolov10-main\yolov10-main\CV\train", transform=transform)
        # class_names = trainset.classes  # 根据训练数据集中的类别顺序获取类别名称
        # predicted_class = class_names[predicted.item()]
        # # print(f"Predicted Class: {predicted_class}")
        # self.cls_result_label.setText(f"图像类别结果:\n{predicted_class}")

    def switch_image(self):
        # 切换到下一张图片
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.show_image(self.image_files[self.current_index])

    def perform_ocr(self, image_path):
        # 使用 PaddleOCR 进行实际的 OCR 识别
        result = self.ocr.ocr(image_path, cls=True)
        result = result[0]
        if result:
            txts, box = [line[1][0] for line in result], [line[0] for line in result]
            # print(box)
        else:
            txts, box = [], []

        # print(txts, box)
        return "\n".join(txts), box
    
    def calculate_similarity(self, text1, text2):
        # 使用 difflib 计算两个文本的相似度
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def update_camera(self):
        # 从摄像头读取画面并更新显示
        ret, frame = self.cap.read()
        
        if ret:
            ori_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色从 BGR 到 RGB
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            # 将摄像头画面保存为临时文件
            temp_image_path = "temp_camera_image.jpg"
            cv2.imwrite(temp_image_path, frame)

            # 进行 OCR 识别
            camera_ocr_result, box = self.perform_ocr(temp_image_path)
            result_img, _ = predict_and_detect(self.yolo1, ori_frame, classes=[], conf=0.7)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            # print(_)
            # print(type(_))
            # print(_[:3])
            if _[0].boxes:
                # print(_[0])
                self.status1 = 'contain heart pattern'
            else:
                self.status1 = 'no heart pattern'

            annotated_frame, fix_value = draw_bounding_boxes(result_img, box)
            # substrings_to_remove = ["FACIAL\n", "TREATMENT\n", "ESSENCE\n","PITERA","S\n","SK\n","SK-\n","SK-I","SK-II\n""I\n"]
            # camera_ocr_result = remove_substrings(camera_ocr_result, substrings_to_remove)
            self.camera_ocr_label.setText(f"OCR 结果:\n{camera_ocr_result}\n矫正值\n{fix_value}")
            

            if len(camera_ocr_result)>=1 and not self.verification_timer.isActive():
                print(camera_ocr_result)
                self.verification_success = False  # 重置标记
                self.verification_timer.start()  # 开始计时器，5秒后触发超时处理


            qt_image = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.camera_label.size()))
            # print(x_mean)
            
            substrings_to_remove = ["FACIAL\n", "TREATMENT\n", "ESSENCE\n","S\n","SK\n","SK-\n","SK-I","SK-II\n","I\n","FACIAL", "TREATMENT", "ESSENCE","S","SK","SK-","SK-II","I","K-II","K-II\n","PITER","PITERA","PTERA","PTER"]
            a = camera_ocr_result
            camera_ocr_result = remove_substrings(camera_ocr_result, substrings_to_remove)
            self.camera_ocr_label.setText(f"OCR 结果:\n{camera_ocr_result}\n矫正值\n{fix_value}\n图像类别结果:\n"+self.status1)

            # self.ocr_result_label.setText(f"图像类别结果:\n"+status1)

            # 计算 OCR 结果的相似度
            similarity = self.calculate_similarity(self.current_image_ocr_result, camera_ocr_result)

            # 如果相似度大于等于 50%，则显示核验通过并切换图片
            if similarity >= 0.4 and fix_value < 15 and len(camera_ocr_result)>=1 and self.status==self.status1:

                QTimer.singleShot(3000, self.show_verification_success)
                result = "核验通过" + camera_ocr_result
                
                # 记录日志
                self.log_event(result)

    def show_verification_success(self):
        """显示核验通过的消息框，并在3秒后自动关闭"""
        # 停止计时器并断开槽连接
        self.verification_timer.stop()
        self.verification_timer.blockSignals(True)  # 阻止计时器信号

        # 更新标记，表示核验已经成功
        self.verification_success = True
        # 显示核验通过的消息框
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setText("核验通过")
        self.msg.setWindowTitle("结果核验")
        self.msg.setStandardButtons(QMessageBox.NoButton)  # 移除默认按钮
        self.msg.show()

        # 确保消息框显示
        QApplication.processEvents()  # 强制刷新事件循环
        self.verification_success = False
        # 3秒后自动关闭消息框
        QTimer.singleShot(3000, self.close_verification_success)

    def close_verification_success(self):
        """在关闭核验通过对话框时重新启用计时器"""
        self.msg.accept()  # 关闭消息框
        self.verification_timer.blockSignals(False)  # 重新启用计时器信号

    def show_manual_confirmation(self):
        """显示长时间未核验通过的对话框，并锁定界面"""
        if self.verification_success:
            return
        result = "核验超时" + '\n' + '超时订单号：{}'.format(self.image_name)
        
        # 记录日志
        self.log_event(result)
        # ctypes.windll.user32.LockWorkStation()  # 锁定 Windows 屏幕
        # 创建模态对话框以锁定屏幕
        dialog = QDialog(self)
        dialog.setWindowTitle("核验超时")
        

        dialog.setModal(True)  # 设置为模态对话框
        dialog.setFixedSize(300, 150)

        # 创建对话框内容
        layout = QVBoxLayout(dialog)
        message_label = QLabel("长时间未核验通过，请进行人工确认。", dialog)
        layout.addWidget(message_label)

        # 添加确认按钮
        manual_button = QPushButton("已人工确认，继续作业", dialog)
        manual_button.clicked.connect(lambda: self.manual_confirmation_action(dialog))
        layout.addWidget(manual_button)

        dialog.exec_()  # 显示模态对话框，直到关闭

    def manual_confirmation_action(self, dialog):
        """手动确认后的操作"""
        # 更新标记，表示人工确认已完成
        self.verification_success = True

        # 关闭对话框
        dialog.accept()

        # 更新 OCR 结果标签为人工确认后的状态
        self.camera_ocr_label.setText("人工确认已完成，继续作业...")

        # 重启计时器，准备下一次 OCR 核验
        self.verification_timer.blockSignals(False)  # 解除信号阻断
        self.verification_success = False
        self.verification_timer.start()  # 重新开始计时，等待10秒超时处理


    def closeEvent(self, event):
        # 关闭程序时释放摄像头资源
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    # 创建应用程序
    app = QApplication(sys.argv)

    # 创建主窗口
    window = ImageLoaderApp()
    window.show()

    # 运行应用程序
    sys.exit(app.exec_())
