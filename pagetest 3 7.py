import sys
import os
import cv2
import fitz
import difflib 
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLOv10
from PySide6.QtCore import QDateTime, QFile
from PySide6.QtWidgets import QApplication, QWidget, QApplication
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import QTimer, Qt, QThread, Signal  # 导入 QTimer
from PySide6.QtUiTools import QUiLoader
from paddleocr import PaddleOCR
from tool import predict_and_detect, remove_substrings, draw_bounding_boxes, delete_all_files_in_folder

class ImageWatcherThread(QThread):
    image_added = Signal(str)
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
    image_added = Signal(str)
    def __init__(self):
        super().__init__()
        self.log_file = "operation_log.txt"
        self.image_folder = r'C:\develop\yolov10-main\yolov10-main\testimages'  # 指定图片文件夹路径
        self.watch_folder = r"C:\MijCtrl\Hot\P04-机"
        self.latest_files = []
        self.image_cache = []
        self.current_index = 0  # 当前显示的图片索引
        self.current_index1 = 0
        self.defect = 0
        self.overtime = False
        self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 打开默认摄像头
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 初始化 PaddleOCR

        self.yolo = YOLOv10(model = r"C:\develop\config\model\common.pt")
        self.yolo1 = YOLOv10(model = r"C:\develop\config\model\colour.pt")
        self.yolo2 = YOLOv10(model = r"C:\develop\config\model\defect.pt")


        # self.verification_timer.timeout.connect()
        self.verification_success = False
        self.righticon = cv2.imread(r'C:\develop\config\icon\right.png')
        self.wrongicon =cv2.imread(r'C:\develop\config\icon\wrong.png')
        self.righticon  = cv2.cvtColor(self.righticon , cv2.COLOR_BGR2RGB)
        self.wrongicon  = cv2.cvtColor(self.wrongicon , cv2.COLOR_BGR2RGB)

        # print('success1')
        # self.initUI()
        qfile_stats = QFile(r'C:\develop\config\ui\SK2.ui')
        # print('success2')
        qfile_stats.open(QFile.ReadOnly)
        # print('success3')
        
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)
                # 设置窗口图标
        self.ui.setWindowIcon(QIcon(r'C:\develop\config\icon\PG.png'))
        # self.ui.textBrowser.connect(QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))
        print('ui success')
        self.ui.pushButton.clicked.connect(self.start_timer)
        # 创建定时器
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.matching)
        # 计数器
        self.counter = 0


        self.options = ["打印时间", "订单号后四位", "底漆", "有效像素", "K是否对齐", "是否无色差", "传感器是否对齐", "传感胶纸是否正常", "制版自动合成(向下)","AI核对结果","备注"]
        self.checkboxes = [self.ui.checkBox_2,self.ui.checkBox_3,self.ui.checkBox_4,self.ui.checkBox_5,self.ui.checkBox_6,self.ui.checkBox_7,self.ui.checkBox_8]
        self.ui.pushButton_2.clicked.connect(self.save_to_excel)
        self.AIresult = "未核验"
        self.text = self.ui.textEdit.toPlainText()
        
        self.image_watcher_thread = ImageWatcherThread(self.watch_folder)
        self.image_watcher_thread.image_added.connect(self.on_new_image_added)
        self.image_watcher_thread.start()
        # self.pdf_files = self.get_pdf_files(self.image_folder)  # 获取所有图片文件名
        self.image_files = self.get_image_files(self.image_folder)  # 获取所有图片文件名
        if self.image_files:
            self.show_image(self.image_files[self.current_index])
        # 使用 QTimer 定时更新摄像头画面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(10)  # 每秒更新一次画面
    def save_to_excel(self):
        # 获取勾选状态
        checked_states = [self.timestamp] +[self.image_name]+  [checkbox.isChecked() for checkbox in self.checkboxes] + [self.AIresult] + [self.text]

        # 创建DataFrame
        df = pd.DataFrame([checked_states], columns=self.options)

        # 检查文件是否存在
        file_exists = os.path.isfile('checkbox_data.xlsx')
        if not file_exists:
            df.to_excel('checkbox_data.xlsx', index=False)
        # 保存到Excel
        with pd.ExcelWriter('checkbox_data.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            if not file_exists:
                # 如果文件不存在，写入表头
                df.to_excel(writer, index=False)
            else:
                # 如果文件存在，追加数据，不写入表头
                for sheetname in writer.book.sheetnames:
                    sheet = writer.book[sheetname]
                    startrow = sheet.max_row if sheet.max_row is not None else 0
                    df.to_excel(writer, index=False, header=False, startrow=startrow)

        print("已保存到 checkbox_data.xlsx")
        self.overtime = False
    def on_new_image_added(self, image_file):

        self.defect = 0
        self.AIresult = "未核验"
        src_path = os.path.join(self.watch_folder, image_file)
        dst_path = os.path.join(self.image_folder, image_file)
        if src_path.lower().endswith('.pdf'):
            # 打开 PDF 文件
            pdf_document = fitz.open(src_path)

            # 遍历每一页
            # for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(0)
            # print(1)
            
            # 将页面转换为图片
            pix = page.get_pixmap()
            # print(22222222222222)
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
            delete_all_files_in_folder(self.image_folder)
            delete_all_files_in_folder(self.watch_folder)

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
            delete_all_files_in_folder(self.image_folder)
            delete_all_files_in_folder(self.watch_folder)
        # self.verification_success = False


    def get_image_files(self, folder):
        # 获取指定文件夹中的所有图片文件
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.pdf')
        # print([f for f in os.listdir(folder) if f.lower().endswith(supported_formats)])
        return [f for f in os.listdir(folder) if f.lower().endswith(supported_formats)]
    def display_image_by_index(self, index):
        pixmap = self.image_cache[index]
        self.ui.label_11.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
        self.current_index = index
    def show_image(self, image_name):
        self.ui.label_12.clear()
        self.ui.textBrowser_3.clear()
        self.ui.textEdit.clear()
        self.timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        # 加载并显示图片
        image_path = os.path.join(self.image_folder, image_name)
        self.image_name = image_name
        # 使用 OpenCV 读取图片
        image = cv2.imread(image_path)
        
        # 将图像从 BGR 转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = image.copy()

        result_img, _ = predict_and_detect(self.yolo, image, classes=[], conf=0.7)
        if _[0].boxes:
            # print(_[0])
            self.status = 'contain heart pattern'
        else:
            self.status = 'no heart pattern'  

        # 转换为 QImage 并设置到 QLabel
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(a.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.label_11.setPixmap(QPixmap.fromImage(qt_image).scaled(self.ui.label_11.size()))

        self.ui.textBrowser.setText(self.timestamp)
        # 显示图片文件名，加入固定字段
        self.ui.textBrowser_2.setText(self.image_name)
        
        # 显示 OCR 结果
        ocr_result, box = self.perform_ocr(image_path)
        # self.ocr_result_label.setText(f"OCR 结果:\n{ocr_result}\n图像类别识别结果：\n"+self.status)
        self.current_image_ocr_result = ocr_result
        substrings_to_remove = ["IMU\n","IMU" ]
        self.current_image_ocr_result = remove_substrings(self.current_image_ocr_result, substrings_to_remove)

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
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色从 BGR 到 RGB
            h, w, ch = frame.shape
            bytes_per_line = ch * w

            # 将摄像头画面保存为临时文件
            temp_image_path = "temp_camera_image.jpg"
            cv2.imwrite(temp_image_path, frame)
            # 进行 OCR 识别
            camera_ocr_result, targets = self.perform_ocr(temp_image_path)
            results = self.yolo1.predict(frame, conf=0.5)

            # saved = 0
            # 从检测结果中提取识别框
            for result in results:
                for box in result.boxes:
                    detected_object = frame[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                    # 截取对象图像
                    # detected_object = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

                    # 开启缺陷检测
                    result_tinys = self.yolo2.predict(detected_object,conf = 0.4)
                    # 关闭缺陷检测
                    # result_tinys = []
                    for result_tiny in result_tinys:
                        # id = 0
                        self.defect += len(result_tiny.boxes)
            if results[0].boxes:
                # print(_[0])
                self.status1 = 'contain heart pattern'
            else:
                self.status1 = 'no heart pattern'

            self.fix_value = draw_bounding_boxes(targets)
            # print(self.status1)
            print('校正值: ', self.fix_value)
            substrings_to_remove = ["BAKAL\n","FACIAL\n", "TREATMENT\n", "ESSENCE\n","S\n","SK\n","SK-\n","SK-I","SK-II\n","I\n","FACIAL", "TREATMENT", "ESSENCE","S","SK","SK-","SK-II","I","K-II","K-II\n","PITER","PITERA"]
            self.camera_ocr_result = remove_substrings(camera_ocr_result, substrings_to_remove)
    def start_timer(self):
        # self.ui.textEdit.clear()
        # self.ui.textBrowser_3.clear()
        self.ui.label_12.setText("核验进行中...")
        self.counter = 0
        self.timer1.start(10)  # 每100毫秒（0.1秒）执行一次任务
    def matching(self): 
        # self.ui.label_12.setText("核验进行中...")
        if self.counter >= 20:  # 运行秒后停止
            self.timer1.stop()
            if self.similarity < 0.4:
                info1 = '字符不匹配 '
            else:
                info1 = ''
            if self.fix_value>=15:
                info2 = '打印位置不匹配 '
            else:
                info2 = ''
            if self.status==self.status1:
                info3 = ''
            else:
                info3 = '图案不匹配'

            result = "核验超时 "+info1+info2+info3
            self.AIresult ="核验超时 "+info1+info2+info3
            self.ui.textBrowser_3.setText(result)
            
            # 转换为 QImage 并设置到 QLabel
            h, w, ch = self.wrongicon.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.wrongicon.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.ui.label_12.setPixmap(QPixmap.fromImage(qt_image).scaled(self.ui.label_12.size()))
            # self.verification_success = True
            self.ui.pushButton.setEnabled(True)
            self.defect = 0
            return
        self.counter +=1
        print('指示物数量:{}'.format(self.defect))
        print(f"核验进行中: {self.counter/10}秒")
        # 禁用按钮以防止重复点击
        self.ui.pushButton.setEnabled(False)
        # self.verification_timer = QTimer(self)
        # self.verification_timer.singleShot(10000, self.show_manual_confirmation)  # 设置计时器为10秒

        self.similarity = self.calculate_similarity(self.current_image_ocr_result, self.camera_ocr_result)
        if self.defect>=3:
            self.timer1.stop()
            result = "核验超时 "+ '有底漆缺陷'
            self.ui.textBrowser_3.setText(result)
            
            # 转换为 QImage 并设置到 QLabel
            h, w, ch = self.wrongicon.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.wrongicon.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.ui.label_12.setPixmap(QPixmap.fromImage(qt_image).scaled(self.ui.label_12.size()))
            self.AIresult = "核验超时 "+ '有底漆缺陷'
            # self.verification_success = True
            self.ui.pushButton.setEnabled(True)
            self.defect = 0
            return
        # 如果相似度大于等于 50%，则显示核验通过并切换图片
        print('*********', self.current_image_ocr_result, '     ', self.camera_ocr_result)

        if self.similarity >= 0.4 and self.fix_value < 15 and self.status==self.status1:
            # print('similarity:', self.similarity)
            # QTimer.singleShot(3000, self.show_verification_success)
            self.timer1.stop()
            result = "核验通过, 请保存核验结果"
            self.AIresult ="核验通过"
            self.ui.textBrowser_3.setText(result)
            
            # 转换为 QImage 并设置到 QLabel
            h, w, ch = self.righticon.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.righticon.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.ui.label_12.setPixmap(QPixmap.fromImage(qt_image).scaled(self.ui.label_12.size()))
            self.verification_success = True
            self.ui.pushButton.setEnabled(True)
            self.defect = 0
            return
    def show_manual_confirmation(self):
        if self.verification_success:
            return
        
        self.overtime = True
        if self.similarity < 0.4:
            info1 = '字符不匹配 '
        else:
            info1 = ''
        if self.fix_value>=15:
            info2 = '打印位置不匹配 '
        else:
            info2 = ''
        if self.status==self.status1:
            info3 = ''
        else:
            info3 = '图案不匹配'

        result = "核验超时 "+info1+info2+info3
        self.AIresult ="核验超时 "+info1+info2+info3
        self.ui.textBrowser_3.setText(result)
        
        # 转换为 QImage 并设置到 QLabel
        h, w, ch = self.wrongicon.shape
        bytes_per_line = ch * w
        qt_image = QImage(self.wrongicon.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.label_12.setPixmap(QPixmap.fromImage(qt_image).scaled(self.ui.label_12.size()))
        # self.verification_success = True
        self.ui.pushButton.setEnabled(True)
        return
    def closeEvent(self, event):
        # 关闭程序时释放摄像头资源
        self.cap.release()
        super().closeEvent(event)
if __name__ == '__main__':
    loader = QUiLoader()
    # app = QApplication(sys.argv)
    # 创建应用程序
    app = QApplication(sys.argv)

    # 创建主窗口
    window = ImageLoaderApp()
    window.ui.show()

    # 运行应用程序
    sys.exit(app.exec_())
