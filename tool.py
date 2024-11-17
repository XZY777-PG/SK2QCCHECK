import cv2
import os
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import time
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
def draw_bounding_boxes(target_locations):
    # 创建一个副本以避免修改原始图像
    # upp = 0
    # downn = 0
    rk = []
    # 遍历目标位置信息列表
    a = 0
    for target in target_locations:
        # 提取目标的四个坐标点
        points = np.array(target, np.int32)
        points = points.reshape((-1, 1, 2))

        # 计算四个顶点的平均值作为横坐标
        x_mean = np.mean(points[:, 0, 0])
        # if downn == 0 or x_mean < downn:
        #     downn = x_mean
        # elif upp ==0 or x_mean > upp:
        #     upp = x_mean
        rk.append(x_mean)
        # 在不规则四边形中间绘制垂直线
        if rk:
            # print(max(rk) - min(rk))
            a = max(rk) - min(rk)    
    return a

def delete_all_files_in_folder(folder_path):
    """
    删除指定文件夹路径下的所有文件，但保留子文件夹。
   
    :param folder_path: 要清空的文件夹路径
    """
    print("将在10秒后删除文件夹中所有文件")
    time.sleep(5)
    # 检查路径是否存在且是一个文件夹
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"路径不存在或不是一个文件夹: {folder_path}")
        return
 
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
       
        # 如果是文件，则删除
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
 
    print("所有文件已删除。")