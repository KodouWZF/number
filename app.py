import tkinter as tk
import cv2
import torch
import numpy as np
import PIL.Image as Image

from torchvision import transforms
from tkinter import filedialog

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）



# 创建主窗口
root = tk.Tk()
root.title("识别程序")

# 设置窗口大小
root.geometry("400x300")

model = torch.load("./models/model_Mnist_10.pth", weights_only=False)  # 加载模型

# 定义选择文件识别函数
def select_file():
    file_path = filedialog.askopenfilename()
    #print(file_path)
    img = Image.open(file_path)
    #print(img.size)
    img = img.resize((28, 28))  # 调整图像大小
    img_array = np.array(img)  # 将图像转换为NumPy数组
    #print(img_array.shape)
    if img_array.shape != (28, 28):
        img_array = img_array[:, :, 0]
    
    #print(img_array.shape)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)  # 将图像转换为Tensor
    #print(img_tensor.shape)
    
    output = model(img_tensor)[0]
    predicted = torch.argmax(output).item()
    result_label.config(text=f"预测结果: {predicted}")
    return predicted



# 定义摄像头识别函数
def camera_recognition():
    # 打开默认摄像头（通常是第一个摄像头）
    cap = cv2.VideoCapture(0)

    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        # 显示帧
        cv2.imshow('frame', frame)
        
        # 等待1ms，如果按下q键或者点击窗口的关闭按钮，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break

    # 当一切都完成时，释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()
    result_label.config(text="识别结果：摄像头已关闭")

# 创建按钮
select_button = tk.Button(root, text="选择文件识别", command=select_file)
camera_button = tk.Button(root, text="摄像头识别", command=camera_recognition)

# 将按钮横向排列在顶部，间距缩小
select_button.pack(side="left", padx=10)
camera_button.pack(side="left", padx=10)

# 创建结果标签
result_label = tk.Label(root, text="识别结果：")

# 布局按钮和标签
select_button.pack(pady=10)
camera_button.pack(pady=10)
result_label.pack(pady=20, side="bottom")

# 运行主循环
root.mainloop()