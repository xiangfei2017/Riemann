#!/usr/bin/env python3
"""
MNIST手写数字识别GUI程序

本程序实现了一个基于tkinter的GUI界面，用于：
1. 基于riemann库对MNIST数据集进行训练，显示训练进度
2. 保存、恢复模型的参数和超参数
3. 鼠标手写数字，用已训练模型识别并动态显示预测结果
"""
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
from PIL import Image, ImageDraw

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann库模块
import riemann
import riemann.nn as nn
import riemann.optim as opt
from riemann.vision.datasets import MNIST
from riemann.vision import transforms
from riemann.utils.data import DataLoader
from riemann import tensor
from riemann.nn.functional import softmax


class MNISTGUI:
    """MNIST手写数字识别GUI程序"""
    
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        self.root.title("MNIST手写数字识别")
        self.root.geometry("680x600")
        self.root.resizable(True, True)
        
        # 初始化变量
        self.model = None
        self.training = False
        self.train_thread = None
        self.data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
        # 共享的测试数据集
        self.test_dataset = None
        # 加载测试数据集
        self._load_test_dataset()
        
        # 检测是否有可用的GPU
        try:
            # 直接检查 cuda 是否可用，不检查属性是否存在
            if hasattr(riemann, 'cuda') and riemann.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        except Exception as e:
            self.device = 'cpu'
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标签页作为主容器
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建模型训练与管理标签页
        self.train_manage_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_manage_tab, text="模型训练与管理")
        
        # 创建手写识别标签页
        self.digit_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.digit_tab, text="手写识别")
        
        # 创建MNIST测试集标签页
        self.mnist_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mnist_tab, text="MNIST测试集")
        
        # 在训练与管理标签页中创建布局
        self.top_frame = ttk.Frame(self.train_manage_tab)
        self.top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建左侧参数输入区域
        self.left_frame = ttk.Frame(self.top_frame, width=220)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_frame.pack_propagate(False)  # 固定宽度
        
        # 创建右侧数据显示区域
        self.right_frame = ttk.Frame(self.top_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 初始化各区域
        self._init_left_panel()  # 左侧参数输入和控制按钮
        self._init_right_panel()  # 右侧训练数据显示和模型状态
        self._init_digit_recognition()  # 手写识别功能
        self._init_mnist_test_tab()  # MNIST测试集标签页
    
    def _load_test_dataset(self):
        """加载测试数据集（只加载一次）"""
        try:
            # 定义数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 加载测试数据集
            print("加载MNIST测试数据集...")
            self.test_dataset = MNIST(
                root=self.data_root,
                train=False,
                transform=transform
            )
            print(f"MNIST测试数据集加载完成，共 {len(self.test_dataset)} 个样本")
        except Exception as e:
            print(f"加载MNIST测试数据集失败: {str(e)}")
            self.test_dataset = None
        
    def _init_left_panel(self):
        """初始化左侧面板，包含训练参数输入和控制按钮"""
        # 创建训练设置框架
        settings_frame = ttk.LabelFrame(self.left_frame, text="训练设置", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        # 批大小设置
        batch_frame = ttk.Frame(settings_frame)
        batch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(batch_frame, text="批大小:", width=8, anchor=tk.E).pack(side=tk.LEFT, padx=5)
        self.batch_var = tk.StringVar(value="100")
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # 训练轮数设置
        epoch_frame = ttk.Frame(settings_frame)
        epoch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(epoch_frame, text="训练轮数:", width=8, anchor=tk.E).pack(side=tk.LEFT, padx=5)
        self.epoch_var = tk.StringVar(value="3")
        ttk.Entry(epoch_frame, textvariable=self.epoch_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # 学习率设置
        lr_frame = ttk.Frame(settings_frame)
        lr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(lr_frame, text="学习率:", width=8, anchor=tk.E).pack(side=tk.LEFT, padx=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # 训练按钮框架
        train_button_frame = ttk.Frame(self.left_frame)
        train_button_frame.pack(fill=tk.X, pady=10)
        self.train_button = ttk.Button(train_button_frame, text="开始训练", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(train_button_frame, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 模型管理框架
        model_frame = ttk.LabelFrame(self.left_frame, text="模型管理", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        # 模型状态
        status_frame = ttk.Frame(model_frame)
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, text="模型状态:").pack(side=tk.LEFT, padx=5)
        self.model_status_var = tk.StringVar(value="模型未初始化")
        ttk.Label(status_frame, textvariable=self.model_status_var, foreground="blue").pack(side=tk.LEFT, padx=5)
        
        # 模型管理按钮
        model_button_frame = ttk.Frame(model_frame)
        model_button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(model_button_frame, text="初始化模型", command=self.init_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_button_frame, text="测试模型", command=self.test_model).pack(side=tk.LEFT, padx=2)
        
        # 第二行按钮
        model_button_frame2 = ttk.Frame(model_frame)
        model_button_frame2.pack(fill=tk.X, pady=5)
        ttk.Button(model_button_frame2, text="保存模型", command=self.save_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_button_frame2, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=2)
    
    def _init_right_panel(self):
        """初始化右侧面板，包含训练数据显示和模型状态"""
        # 创建训练日志框架
        log_frame = ttk.LabelFrame(self.right_frame, text="训练日志", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        progress_frame = ttk.Frame(log_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        # 设置进度条框架为相对定位
        progress_frame.pack_propagate(False)
        progress_frame.config(height=30)
        
        # 创建进度条
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.BOTH, expand=True)
        
        # 创建百分比标签，叠加在进度条上
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # 训练日志
        self.log_text = tk.Text(log_frame, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text.config(state=tk.DISABLED)
    
    def _init_bottom_panel(self):
        """初始化底部面板，包含训练进度条"""
        # 创建进度条框架
        progress_frame = ttk.Frame(self.bottom_frame)
        progress_frame.pack(fill=tk.X)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, padx=(0, 10))
        
        # 进度百分比显示
        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.RIGHT)
    
    def _init_digit_recognition(self):
        """初始化手写识别功能"""
        # 创建主框架
        main_frame = ttk.Frame(self.digit_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建顶部图像区域
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.X, pady=5)
        
        # 创建手写区域框架（正方形）
        draw_frame = ttk.LabelFrame(image_frame, text="手写区域 (28x28)", padding="5")
        draw_frame.pack(side=tk.LEFT, padx=2)
        
        # 设置手写区域为正方形（稍大以容纳画布、标题栏和边框）
        draw_frame.config(width=290, height=310)
        draw_frame.pack_propagate(False)
        
        # 创建画布（28x28像素，放大显示）
        self.canvas = tk.Canvas(draw_frame, width=280, height=280, bg="white", borderwidth=1, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建处理后图像显示框架
        processed_frame = ttk.LabelFrame(image_frame, text="处理后图像", padding="5")
        processed_frame.pack(side=tk.RIGHT, padx=2)
        
        # 设置处理后图像区域为正方形（稍大以容纳画布、标题栏和边框）
        processed_frame.config(width=290, height=310)
        processed_frame.pack_propagate(False)
        
        # 创建处理后图像显示画布
        self.processed_canvas = tk.Canvas(processed_frame, width=280, height=280, bg="white", borderwidth=1, relief=tk.SUNKEN)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 画布绑定事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # 画笔设置
        self.last_x = None
        self.last_y = None
        self.pen_width = 15
        self.pen_color = "black"
        
        # 创建PIL图像用于处理（实际大小28x28）
        self.image = Image.new("L", (28, 28), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # 创建按钮框架（放在区域外部）
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        # 清除按钮
        ttk.Button(button_frame, text="清除", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # 识别按钮
        ttk.Button(button_frame, text="识别", command=self.recognize_digit).pack(side=tk.LEFT, padx=5)
        
        # 预测结果显示
        result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        
        # 预测数字显示
        self.result_var = tk.StringVar(value="预测结果将显示在这里")
        ttk.Label(result_frame, textvariable=self.result_var, font=("Arial", 16)).pack(pady=5)
        
        # 预测概率显示
        self.prob_frame = ttk.Frame(result_frame)
        self.prob_frame.pack(fill=tk.X, pady=5)
        
        # 动态预测选项
        dynamic_frame = ttk.Frame(result_frame)
        dynamic_frame.pack(fill=tk.X, pady=5, expand=True)
        self.dynamic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dynamic_frame, text="动态预测", variable=self.dynamic_var).pack(side=tk.LEFT, padx=5, expand=True)
    
    def init_model(self):
        """初始化模型"""
        try:
            # 创建分类器模型，使用在__init__中检测好的设备
            self.model = Classifier(device=self.device)
            self.model_status_var.set("模型已初始化")
            self.log("模型初始化成功！")
        except Exception as e:
            self.log(f"模型初始化失败: {str(e)}")
    
    def start_training(self):
        """开始训练"""
        if not self.model:
            # 自动初始化模型，使用在__init__中检测好的设备
            try:
                self.model = Classifier(device=self.device)
                self.model_status_var.set("模型已初始化")
                self.log("模型初始化成功！")
            except Exception as e:
                self.log(f"模型初始化失败: {str(e)}")
                return
        
        # 获取训练参数
        try:
            batch_size = int(self.batch_var.get())
            epochs = int(self.epoch_var.get())
            lr = float(self.lr_var.get())
        except ValueError:
            self.log("错误: 请输入有效的训练参数！")
            return
        
        # 禁用按钮
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 清空日志
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # 开始训练线程
        self.training = True
        self.train_thread = threading.Thread(target=self.train_model, args=(batch_size, epochs, lr))
        self.train_thread.daemon = True
        self.train_thread.start()
    
    def stop_training(self):
        """停止训练"""
        self.training = False
        self.stop_button.config(state=tk.DISABLED)
    
    def train_model(self, batch_size, epochs, lr):
        """训练模型"""
        try:
            # 显示使用的设备类型
            self.log(f"使用 {self.device.upper()} 进行训练")
            
            # 验证模型设备一致性
            if hasattr(self.model, 'parameters'):
                first_param = next(self.model.parameters(), None)
                if first_param is not None:
                    param_device = getattr(first_param, 'device', 'unknown')
                    self.log(f"模型参数设备: {param_device}")
            
            # 定义数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 加载数据集
            self.log("加载数据集...")
            train_dataset = MNIST(
                root=self.data_root,
                train=True,
                transform=transform
            )
            # 使用共享的测试数据集
            if self.test_dataset is None:
                test_dataset = MNIST(
                    root=self.data_root,
                    train=False,
                    transform=transform
                )
            else:
                test_dataset = self.test_dataset
            
            # 创建数据加载器
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=100,  # 增大 batch_size 提高评估速度
                shuffle=False
            )
            
            self.log(f"训练集大小: {len(train_dataset)}")
            self.log(f"测试集大小: {len(test_dataset)}")
            
            # 更新模型的学习率
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = lr
                self.log(f"优化器学习率: {param_group['lr']}")
            
            # 训练模型
            total_batches = epochs * len(train_loader)
            current_batch = 0
            
            for epoch in range(epochs):
                if not self.training:
                    self.log("训练已停止")
                    break
                
                self.model.train()
                epoch_loss = 0.0
                
                for batch_idx, batch in enumerate(train_loader):
                    if not self.training:
                        break
                    
                    img_tensors, target_tensors = batch
                    loss = self.model.train_step(img_tensors, target_tensors)
                    epoch_loss += loss.item()
                    
                    # 更新进度
                    current_batch += 1
                    progress = (current_batch / total_batches) * 100
                    self.progress_var.set(progress)
                    
                    # 更新进度百分比显示
                    self.root.after(0, lambda p=progress: self.progress_label.config(text=f"{p:.1f}%"))
                    
                    # 显示当前批次损失
                    if batch_idx % 10 == 0:
                        self.log(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                # 计算平均损失
                avg_loss = epoch_loss / len(train_loader)
                self.log(f'Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}')
                
                # 在测试集上评估
                self.model.eval()
                test_accuracy, test_loss = self.model.evaluate(test_loader)
                self.log(f'测试集准确率: {test_accuracy:.4f}, 测试损失: {test_loss:.4f}')
                self.log('-' * 50)
            
            if self.training:
                self.log("训练完成！")
                self.log("成功: 训练完成！")
        except Exception as e:
            self.log(f"训练出错: {str(e)}")
            self.log(f"错误: 训练失败: {str(e)}")
        finally:
            # 恢复按钮状态
            self.training = False
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.progress_var.set(100))
    
    def log(self, message):
        """添加日志信息"""
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        self.root.after(0, update_log)
    
    def save_model(self):
        """保存模型"""
        if not self.model:
            self.log("警告: 请先初始化模型！")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("模型文件", "*.pth"), ("所有文件", "*")],
            initialdir=os.path.join(os.path.dirname(__file__), "models")
        )
        
        if file_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            try:
                # 保存模型
                self.model.save(file_path)
                self.log("模型保存成功！")
            except Exception as e:
                self.log(f"模型保存失败: {str(e)}")
    
    def load_model(self):
        """加载模型"""
        # 选择模型文件
        file_path = filedialog.askopenfilename(
            filetypes=[("模型文件", "*.pth"), ("所有文件", "*")],
            initialdir=os.path.join(os.path.dirname(__file__), "models")
        )
        
        if file_path:
            try:
                # 初始化模型，使用已检测的设备
                if not self.model:
                    self.model = Classifier(device=self.device)
                
                # 加载模型
                self.model.load(file_path)
                
                # 确保模型在正确的设备上（只进行一次设备迁移）
                if self.model.device != self.device:
                    self.model.to(self.device)
                    self.log(f"模型已从 {self.model.device} 迁移到 {self.device}")
                else:
                    self.log(f"模型设备与目标设备一致: {self.device}")
                
                self.model_status_var.set("模型已加载")
                self.log("模型加载成功！")
                # 显示设备信息
                self.log(f"模型加载后设备: {self.model.device}")
            except Exception as e:
                self.log(f"模型加载失败: {str(e)}")
    
    def test_model(self):
        """测试模型"""
        if not self.model:
            self.log("警告: 请先初始化或加载模型！")
            return
        
        try:
            # 定义数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 使用共享的测试数据集
            if self.test_dataset is None:
                test_dataset = MNIST(
                    root=self.data_root,
                    train=False,
                    transform=transform
                )
            else:
                test_dataset = self.test_dataset
            
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=100,  # 增大 batch_size 提高测试速度
                shuffle=False
            )
            
            # 测试模型
            self.log("测试模型...")
            self.model.eval()
            test_accuracy, test_loss = self.model.evaluate(test_loader)
            
            self.log(f"测试集准确率: {test_accuracy:.4f}")
            self.log(f"测试集损失: {test_loss:.4f}")
            
            self.log(f"测试结果 - 准确率: {test_accuracy:.4f}, 损失: {test_loss:.4f}")
        except Exception as e:
            self.log(f"测试失败: {str(e)}")
    
    def start_draw(self, event):
        """开始绘制"""
        # 画布大小是280x280，映射到28x28的PIL图像
        scaled_x = int(event.x * 28 / 280)
        scaled_y = int(event.y * 28 / 280)
        self.last_x, self.last_y = event.x, event.y
        self.last_scaled_x, self.last_scaled_y = scaled_x, scaled_y
    
    def draw(self, event):
        """绘制"""
        if self.last_x and self.last_y:
            # 在画布上绘制
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.pen_width, fill=self.pen_color,
                capstyle=tk.ROUND, joinstyle=tk.ROUND
            )
            
            # 映射到28x28的PIL图像
            scaled_x = int(event.x * 28 / 280)
            scaled_y = int(event.y * 28 / 280)
            
            # 计算合适的画笔宽度（根据缩放比例，适中增大以匹配MNIST样本的笔画粗细）
            scaled_pen_width = max(2, int(self.pen_width * 28 / 280 * 1.2))
            
            # 在PIL图像上绘制
            self.draw.line(
                [(self.last_scaled_x, self.last_scaled_y), (scaled_x, scaled_y)],
                fill=0, width=scaled_pen_width  # 使用缩放后的画笔宽度
            )
            
            # 更新坐标
            self.last_x, self.last_y = event.x, event.y
            self.last_scaled_x, self.last_scaled_y = scaled_x, scaled_y
            
            # 动态预测
            if self.dynamic_var.get() and self.model:
                self.recognize_digit()
    
    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete("all")
        self.processed_canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.last_scaled_x, self.last_scaled_y = None, None
        self.result_var.set("预测结果将显示在这里")
        
        # 清空概率显示
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
    
    def recognize_digit(self):
        """识别手写数字"""
        if not self.model:
            self.log("警告: 请先初始化或加载模型！")
            return
        
        try:
            # 调整图像大小为28x28
            img = self.image.resize((28, 28), Image.NEAREST)
            
            # 转换为numpy数组
            img_array = np.array(img, dtype=np.float32)
            
            # 反转颜色（MNIST是黑底白字，我们的画布是白底黑字）
            img_array = 255.0 - img_array
            
            # 二值化处理
            threshold = 128.0
            img_array = np.where(img_array > threshold, 255.0, 0.0)
            
            # 确保img_array是28x28大小
            if img_array.shape != (28, 28):
                # 如果不是28x28大小，创建一个新的28x28图像
                temp_img = np.ones((28, 28)) * 255.0
                # 计算居中位置
                h, w = img_array.shape
                start_h = (28 - h) // 2
                start_w = (28 - w) // 2
                # 将原始图像放入居中位置
                temp_img[start_h:start_h+h, start_w:start_w+w] = img_array
                img_array = temp_img
            
            # 归一化
            img_array = img_array / 255.0
            img_array = (img_array - 0.1307) / 0.3081
            
            # 转换为tensor
            img_tensor = tensor(img_array).reshape(1, 1, 28, 28)
            
            # 将输入数据迁移到模型所在设备
            img_tensor = img_tensor.to(self.model.device)
            
            # 设置模型为评估模式，禁用 dropout 和 batch norm 的更新
            self.model.eval()
            
            # 预测
            outputs = self.model.forward(img_tensor)
            
            # 恢复模型为训练模式
            self.model.train()
            
            # 获取预测结果
            predicted = outputs.argmax(dim=1).item()
            
            # 显示处理后的图像
            # 将28x28的处理后图像放大到280x280显示
            processed_img_data = ((img_array * 0.3081) + 0.1307) * 255.0
            processed_img_data = np.clip(processed_img_data, 0, 255).astype(np.uint8)
            # 确保processed_img_data是28x28大小
            if processed_img_data.shape != (28, 28):
                temp_data = np.ones((28, 28), dtype=np.uint8) * 255
                h, w = processed_img_data.shape
                start_h = (28 - h) // 2
                start_w = (28 - w) // 2
                temp_data[start_h:start_h+h, start_w:start_w+w] = processed_img_data
                processed_img_data = temp_data
            # 创建并显示图像
            processed_img = Image.fromarray(processed_img_data)
            processed_img = processed_img.resize((280, 280), Image.NEAREST)
            
            # 将PIL图像转换为tkinter PhotoImage
            from PIL import ImageTk
            processed_photo = ImageTk.PhotoImage(processed_img)
            
            # 清除并显示处理后图像
            self.processed_canvas.delete("all")
            self.processed_canvas.create_image(0, 0, anchor=tk.NW, image=processed_photo)
            
            # 保存引用，避免垃圾回收
            self.processed_photo = processed_photo
            
            # 获取概率
            prob = softmax(outputs, dim=1).detach().tolist()[0]
            
            # 更新结果显示
            self.result_var.set(f"预测结果: {predicted}")
            
            # 更新概率显示
            for widget in self.prob_frame.winfo_children():
                widget.destroy()
            
            for i in range(10):
                prob_label = ttk.Label(self.prob_frame, text=f"{i}: {prob[i]:.2f}")
                prob_label.pack(side=tk.LEFT, padx=10)
                
                # 高亮预测结果
                if i == predicted:
                    prob_label.config(foreground="red", font=("Arial", 10, "bold"))
        except Exception as e:
            self.log(f"识别失败: {str(e)}")

    def _init_mnist_test_tab(self):
        """初始化MNIST测试集标签页"""
        # 创建主框架
        main_frame = ttk.Frame(self.mnist_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建样本显示区域
        sample_frame = ttk.LabelFrame(main_frame, text="MNIST测试样本", padding="10")
        sample_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建网格框架
        grid_frame = ttk.Frame(sample_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 初始化样本显示相关变量
        self.mnist_samples = []
        self.mnist_labels = []
        self.current_page = 0
        self.images_per_page = 12  # 2x6网格
        
        # 加载MNIST测试数据集
        self._load_mnist_test_data()
        
        # 创建2x6的样本网格
        self.sample_buttons = []
        for i in range(2):
            row_buttons = []
            for j in range(6):
                button = ttk.Button(grid_frame, width=12)
                # 设置按钮大小，使高度和宽度一致（正方形）
                button.grid(row=i, column=j, padx=1, pady=1, ipady=12)
                row_buttons.append(button)
            self.sample_buttons.append(row_buttons)
        
        # 创建分页按钮框架
        nav_frame = ttk.Frame(sample_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        
        # 下一页按钮
        ttk.Button(nav_frame, text="下一页", command=self.show_next_page).pack(side=tk.RIGHT, padx=5)
        
        # 预测结果显示区域
        self.mnist_result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="10")
        self.mnist_result_frame.pack(fill=tk.X, pady=5)
        
        # 预测数字显示
        self.mnist_result_var = tk.StringVar(value="点击上方样本图像查看预测结果")
        ttk.Label(self.mnist_result_frame, textvariable=self.mnist_result_var, font=("Arial", 16)).pack(pady=5)
        
        # 预测概率显示
        self.mnist_prob_frame = ttk.Frame(self.mnist_result_frame)
        self.mnist_prob_frame.pack(fill=tk.X, pady=5)
        
        # 显示第一页样本
        self.show_current_page()
    
    def _load_mnist_test_data(self):
        """加载MNIST测试数据集"""
        try:
            # 使用共享的测试数据集
            if self.test_dataset is None:
                # 定义数据变换
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                # 加载测试数据集
                test_dataset = MNIST(
                    root=self.data_root,
                    train=False,
                    transform=transform
                )
            else:
                test_dataset = self.test_dataset
            
            # 提取样本和标签
            self.mnist_test_dataset = test_dataset
            
            # 加载前200个样本用于显示
            self.mnist_samples = []
            self.mnist_labels = []
            for i in range(min(200, len(test_dataset))):
                img, label = test_dataset[i]
                self.mnist_samples.append(img)
                self.mnist_labels.append(label)
                
        except Exception as e:
            print(f"加载MNIST测试数据失败: {str(e)}")
            self.mnist_samples = []
            self.mnist_labels = []
    
    def show_current_page(self):
        """显示当前页的样本"""
        if not self.mnist_samples:
            return
        
        # 计算当前页的起始索引
        start_idx = self.current_page * self.images_per_page
        
        # 清除之前的按钮绑定
        for i in range(2):
            for j in range(6):
                idx = start_idx + i * 6 + j
                if idx < len(self.mnist_samples):
                    # 获取图像和标签
                    img = self.mnist_samples[idx]
                    label = self.mnist_labels[idx]
                    
                    # 转换为PIL图像
                    img_np = img.numpy().squeeze()
                    img_np = ((img_np * 0.3081) + 0.1307) * 255
                    img_np = img_np.astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    
                    # 放大图像（更大尺寸以填充按钮）
                    pil_img = pil_img.resize((90, 90), Image.NEAREST)
                    
                    # 转换为PhotoImage
                    from PIL import ImageTk
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # 更新按钮
                    button = self.sample_buttons[i][j]
                    button.config(image=photo, text="")
                    button.image = photo  # 保存引用
                    
                    # 绑定点击事件
                    button.config(command=lambda idx=idx: self.predict_mnist_sample(idx))
                else:
                    # 清空超出范围的按钮
                    button = self.sample_buttons[i][j]
                    button.config(image="", text="")
                    button.config(command=None)
    
    def show_next_page(self):
        """显示下一页样本"""
        if not self.mnist_samples:
            return
        
        # 计算总页数
        total_pages = (len(self.mnist_samples) + self.images_per_page - 1) // self.images_per_page
        
        # 切换到下一页
        self.current_page = (self.current_page + 1) % total_pages
        self.show_current_page()
    
    def predict_mnist_sample(self, idx):
        """预测MNIST样本"""
        if not self.model:
            self.mnist_result_var.set("请先初始化或加载模型！")
            return
        
        if idx >= len(self.mnist_samples):
            return
        
        try:
            # 获取样本
            img = self.mnist_samples[idx]
            true_label = self.mnist_labels[idx]
            
            # 转换为tensor
            img_tensor = img.unsqueeze(0)
            
            # 将输入数据迁移到模型所在设备
            img_tensor = img_tensor.to(self.model.device)
            
            # 设置模型为评估模式，禁用 dropout 和 batch norm 的更新
            self.model.eval()
            
            # 预测
            outputs = self.model.forward(img_tensor)
            predicted = outputs.argmax(dim=1).item()
            
            # 恢复模型为训练模式
            self.model.train()
            
            # 计算概率
            prob = softmax(outputs, dim=1).detach().tolist()[0]
            
            # 更新结果显示
            self.mnist_result_var.set(f"真实标签: {true_label}, 预测结果: {predicted}")
            
            # 更新概率显示
            for widget in self.mnist_prob_frame.winfo_children():
                widget.destroy()
            
            for i in range(10):
                prob_label = ttk.Label(self.mnist_prob_frame, text=f"{i}: {prob[i]:.3f}")
                prob_label.pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            self.mnist_result_var.set(f"预测失败: {str(e)}")

class Classifier(nn.Module):
    """MNIST手写数字分类器 - 现代LeNet结构"""
    
    def __init__(self, device='cpu'):
        """初始化分类器"""
        super().__init__()
        
        # 保存设备信息
        self.device = device
        
        # 定义现代LeNet结构
        # LeNet是一个经典的用于手写数字识别的卷积神经网络结构
        self.features = nn.Sequential(
            # 第一层卷积 - LeNet风格
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # 1x28x28 -> 6x28x28
            nn.BatchNorm2d(6),  # 添加批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x28x28 -> 6x14x14
            
            # 第二层卷积 - LeNet风格
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 6x14x14 -> 16x10x10
            nn.BatchNorm2d(16),  # 添加批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x10x10 -> 16x5x5
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  # 16x5x5 = 400
            nn.BatchNorm1d(120),  # 添加批量归一化
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),  # 添加批量归一化
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 10)
        )
        
        # 交叉熵损失函数
        self.loss_func = nn.CrossEntropyLoss()
        
        # 将模型移动到指定设备
        if hasattr(self, 'to'):
            super().to(device)
        
        # Adam优化器，优化学习率设置（在设备移动后初始化，确保使用正确设备上的参数）
        self.optimizer = opt.Adam(self.parameters(), 
                                lr=0.001,  # 初始学习率
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=5e-4,  # 增加权重衰减
                                amsgrad=False)
    
    def forward(self, inputs):
        """前向传播"""
        x = self.features(inputs)
        x = self.classifier(x)
        return x
    
    def train_step(self, inputs, targets):
        """训练步骤"""
        # 将输入数据迁移到模型所在设备
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.optimizer.zero_grad(True)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, dataloader):
        """评估模型"""
        # 设置模型为评估模式，禁用 dropout 和 batch norm 的更新
        self.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            img_tensors, target_tensors = batch
            
            # 将输入数据迁移到模型所在设备
            img_tensors = img_tensors.to(self.device)
            target_tensors = target_tensors.to(self.device)
            
            outputs = self.forward(img_tensors)
            
            # 计算损失
            loss = self.loss_func(outputs, target_tensors)
            total_loss += loss.item()
            
            # 计算准确率
            predicted = outputs.argmax(dim=1)
            total += target_tensors.size(0)
            correct += (predicted == target_tensors).sum().item()
        
        # 恢复模型为训练模式
        self.train()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss
    
    def save(self, path):
        """保存模型"""
        # 构建完整的检查点字典
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'device': self.device
        }
        # 使用 Riemann 内置的 save 函数
        riemann.save(checkpoint, path)

    def load(self, path):
        """加载模型"""
        # 使用 Riemann 内置的 load 函数
        checkpoint = riemann.load(path)
        
        # 加载模型状态（不进行设备迁移，保持原始设备）
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复设备信息
        if 'device' in checkpoint:
            self.device = checkpoint['device']
        
        # 重新初始化优化器，确保它使用当前模型的参数
        # 保存优化器的超参数
        lr = self.optimizer.param_groups[0]['lr']
        betas = self.optimizer.param_groups[0]['betas']
        eps = self.optimizer.param_groups[0]['eps']
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        amsgrad = self.optimizer.param_groups[0].get('amsgrad', False)
        
        # 重新初始化优化器，使用当前模型的参数
        self.optimizer = opt.Adam(self.parameters(), 
                                lr=lr,
                                betas=betas,
                                eps=eps,
                                weight_decay=weight_decay,
                                amsgrad=amsgrad)
        
        # 记录加载信息
        print(f"模型加载成功，原设备: {checkpoint.get('device', 'unknown')}")
        print(f"当前模型设备: {self.device}")

    def to(self, device):
        """将模型移动到指定设备"""
        # 调用父类的 to 方法移动模型参数
        super().to(device)
        # 更新设备信息
        self.device = device
        
        # 更新优化器，确保它使用新设备上的参数
        # 保存优化器的超参数
        lr = self.optimizer.param_groups[0]['lr']
        betas = self.optimizer.param_groups[0]['betas']
        eps = self.optimizer.param_groups[0]['eps']
        weight_decay = self.optimizer.param_groups[0]['weight_decay']
        amsgrad = self.optimizer.param_groups[0].get('amsgrad', False)
        
        # 重新初始化优化器，使用新设备上的参数
        self.optimizer = opt.Adam(self.parameters(), 
                                lr=lr,
                                betas=betas,
                                eps=eps,
                                weight_decay=weight_decay,
                                amsgrad=amsgrad)
        
        return self

def main():
    """主函数"""
    root = tk.Tk()
    app = MNISTGUI(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")