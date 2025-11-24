import os
import re
import csv
import math
import time
import heapq
import torch
import pickle
import random
import itertools
import threading
import subprocess
import matplotlib
import torch.nn as nn
import snntorch as snn
import concurrent.futures
import matplotlib.pyplot as plt  # 导入 matplotlib
import matplotlib.patches as patches  # 导入绘制圆形的模块
import networkx as nx
# 测试性能库
import cProfile
import pstats

from snntorch import surrogate
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from scipy.sparse import csr_matrix
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.cm as cm  # 导入 cm 模块
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from DNN_Generator import run_dnn_gui_and_get_network
from datetime import datetime
from  zoneinfo import ZoneInfo
from pathlib import Path
from math import sqrt, log, erf
from collections import defaultdict, deque, OrderedDict

RUN_TAG = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
OUTDIR = Path("out") / RUN_TAG
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===== 统一时间戳 / 输出目录 / 命名助手 =====
def get_model_tag(m) -> str:
    """
    从模型对象/类/函数/字符串提取一个可用于文件名的简洁标签。
    例如 ResNet18 → "ResNet18", "my-model v2" → "my_model_v2"
    """
    if isinstance(m, str):
        name = m
    elif hasattr(m, "__name__"):           # 类/函数/可调用
        name = m.__name__
    elif hasattr(m, "__class__"):          # 实例
        name = m.__class__.__name__
    else:
        name = str(m)
    # 文件名安全化：只保留字母数字与下划线，其余替换为下划线，并折叠连续下划线
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "Model"

def ts_name(prefix: str, suffix: str = ".png") -> str:
    """生成带时间戳的文件名（不含路径）"""
    return f"{prefix}_{RUN_TAG}{suffix}"

def out_path(basename:str) -> Path:
    """拼出文件保存名"""
    return OUTDIR / basename

SAVE_FILE_NAME = "savefile"
matplotlib.use('TKAgg')

gnc_coordinates = {
    0: (3, 3),  1: (2, 3),  2: (1, 3),  3: (0, 3),
    4: (3, 2),  5: (2, 2),  6: (1, 2),  7: (0, 2),
    8: (3, 1),  9: (2, 1), 10: (1, 1), 11: (0, 1),
    12: (3, 0), 13: (2, 0), 14: (1, 0), 15: (0, 0)
}


# 新层类型注册器，用户可自定义层类型
class LayerHandlerRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, layer_type, handler):
        """
        注册一个新的层类型及其处理函数。

        Parameters:
        - layer_type: 新的层的类类型（如 nn.Conv2d）。
        - handler: 处理该层的函数。
        """
        if not issubclass(layer_type, nn.Module):
            raise ValueError("layer_type 必须是 nn.Module 的子类。")
        if not callable(handler):
            raise ValueError("handler 必须是可调用的。")
        self._registry[layer_type] = handler

    def get_handler(self, layer):
        """
        获取与给定层类型对应的处理函数。

        Parameters:
        - layer: 网络层实例。

        Returns:
        - handler 函数或 None。
        """
        for layer_type, handler in self._registry.items():
            if isinstance(layer, layer_type):
                return handler
        return None


class LIFNeuronLayer(nn.Module):

    def __init__(self, out_features: int, threshold: float = 1.0, alpha: float = 0.9):
        super(LIFNeuronLayer, self).__init__()
        self.out_features = out_features  # 输出神经元的数量
        self.threshold = threshold  # 膜电位阈值
        self.alpha = alpha  # 膜电位的衰减因子

        # 初始化膜电位
        self.register_buffer("membrane_potential", torch.zeros(1, out_features))

    def forward(self, x: torch.Tensor):
        """
        计算前向传播，通过膜电位更新进行脉冲发放。
        """
        # 假设 x 的形状是 [batch_size, in_features]
        # 膜电位更新
        self.membrane_potential = self.alpha * self.membrane_potential + x  # 每个时间步更新膜电位

        # 判断是否发放脉冲
        spike = (self.membrane_potential >= self.threshold).float()

        # 如果发放脉冲，重置膜电位为0
        self.membrane_potential = torch.where(spike > 0, torch.zeros_like(self.membrane_potential),
                                              self.membrane_potential)

        return spike  # 返回脉冲（1 或 0）


# 初始化全局注册器
layer_handler_registry = LayerHandlerRegistry()


def handle_conv1d(layer, in_channels, in_length, current_id):
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
    stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    dilation = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation

    # 计算输出长度
    l_out = math.floor(
        (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )

    # 分配神经元ID
    layer_neurons = []
    for out_c in range(out_channels):
        for l in range(l_out):
            layer_neurons.append(current_id)
            current_id += 1

    # 更新下一层的输入维度
    new_in_channels, new_in_length = out_channels, l_out

    return layer_neurons, new_in_channels, new_in_length, current_id


# 处理 Conv2d 层的函数
def handle_conv2d(layer, in_channels, in_height, in_width, current_id):
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation

    # 计算输出特征图的尺寸
    h_out = math.floor(
        (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w_out = math.floor(
        (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    # 分配神经元ID
    layer_neurons = []
    for out_c in range(out_channels):
        for h in range(h_out):
            for w in range(w_out):
                layer_neurons.append(current_id)
                current_id += 1

    # 更新下一层的输入维度
    new_in_channels, new_in_height, new_in_width = out_channels, h_out, w_out

    return layer_neurons, new_in_channels, new_in_height, new_in_width, current_id


def handle_conv3d(layer, in_channels, in_depth, in_height, in_width, current_id):
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,) * 3
    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,) * 3
    dilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,) * 3

    # 计算输出维度
    d_out = math.floor(
        (in_depth + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    h_out = math.floor(
        (in_height + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )
    w_out = math.floor(
        (in_width + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) / stride[2] + 1
    )

    # 分配神经元ID
    layer_neurons = []
    for out_c in range(out_channels):
        for d in range(d_out):
            for h in range(h_out):
                for w in range(w_out):
                    layer_neurons.append(current_id)
                    current_id += 1

    # 更新下一层的输入维度
    new_in_channels, new_in_depth, new_in_height, new_in_width = out_channels, d_out, h_out, w_out

    return layer_neurons, new_in_channels, new_in_depth, new_in_height, new_in_width, current_id


# 处理 Linear 层的函数
def handle_linear(layer, in_features, current_id):
    out_features = layer.out_features
    layer_neurons = list(range(current_id, current_id + out_features))
    current_id += out_features
    return layer_neurons, out_features, None, None, current_id


# 处理 BatchNorm2d 层的函数
def handle_batchnorm(layer, in_channels, in_height, in_width, current_id):
    # BatchNorm 层不增加神经元
    return [], in_channels, in_height, in_width, current_id


# # 处理 ReLU 层的函数
# ---- 测试其它网络时用到，其余时候注释掉
def handle_relu(layer, in_channels, in_height, in_width, current_id):
    # ReLU 层不增加神经元
    return [], in_channels, in_height, in_width, current_id


# 处理 MaxPool2d 层的函数
def handle_maxpool2d(layer, in_channels, in_height, in_width, current_id):
    kernel_size = layer.kernel_size
    stride = layer.stride if layer.stride else layer.kernel_size
    padding = layer.padding

    # 计算输出尺寸
    h_out = math.floor(
        (in_height + 2 * padding - kernel_size) / stride + 1
    )
    w_out = math.floor(
        (in_width + 2 * padding - kernel_size) / stride + 1
    )

    # MaxPool 层不增加神经元，但更新输出尺寸
    return [], in_channels, h_out, w_out, current_id


# 处理 Dropout 层的函数
def handle_dropout(layer, in_channels, in_height, in_width, current_id):
    # Dropout 层不增加神经元
    return [], in_channels, in_height, in_width, current_id


# 处理基于Leaky模型的自定义脉冲神经元层
def handle_self_define_layer_lif(layer, in_features, current_id):
    """
    给层分配神经元 ID 的 handler。

    参数:
    - layer: 自定义层实例
    - in_features: 上一层的特征数（输入size）
    - current_id: 当前神经元 ID 起点

    返回:
    - layer_neurons: 该层的神经元 ID 列表
    - out_features: 该层的输出神经元数目
    - None, None: 因为 1D 情况，我们不需要 in_length
    - updated_current_id: 更新后的 current_id
    """
    out_features = layer.out_features  # 神经元数目
    layer_neurons = list(range(current_id, current_id + out_features))  # 给该层分配神经元 ID
    current_id += out_features  # 更新当前 ID

    return layer_neurons, out_features, None, None, current_id


def register_layer_handler(layer_type):
    """
    装饰器，用于注册新的层类型及其处理函数。

    示例用法:
    # 用户自定义的层
    class CustomLayer(nn.Module):
        def __init__(self, ...):
            super(CustomLayer, self).__init__()
            # 层的初始化

        def forward(self, x):
            # 前向传播逻辑
            return x

    # 用户定义的处理函数
    @register_layer_handler(CustomLayer)
    def handle_custom_layer(layer, in_channels, in_height, in_width, current_id):
        # 假设 CustomLayer 的输出维度与输入相同
        out_channels = in_channels
        h_out, w_out = in_height, in_width

        # 为 CustomLayer 的每个输出神经元分配ID
        layer_neurons = []
        for c in range(out_channels):
            for h in range(h_out):
                for w in range(w_out):
                    layer_neurons.append(current_id)
                    current_id += 1

        # 更新下一层的输入维度
        new_in_channels, new_in_height, new_in_width = out_channels, h_out, w_out

        return layer_neurons, new_in_channels, new_in_height, new_in_width, current_id
    """

    def decorator(func):
        layer_handler_registry.register(layer_type, func)
        return func

    return decorator


# 注册默认支持的层类型及其处理函数
layer_handler_registry.register(nn.Conv1d, handle_conv1d)
layer_handler_registry.register(nn.Conv2d, handle_conv2d)
layer_handler_registry.register(nn.Conv3d, handle_conv3d)
layer_handler_registry.register(nn.Linear, handle_linear)
layer_handler_registry.register(nn.BatchNorm2d, handle_batchnorm)
layer_handler_registry.register(nn.ReLU, handle_relu)
layer_handler_registry.register(nn.MaxPool2d, handle_maxpool2d)
layer_handler_registry.register(nn.Dropout, handle_dropout)
# 处理自定义类leaky层
register_layer_handler(LIFNeuronLayer)(handle_self_define_layer_lif)


class SimpleSNN(nn.Module):
    def __init__(self):
        super(SimpleSNN, self).__init__()

        # 输入层到卷积层1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)  # 输入为3通道
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        # 卷积层1到卷积层2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        # 卷积层2到全连接层1
        self.fc1 = nn.Linear(3 * 3 * 3, 16)  # 输入为3 * 3 * 3（卷积输出），输出为16个神经元 第一个8即为衔接conv的input channel
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

        # 全连接层1到输出层
        self.fc2 = nn.Linear(16, 2)  # 输出层为2个类别
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # 初始化LIF膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # 输入层到卷积层1
        spk1, mem1 = self.lif1(self.conv1(x), mem1)
        # 卷积层1到卷积层2
        spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
        # 卷积层2到全连接层1
        spk2_flat = spk2.view(spk2.size(0), -1)  # 展平
        spk3, mem3 = self.lif3(self.fc1(spk2_flat), mem3)
        # 全连接层1到输出层
        spk4, mem4 = self.lif4(self.fc2(spk3), mem4)
        return spk4, mem4

# ----------------------------- Samples --------------------------------------------------------------------------------
LeNet_MNIST = nn.Sequential(
    # 第1层卷积：输入通道1，输出通道6，卷积核5×5（无填充），输出尺寸24×24
    nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),       # 池化后输出尺寸12×12
    # 第2层卷积：输入通道6，输出通道16，卷积核5×5（无填充），输出尺寸8×8
    nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),       # 池化后输出尺寸4×4
    # 第3层卷积（等效全连接）：输入通道16，输出通道120，卷积核4×4，输出尺寸1×1
    nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0),
    nn.ReLU(),
    # 第4层卷积（等效全连接）：输入通道120，输出通道84，卷积核1×1，输出尺寸1×1
    nn.Conv2d(120, 84, kernel_size=1, stride=1, padding=0),
    nn.ReLU(),
    # 第5层卷积（输出层）：输入通道84，输出通道10，卷积核1×1，输出尺寸1×1
    nn.Conv2d(84, 10, kernel_size=1, stride=1, padding=0)
)

AlexNet = nn.Sequential(
    # 第1层卷积：输出64通道，3×3卷积，stride=2，padding=1，将28×28下采样到14×14
    nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),        # 池化后 14×14 -> 7×7
    # 第2层卷积：输出128通道，3×3卷积，stride=1，padding=1，保持7×7
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 池化后 7×7 -> 3×3
    # 第3层卷积：输出192通道，3×3卷积，stride=1，padding=1，保持3×3
    nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 第4层卷积：输出192通道，3×3卷积，stride=1，padding=1，保持3×3
    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 第5层卷积：输出128通道，3×3卷积，stride=1, padding=1，保持3×3
    nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 池化后 3×3 -> 1×1
    # 全连接部分等效实现：使用1×1卷积模拟全连接层
    nn.Conv2d(128, 1024, kernel_size=1, stride=1, padding=0),  # 等效4096神经元全连接（缩小为1024）
    nn.ReLU(),
    nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0), # 第二层全连接
    nn.ReLU(),
    nn.Conv2d(1024, 10, kernel_size=1, stride=1, padding=0)    # 输出层，全连接到10类
)

MobileNet = nn.Sequential(
    # 初始标准卷积
    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 输出通道32，28×28 -> 14×14
    nn.ReLU(),
    # 深度可分离卷积块1（不降采样）
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Depthwise 32->32，保持14×14
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),            # Pointwise 32->64
    nn.ReLU(),
    # 深度可分离卷积块2（降采样）
    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),  # Depthwise 64->64，14×14 -> 7×7
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),           # Pointwise 64->128
    nn.ReLU(),
    # 深度可分离卷积块3（不降采样）
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128),  # Depthwise 128->128，保持7×7
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),              # Pointwise 128->128（通道不变）
    nn.ReLU(),
    # 深度可分离卷积块4（降采样）
    nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128),  # Depthwise 128->128，7×7 -> 4×4
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),             # Pointwise 128->256
    nn.ReLU(),
    # 深度可分离卷积块5（降采样）
    nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256),  # Depthwise 256->256，4×4 -> 2×2
    nn.ReLU(),
    nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),             # Pointwise 256->512
    nn.ReLU(),
    # 深度可分离卷积块6（降采样到1×1）
    nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512),  # Depthwise 512->512，2×2 -> 1×1
    nn.ReLU(),
    nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),            # Pointwise 512->1024，输出1×1特征
    nn.ReLU(),
    # 输出层，1×1卷积（等效全连接）
    nn.Conv2d(1024, 10, kernel_size=1, stride=1, padding=0)              # 输出10个神经元
)

InceptionV3 = nn.Sequential(
    # 初始卷积层堆叠
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),    # 保持28×28
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),   # 下采样到14×14
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   # 保持14×14，增加通道
    nn.ReLU(),
    nn.Conv2d(64, 80, kernel_size=1, stride=1, padding=0),   # 1×1卷积，通道压缩/变换
    nn.ReLU(),
    nn.Conv2d(80, 128, kernel_size=3, stride=1, padding=1),  # 保持14×14
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),                   # 下采样到7×7
    # 模拟 Inception 模块：混合1×1、3×3、5×5卷积
    nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),  # 1×1卷积
    nn.ReLU(),
    nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),  # 3×3卷积
    nn.ReLU(),
    nn.Conv2d(192, 256, kernel_size=5, stride=1, padding=2),  # 较大卷积核5×5
    nn.ReLU(),
    nn.Conv2d(256, 320, kernel_size=3, stride=2, padding=1),  # 下采样到4×4
    nn.ReLU(),
    nn.Conv2d(320, 512, kernel_size=3, stride=1, padding=1),  # 保持4×4，增大通道
    nn.ReLU(),
    nn.Conv2d(512, 1024, kernel_size=4, stride=1, padding=0), # 卷积核4×4覆盖全局，输出1×1
    nn.ReLU(),
    nn.Conv2d(1024, 10, kernel_size=1, stride=1, padding=0)   # 输出层，10个神经元
)

ResNet18 = nn.Sequential(
    # 初始卷积层（类似ResNet-18的7×7卷积+Pool合并效果）
    nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),   # 28×28 -> 14×14, 输出通道64
    nn.ReLU(),
    # 第1组卷积块 (残差结构简化为顺序两层卷积, 通道64, 保持14×14)
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 第2组卷积块 (下采样至7×7, 通道提升至128)
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14×14 -> 7×7
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 第3组卷积块 (下采样至4×4, 通道提升至256)
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 7×7 -> 4×4
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 第4组卷积块 (下采样至2×2, 通道提升至512)
    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 4×4 -> 2×2
    nn.ReLU(),
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    # 输出层：2×2卷积将特征图变为1×1，输出10个通道
    nn.Conv2d(512, 10, kernel_size=2, stride=1, padding=0)    # 2×2 -> 1×1，输出10
)

# -------------------------------------- Functionalities --------------------------------------------------------------

def get_model_input_size(model: nn.Module, sample_input: torch.Tensor) -> Any:
    """
    自动推断模型的输入尺寸。

    参数:
    - model: 神经网络模型 (`nn.Module`)。
    - sample_input: 一个代表输入数据的样本张量。

    返回:
    - input_size: 一个元组，表示输入的维度（通道数，高度，宽度）。
    """

    def hook(module, input, output):
        nonlocal input_size
        if input_size is None:
            print(f"钩子触发，输入形状: {input[0].shape}")  # 输出输入形状
            input_size = input[0].shape[1:]  # 跳过 batch 维度

    input_size = None  # 初始化为 None
    handle = None  # 确保 handle 进行了初始化

    # 注册钩子到第一个叶子节点（不包含子模块的模块）
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 没有子模块的模块，一般是具体的层，如 nn.Conv2d, nn.Linear
            print(f"注册钩子到模块: {name}")
            handle = module.register_forward_hook(hook)
            break

    if handle is None:
        raise ValueError("没有找到合适的叶子模块来注册钩子。")

    # 执行一次前向传播
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁梯度运算，加速并节省内存
        model(sample_input)

    # 移除钩子
    handle.remove()

    if input_size is None:
        raise ValueError("无法自动推断input size。请手动提供 `input_size`。")

    return input_size


def assign_neuron_ids_1d(model: nn.Module, input_size):
    """
    为1D网络中的每个神经元分配唯一的全局ID。

    参数:
    - model: 神经网络模型 (`nn.Module`)。
    - input_size: 表示输入维度的元组（通道数，长度）。

    返回:
    - neuron_id_map: 一个字典，将层名称映射到神经元ID的列表。
    - total_neurons: 网络中神经元的总数。
    """
    neuron_id_map = {}
    current_id = 0

    # 输入层神经元ID
    in_channels, in_length = input_size
    input_neurons = list(range(current_id, current_id + in_channels * in_length))
    neuron_id_map['input'] = input_neurons
    current_id += in_channels * in_length

    # 收集所有支持的层
    supported_layers = []
    for name, layer in model.named_modules():
        if name == '':
            continue  # 跳过顶层模块
        handler = layer_handler_registry.get_handler(layer)
        if handler is not None:
            supported_layers.append((name, layer))

    if not supported_layers:
        raise ValueError("模型中没有找到支持的层类型。")

    # 记录最后一个支持的层
    last_layer_name = supported_layers[-1][0]

    # 迭代每一层并分配神经元ID
    for name, layer in supported_layers:
        handler = layer_handler_registry.get_handler(layer)
        if handler is None:
            continue  # 跳过不支持的层

        # 根据层类型调用相应的处理函数
        if isinstance(layer, nn.Conv1d):
            layer_neurons, in_channels, in_length, current_id = handler(
                layer, in_channels, in_length, current_id
            )
        elif isinstance(layer, nn.Linear):
            layer_neurons, in_features, _, _, current_id = handler(
                layer, in_channels, current_id
            )
        elif isinstance(layer, (nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d, nn.Dropout)):
            layer_neurons, in_channels, in_length, current_id = handler(
                layer, in_channels, in_length, current_id
            )
        elif isinstance(layer, LIFNeuronLayer):
            # 处理 LIFNeuronLayer
            layer_neurons, out_feats, _, _, current_id = handler(
                layer, in_channels, current_id
            )
        else:
            # 对于未明确处理的层类型
            layer_neurons = []

        if layer_neurons:
            # 判断是否为输出层
            if name == last_layer_name:
                key = 'output'
            else:
                key = name
            neuron_id_map[key] = layer_neurons

    total_neurons = current_id
    return neuron_id_map, total_neurons


def assign_neuron_ids_2d(model: nn.Module, input_size):
    """
    为2D网络中的每个神经元分配唯一的全局ID。

    参数:
    - model: 神经网络模型 (`nn.Module`)。
    - input_size: 表示输入维度的元组（通道数，高度，宽度）。

    返回:
    - neuron_id_map: 一个字典，将层名称映射到神经元ID的列表。
    - total_neurons: 网络中神经元的总数。
    """
    neuron_id_map = {}
    current_id = 0

    # 输入层神经元ID
    in_channels, in_height, in_width = input_size
    input_neurons = list(range(current_id, current_id + in_channels * in_height * in_width))
    neuron_id_map['input'] = input_neurons
    current_id += in_channels * in_height * in_width

    # 收集所有支持的层
    supported_layers = []
    for name, layer in model.named_modules():
        if name == '':
            continue  # 跳过顶层模块
        handler = layer_handler_registry.get_handler(layer)
        if handler is not None:
            supported_layers.append((name, layer))

    if not supported_layers:
        raise ValueError("模型中没有找到支持的层类型。")

    # 记录最后一个支持的层
    last_layer_name = supported_layers[-1][0]

    # 迭代每一层并分配神经元ID
    for name, layer in supported_layers:
        handler = layer_handler_registry.get_handler(layer)
        if handler is None:
            continue  # 跳过不支持的层

        # 根据层类型调用相应的处理函数
        if isinstance(layer, nn.Conv2d):
            layer_neurons, in_channels, in_height, in_width, current_id = handler(
                layer, in_channels, in_height, in_width, current_id
            )
        elif isinstance(layer, nn.Linear):
            in_features = in_channels * (in_height or 1) * (in_width or 1)
            # layer_neurons, in_features, _, _, current_id = handler(
            #     layer, in_channels, current_id
            # )
            layer_neurons, out_features, _, _, current_id = handle_linear(layer, in_features, current_id)
            in_channels = out_features
            in_height, in_width = 1, 1
        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Dropout)):
            layer_neurons, in_channels, in_height, in_width, current_id = handler(
                layer, in_channels, in_height, in_width, current_id
            )
        else:
            # 对于未明确处理的层类型
            layer_neurons = []

        if layer_neurons:
            # 判断是否为输出层
            if name == last_layer_name:
                key = 'output'
            else:
                key = name
            neuron_id_map[key] = layer_neurons

    total_neurons = current_id
    return neuron_id_map, total_neurons


def assign_neuron_ids_3d(model: nn.Module, input_size):
    """
    为3D网络中的每个神经元分配唯一的全局ID。

    参数:
    - model: 神经网络模型 (`nn.Module`)。
    - input_size: 表示输入维度的元组（通道数，深度，高度，宽度）。

    返回:
    - neuron_id_map: 一个字典，将层名称映射到神经元ID的列表。
    - total_neurons: 网络中神经元的总数。
    """
    neuron_id_map = {}
    current_id = 0

    # 输入层神经元ID
    in_channels, in_depth, in_height, in_width = input_size
    input_neurons = list(range(current_id, current_id + in_channels * in_depth * in_height * in_width))
    neuron_id_map['input'] = input_neurons
    current_id += in_channels * in_depth * in_height * in_width

    # 收集所有支持的层
    supported_layers = []
    for name, layer in model.named_modules():
        if name == '':
            continue  # 跳过顶层模块
        handler = layer_handler_registry.get_handler(layer)
        if handler is not None:
            supported_layers.append((name, layer))

    if not supported_layers:
        raise ValueError("模型中没有找到支持的层类型。")

    # 记录最后一个支持的层
    last_layer_name = supported_layers[-1][0]

    # 迭代每一层并分配神经元ID
    for name, layer in supported_layers:
        handler = layer_handler_registry.get_handler(layer)
        if handler is None:
            continue  # 跳过不支持的层

        # 根据层类型调用相应的处理函数
        if isinstance(layer, nn.Conv3d):
            layer_neurons, in_channels, in_depth, in_height, in_width, current_id = handler(
                layer, in_channels, in_depth, in_height, in_width, current_id
            )
        elif isinstance(layer, nn.Linear):
            layer_neurons, in_features, _, _, current_id = handler(
                layer, in_channels, current_id
            )
        elif isinstance(layer, (nn.BatchNorm3d, nn.ReLU, nn.MaxPool3d, nn.Dropout)):
            layer_neurons, in_channels, in_depth, in_height, in_width, current_id = handler(
                layer, in_channels, in_depth, in_height, in_width, current_id
            )
        else:
            # 对于未明确处理的层类型
            layer_neurons = []

        if layer_neurons:
            # 判断是否为输出层
            if name == last_layer_name:
                key = 'output'
            else:
                key = name
            neuron_id_map[key] = layer_neurons

    total_neurons = current_id
    return neuron_id_map, total_neurons


def assign_select(model: nn.Module, input_size: Any):
    """
    根据输入尺寸选择合适的 assign_neuron_ids_xd 函数。

    参数:
    - model: 神经网络模型 (`nn.Module`)。
    - input_size: 表示输入维度的元组。

    返回:
    - neuron_id_map: 一个字典，将层名称映射到神经元ID的列表。
    - total_neurons: 网络中神经元的总数。
    """
    if len(input_size) == 2:
        # 1D
        return assign_neuron_ids_1d(model, input_size)
    elif len(input_size) == 3:
        # 2D
        return assign_neuron_ids_2d(model, input_size)
    elif len(input_size) == 4:
        # 3D
        return assign_neuron_ids_3d(model, input_size)
    else:
        raise ValueError("不支持的 input_size 维度。")

# ---- 测试DNN网络专用 ----
def build_connections_1d(model: nn.Module, neuron_id_map: dict, input_size):
    """
    Constructs the connection relationships between neurons in a 1D network.
    This function supports Conv1d and Linear layers.

    Parameters:
    - model: The neural network model (`nn.Module`).
    - neuron_id_map: Dictionary mapping layer names (or 'input','output') to lists of neuron IDs.
    - input_size: Tuple representing the 1D input dimensions: (in_channels, in_length).

    Returns:
    - connections: List of tuples representing connections: (source_id, target_id, weight).
    """
    connections = []

    # ---------------------- 1) 解析输入层信息 ----------------------
    # 假设 input_size = (in_channels, in_length)
    in_channels, in_length = input_size
    input_neurons = neuron_id_map['input']

    # ---------------------- 2) 收集 Conv1d 和 Linear 层 ----------------------
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            layers.append((name, layer))

    if not layers:
        raise ValueError("No Conv1d or Linear layers found in the model.")

    # 最后一个 Conv1d/Linear 层
    output_layer_name, output_layer = layers[-1]

    # 将“上一层的神经元列表”初始化为输入层的神经元
    prev_neurons = input_neurons

    # ---------------------- 3) 逐层构建连接 ----------------------
    for idx, (name, layer) in enumerate(layers):
        is_output = (layer == output_layer)
        current_key = 'output' if is_output else name
        current_layer_neurons = neuron_id_map[current_key]

        if isinstance(layer, nn.Conv1d):
            # ---- 提取 Conv1d 参数 ----
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
            stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride,)
            padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,)
            dilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation,)

            # ---- 计算输出长度 ----
            l_out = math.floor(
                (in_length + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            )

            # 遍历输出特征图 positions (out_c, out_pos)
            idx_current = 0
            for out_c in range(out_channels):
                for out_pos in range(l_out):
                    target_id = current_layer_neurons[idx_current]

                    # 计算在输入上的receptive field
                    # out_pos 对应输入起点 = out_pos * stride[0], ...
                    for k_x in range(kernel_size[0]):
                        in_pos = out_pos * stride[0] + k_x * dilation[0] - padding[0]
                        # 判断 in_pos 是否在 [0, in_length)
                        if 0 <= in_pos < in_length:
                            for in_c in range(layer.in_channels):
                                # 计算 source neuron index
                                source_neuron_index = in_c * in_length + in_pos
                                if source_neuron_index >= len(prev_neurons):
                                    raise IndexError(
                                        f"Source neuron index {source_neuron_index} out of range "
                                        f"for previous layer with length {len(prev_neurons)}."
                                    )
                                source_id = prev_neurons[source_neuron_index]
                                # 获取权重
                                w = layer.weight.data[out_c, in_c, k_x].item()
                                if w != 0:
                                    connections.append((source_id, target_id, w))
                    idx_current += 1

            # ---- 更新输入维度并记录当前层神经元为下一层输入 ----
            in_channels = out_channels
            in_length = l_out
            prev_neurons = current_layer_neurons

        elif isinstance(layer, nn.Linear):
            out_features = layer.out_features
            in_features = layer.in_features
            weight_matrix = layer.weight.data  # shape (out_features, in_features)

            for out_idx in range(out_features):
                for in_idx in range(in_features):
                    w = weight_matrix[out_idx, in_idx].item()
                    if w != 0:
                        source_id = prev_neurons[in_idx]
                        target_id = current_layer_neurons[out_idx]
                        connections.append((source_id, target_id, w))

            # ---- 更新 'previous neurons' ----
            prev_neurons = current_layer_neurons

    return connections


## 2D 版本的build connection
def build_connections(model: nn.Module, neuron_id_map: dict, input_size):
    """
    Constructs the connection relationships between neurons in the network.
    Also handles:
      - MaxPool2d layers to update dimension without building any new connections
      - Grouped/Depthwise Conv2d (groups>1) by indexing weight.shape = (out_channels, in_channels//groups, kH, kW)

    Parameters:
    - model: The neural network model (nn.Module).
    - neuron_id_map: Dictionary {layer_name or 'input'/'output': [list of neuron IDs]}.
    - input_size: (in_channels, in_height, in_width).

    Returns:
    - connections: List of (source_id, target_id, weight).
    """
    import math
    import torch.nn as nn
    import numpy as np
    from collections import defaultdict

    connections = []

    in_channels, in_height, in_width = input_size
    input_neurons = neuron_id_map['input']

    # Collect layers including Conv2d, Linear, MaxPool2d
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
            layers.append((name, layer))

    if not layers:
        raise ValueError("No Conv2d/Linear/MaxPool2d layers found in model.")

    # Identify the final layer
    output_layer_name, output_layer = layers[-1]
    prev_neurons = input_neurons  # neurons from previous layer

    for idx, (name, layer) in enumerate(layers):
        is_output = (layer == output_layer)
        current_key = 'output' if is_output else name

        # ----------------------------------------------------------------------
        # If it's a MaxPool2d => just update the dimension, no new connections
        # ----------------------------------------------------------------------
        if isinstance(layer, nn.MaxPool2d):
            ksize = layer.kernel_size
            stride = layer.stride or ksize
            pad = layer.padding

            # unify possible int/tuple forms
            if isinstance(ksize, int):
                ksize = (ksize, ksize)
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(pad, int):
                pad = (pad, pad)

            # compute output dimension
            h_out = math.floor((in_height + 2*pad[0] - ksize[0]) / stride[0] + 1)
            w_out = math.floor((in_width  + 2*pad[1] - ksize[1]) / stride[1] + 1)

            # update dimension
            in_height, in_width = h_out, w_out
            # prev_neurons remain the same
            continue

        # get the list of target neurons from neuron_id_map
        current_layer_neurons = neuron_id_map[current_key]

        # ----------------------------------------------------------------------
        # Handle Conv2d: normal or grouped (including depthwise)
        # ----------------------------------------------------------------------
        if isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels
            in_channels_ = layer.in_channels    # actual input channels from layer definition
            groups_ = layer.groups
            ksize = layer.kernel_size
            stride = layer.stride
            pad = layer.padding
            dil = layer.dilation

            # compute output dimension
            h_out = math.floor(
                (in_height + 2*pad[0] - dil[0]*(ksize[0]-1) - 1) / stride[0] + 1
            )
            w_out = math.floor(
                (in_width  + 2*pad[1] - dil[1]*(ksize[1]-1) - 1) / stride[1] + 1
            )

            # for grouped conv:
            #   weight.shape = (out_channels, in_channels_//groups_, kH, kW)
            in_ch_per_group = in_channels_ // groups_
            out_ch_per_group = out_channels // groups_

            idx_current = 0  # index into current_layer_neurons
            for oc in range(out_channels):
                # find which group oc belongs to
                group_idx = oc // out_ch_per_group
                # global input channels that belong to this group => [in_c_start, in_c_start+in_ch_per_group)
                in_c_start = group_idx * in_ch_per_group

                for oh in range(h_out):
                    for ow in range(w_out):
                        tgt_id = current_layer_neurons[idx_current]
                        idx_current += 1

                        # Receptive field
                        for kh in range(ksize[0]):
                            for kw in range(ksize[1]):
                                ih = oh*stride[0] + kh*dil[0] - pad[0]
                                iw = ow*stride[1] + kw*dil[1] - pad[1]
                                if (0 <= ih < in_height) and (0 <= iw < in_width):
                                    # local channels in [0, in_ch_per_group)
                                    for ic_local in range(in_ch_per_group):
                                        ic_global = in_c_start + ic_local
                                        src_idx = ic_global*in_height*in_width + ih*in_width + iw

                                        if src_idx >= len(prev_neurons):
                                            raise IndexError(
                                                f"[Conv2d] source idx {src_idx} out of range {len(prev_neurons)}"
                                            )
                                        s_id = prev_neurons[src_idx]
                                        # weight index = (oc, ic_local, kh, kw)
                                        w_val = layer.weight.data[oc, ic_local, kh, kw].item()
                                        if w_val != 0:
                                            connections.append((s_id, tgt_id, w_val))

            # update dimension for next layer
            in_channels = out_channels
            in_height = h_out
            in_width = w_out
            prev_neurons = current_layer_neurons

        # ----------------------------------------------------------------------
        # Handle Linear
        # ----------------------------------------------------------------------
        elif isinstance(layer, nn.Linear):
            out_features = layer.out_features
            in_features  = layer.in_features
            weight_matrix = layer.weight.data.numpy()

            for o_idx in range(out_features):
                for i_idx in range(in_features):
                    w_val = weight_matrix[o_idx, i_idx]
                    if w_val != 0:
                        if i_idx >= len(prev_neurons):
                            raise IndexError(
                                f"[Linear] i_idx={i_idx} >= len(prev_neurons)={len(prev_neurons)}"
                            )
                        s_id = prev_neurons[i_idx]
                        t_id = current_layer_neurons[o_idx]
                        connections.append((s_id, t_id, w_val))

            # after linear => flatten dimension => (out_features,1,1)
            in_channels = out_features
            in_height   = 1
            in_width    = 1
            prev_neurons= current_layer_neurons

        else:
            # should not happen if we only have conv, pool, linear
            pass

    return connections



# Define the GNC (General Neuron Container) class
class GNC:
    def __init__(self, gnc_id):
        """
        Initializes a GNC with a unique identifier.

        Parameters:
        - gnc_id: Integer representing the GNC number (0-15).
        """
        if not (0 <= gnc_id < 16):
            raise ValueError("GNC ID must be between 0 and 15.")
        self.id = gnc_id
        self.contains = set()

    def add(self, neuron_ids):
        """
        Adds neuron IDs to the GNC, ensuring no duplicates.

        Parameters:
        - neuron_ids: A single neuron ID (int) or a list of neuron IDs.
        """
        if isinstance(neuron_ids, list):
            duplicates = set(neuron_ids) & self.contains
            if duplicates:
                print(f"[GNC {self.id}] Duplicate neuron IDs {duplicates} already present.")
            self.contains.update(neuron_ids)
        else:
            if neuron_ids in self.contains:
                print(f"[GNC {self.id}] Duplicate neuron ID {neuron_ids} already present.")
            else:
                self.contains.add(neuron_ids)

    def get_contains(self):
        """
        Returns a sorted list of neuron IDs contained in the GNC.
        """
        return sorted(list(self.contains))

    def __str__(self):
        return f"GNC {self.id} contains neurons: {self.get_contains()}"


# Define the NFU (Neuron Function Unit) class
class NFU:
    def __init__(self, nfu_id):
        """
        Initializes an NFU with a unique identifier and 16 GNCs.

        Parameters:
        - nfu_id: Integer representing the NFU number.
        """
        self.id = nfu_id
        self.GNCs = {gnc_id: GNC(gnc_id) for gnc_id in range(16)}

    def get_GNC(self, gnc_id):
        """
        Retrieves a specific GNC within the NFU.

        Parameters:
        - gnc_id: Integer representing the GNC number (0-15).

        Returns:
        - GNC instance.
        """
        if gnc_id in self.GNCs:
            return self.GNCs[gnc_id]
        else:
            raise ValueError(f"GNC ID {gnc_id} is out of range (0-15).")

    def __str__(self):
        gnc_str = "\n  ".join([str(gnc) for gnc in self.GNCs.values()])
        return f"NFU {self.id} with GNCs:\n  {gnc_str}"


def save_plot():
    try:
        plt.savefig('gnc_mapping.png')
        print("[DEBUG] Saved plot to 'gnc_mapping.png'.")
    except Exception as e:
        print(f"[ERROR] Failed to save plot: {e}")

def plot_gnc_mapping(gnc_coordinates, gnc_usage):
    print("[DEBUG] Starting plot_gnc_mapping...")
    plot_start_time = time.time()

    fig, ax = plt.subplots(figsize=(8, 8))
    print("[DEBUG] Created matplotlib figure and axes.")

    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')  # 隐藏坐标轴
    print("[DEBUG] Configured plot axes.")

    # 定义颜色映射
    cmap = plt.get_cmap('Reds')  # 使用 plt.get_cmap 以避免弃用警告
    norm = Normalize(vmin=0, vmax=1)
    print("[DEBUG] Defined color map and normalization.")

    # 绘制 GNC 圆形
    for gnc_id, (x, y) in gnc_coordinates.items():
        usage = gnc_usage.get(gnc_id, 0)
        color = cmap(norm(usage))
        circle = patches.Circle((x, y), 0.4, facecolor=color, edgecolor='black')
        ax.add_patch(circle)
        print(f"[DEBUG] Drew GNC {gnc_id} at ({x}, {y}) with usage ratio {usage:.2f}.")

        # 添加 GNC ID 文本
        text_color = 'white' if usage > 0.5 else 'black'
        ax.text(x, y, str(gnc_id), horizontalalignment='center', verticalalignment='center',
                fontsize=12, color=text_color)
        print(f"[DEBUG] Added GNC ID text for GNC {gnc_id} with text color {text_color}.")

        # 添加使用百分比
        ax.text(x, y - 0.5, f"{usage * 100:.1f}%", horizontalalignment='center',
                verticalalignment='center', fontsize=10, color='black')
        print(f"[DEBUG] Added usage percentage text for GNC {gnc_id}.")

    print("[DEBUG] Finished drawing GNC circles and texts.")

    # 添加标题
    ax.set_title('GNC Mapping Visualization', fontsize=16)
    print("[DEBUG] Added plot title.")

    # 添加颜色条
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Usage Ratio')
    print("[DEBUG] Added color bar.")

    # 在主线程中保存图像
    save_plot()

    plot_end_time = time.time()
    print(f"[DEBUG] plot_gnc_mapping executed in {plot_end_time - plot_start_time:.2f} seconds.")


# ---------- 绘图工具-带文件名 ----------
def save_plot_filename(filename: str):
    try:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"[DEBUG] Saved plot to '{filename}'.")
    except Exception as e:
        print(f"[ERROR] Failed to save plot '{filename}': {e}")

def plot_gnc_mapping_filname(gnc_coordinates, gnc_usage, filename):
    print(f"[DEBUG] plot_gnc_mapping for {filename} ...")
    t0 = time.time()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    cmap = plt.get_cmap('Reds')
    norm = Normalize(vmin=0, vmax=1)

    for gid, (x, y) in gnc_coordinates.items():
        usage = gnc_usage.get(gid, 0)
        color = cmap(norm(usage))
        circle = patches.Circle((x, y), 0.4, facecolor=color, edgecolor='black')
        ax.add_patch(circle)

        txt_color = 'white' if usage > 0.5 else 'black'
        ax.text(x, y, str(gid), ha='center', va='center', fontsize=12, color=txt_color)
        ax.text(x, y - 0.5, f"{usage*100:.1f}%", ha='center', va='center', fontsize=9)

    ax.set_title('GNC Mapping (Usage)', fontsize=14)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Usage Ratio')

    save_plot_filename(filename)
    plt.close(fig)
    print(f"[DEBUG] plotting done in {time.time()-t0:.2f}s")


def cluster(neuron_id_map, connections, relation=1):
    """
    Maps neurons to the NFU's GNCs based on their connections and the defined 4x4 mesh.
    Now enforces strict 'by-layer then by-GNC' placement:
      - strictly layer-by-layer (iterate layer_order in order);
      - strictly by-GNC: as long as the currently opened GNC has free capacity, never open a new one.

    Parameters:
    - neuron_id_map: {layer_name: [neuron IDs]}
    - connections: [(source_id, target_id, weight)]
    - relation: how many ancestor generations to consider when measuring "closeness" (for picking
                the *next* GNC to open only; does NOT override strict-by-GNC rule)

    Returns:
    - input_mapping: {input_neuron_id: gnc_id}
    - output_mapping:{output_neuron_id: gnc_id}
    - nfu: NFU instance
    - longest_time_expression: str
    """
    print("[DEBUG] Starting cluster function...")
    start_time = time.time()

    # GNC coordinates (unchanged)
    gnc_coordinates = {
        0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
        4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
        8:(3,1), 9:(2,1), 10:(1,1), 11:(0,1),
        12:(3,0), 13:(2,0), 14:(1,0), 15:(0,0),
    }
    CAP = 8192  # 单个 GNC 容量（保持与原实现一致）

    # --- NFU / 状态集 ---
    nfu = NFU(nfu_id=0)
    full_gncs = set()      # 已满的 GNC
    opened_gncs = set()    # 已启用（出现过放置）的 GNC
    active_gnc = None      # 严格逐GNC：当前唯一可放置且未满的 GNC

    # --- 反向边映射：target -> [sources] ---
    target_to_sources = defaultdict(list)
    for src, tgt, _w in connections:
        target_to_sources[tgt].append(src)

    # 祖先收集（至多 relation 代）
    def get_ancestors(node, max_relation):
        visited, q, anc = set(), [(node, 0)], set()
        while q:
            cur, d = q.pop(0)
            if d >= max_relation:
                continue
            for p in target_to_sources.get(cur, []):
                if p not in visited:
                    visited.add(p)
                    anc.add(p)
                    q.append((p, d+1))
        return anc

    # 曼哈顿距离
    def manhattan_distance(gnc1, gnc2):
        x1, y1 = gnc_coordinates[gnc1]
        x2, y2 = gnc_coordinates[gnc2]
        return abs(x1 - x2) + abs(y1 - y2)

    print("[DEBUG] Defined manhattan_distance & get_ancestors.")

    # 选“下一个要启用的新 GNC”（只有当 active_gnc 已满才会被调用）
    def pick_new_gnc(parent_gncs):
        # 优先选尚未启用且未满的 GNC；若都已启用，则在未满 GNC 中选一个
        candidates = [g for g in range(16) if g not in full_gncs and g not in opened_gncs]
        if not candidates:
            candidates = [g for g in range(16) if g not in full_gncs]
        if not candidates:
            return None

        if parent_gncs:
            def score(g):
                return min(manhattan_distance(g, pg) for pg in parent_gncs)
            return min(candidates, key=lambda g: (score(g), g))  # 最近 + 编号小
        else:
            return min(candidates)  # 无父辈：取最小编号，保持稳定性

    # --- 层序 ---
    layer_order = list(neuron_id_map.keys())
    if 'input' in neuron_id_map:
        layer_order = ['input'] + [k for k in layer_order if k != 'input']
    if 'output' in neuron_id_map:
        layer_order = [k for k in layer_order if k != 'output'] + ['output']

    # 记录：神经元 -> GNC
    neuron_to_gnc = {}

    print("Assigning neurons with strict layer-by-layer AND by-GNC policy...")

    # 严格逐层 + 严格逐GNC：只要 active_gnc 未满，就只能放到 active_gnc；
    # 只有当 active_gnc 满了，才会 pick_new_gnc(...) 启用下一个。
    for layer_idx in tqdm(range(0, len(layer_order)), desc="Layers", unit="layer"):
        layer_name = layer_order[layer_idx]
        layer_neurons = neuron_id_map[layer_name]

        for neuron in tqdm(layer_neurons, desc=f"Layer {layer_name}", leave=False, unit="neuron"):
            # 收父辈所在 GNC（用于“新开”时的选址参考；不会打破逐GNC）
            ancestors = get_ancestors(neuron, relation)
            parent_gncs = {neuron_to_gnc[a] for a in ancestors if a in neuron_to_gnc}

            # 若当前没有活动 GNC 或者已满 -> 必须“新开一个”
            if active_gnc is None or len(nfu.get_GNC(active_gnc).contains) >= CAP:
                ng = pick_new_gnc(parent_gncs)
                if ng is None:
                    print("NFU 0 cannot accommodate all neurons. Need more NFUs.")
                    return None, None, None, None
                active_gnc = ng
                opened_gncs.add(active_gnc)
                # 不在此处加入 full_gncs；放完再判断

            # —— 严格逐GNC：只允许放在 active_gnc —— #
            nfu.get_GNC(active_gnc).add(neuron)
            neuron_to_gnc[neuron] = active_gnc

            # 满了就记满；下一次循环会触发“新开”
            if len(nfu.get_GNC(active_gnc).contains) >= CAP:
                full_gncs.add(active_gnc)
                # 不立即切换，下一个神经元/下一个层会触发新开逻辑

    print("[DEBUG] Finished assigning neurons to all layers.")

    # 5) Input mapping
    input_neurons = neuron_id_map.get('input', [])
    input_mapping = {n: neuron_to_gnc.get(n, None) for n in input_neurons}
    print("[DEBUG] Created input_mapping.")

    # 6) Output mapping
    output_neurons = neuron_id_map.get('output', [])
    output_mapping = {n: neuron_to_gnc.get(n, None) for n in output_neurons}
    print("[DEBUG] Created output_mapping.")

    # 7) 可视化 GNC 使用率
    print("Visualizing GNC mapping...")
    gnc_usage = {}
    for gnc_id in range(16):
        count = len(nfu.get_GNC(gnc_id).contains)
        usage_ratio = min(count / CAP, 1.0)
        gnc_usage[gnc_id] = usage_ratio
        print(f"[DEBUG] GNC {gnc_id}: {count} neurons, usage ratio {usage_ratio:.2f}")
    print("[DEBUG] Finished calculating GNC usage ratios.")
    plot_gnc_mapping(gnc_coordinates, gnc_usage)

    # 8) 最长耗时路径（GNC 级）：严格逐GNC应使其为 DAG
    print("[DEBUG] Starting to compute the longest time path.")
    G = nx.DiGraph()
    for src_neuron, tgt_neuron, _weight in connections:
        src_gnc = neuron_to_gnc.get(src_neuron)
        tgt_gnc = neuron_to_gnc.get(tgt_neuron)
        if src_gnc is None or tgt_gnc is None:
            continue
        if src_gnc == tgt_gnc:
            continue  # 0-hop 内部连接不作为跨核边
        G.add_edge(src_gnc, tgt_gnc)

    print("[DEBUG] Built GNC connection graph.")
    if not nx.is_directed_acyclic_graph(G):
        print("[WARNING] The GNC connection graph contains cycles. Longest path is undefined.")
        # 可选：打印 2-cycle 诊断，便于排查（保持最小改动，这里只给提示）
        longest_time_expression = "Undefined (Graph contains cycles)"
    else:
        try:
            # 若希望以 hop 为权重：nx.dag_longest_path(G, weight=None) 已是“边数”，
            # 你也可以在边属性里放 hop 并用 weight='hop'。
            longest_path = nx.dag_longest_path(G)
            print(f"[DEBUG] Found the longest path: {longest_path}")
            num_hops = len(longest_path) - 1
            num_gncs = len(longest_path)
            longest_time_expression = f"Longest_Time = ({num_hops} * T0) + ({num_gncs} * T1) + T2"
            print(f"[DEBUG] Longest time expression: {longest_time_expression}")
        except Exception as e:
            print(f"[ERROR] Failed to compute the longest path: {e}")
            longest_time_expression = "Error in computing the longest path"

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Clustering completed in {elapsed_time:.2f} seconds.")

    return input_mapping, output_mapping, nfu, longest_time_expression



def cluster_input2center(neuron_id_map, connections, relation=1):
    """
    Maps neurons to the NFU's GNCs based on their connections and the defined 4x4 mesh.
    This variant assigns input neurons to GNCs in the order [13, 14, 10, 9].

    Parameters:
    - neuron_id_map: Dictionary mapping layer names to lists of neuron IDs.
    - connections: List of tuples representing connections (source_id, target_id, weight).
    - relation: Integer representing how many generations of ancestors we consider.
      relation=1 -> immediate parents,
      relation=2 -> parents + grandparents,
      relation=3 -> parents + grandparents + great-grandparents, etc.

    Returns:
    - input_mapping: Dictionary mapping input neuron IDs to their respective GNC numbers.
    - output_mapping: Dictionary mapping output neuron IDs to their respective GNC numbers.
    - nfu: The single NFU instance used.
    """

    # GNC coordinates (corrected)
    gnc_coordinates = {
        0: (3, 3),  # Right Top
        1: (2, 3),
        2: (1, 3),
        3: (0, 3),  # Left Top
        4: (3, 2),
        5: (2, 2),
        6: (1, 2),
        7: (0, 2),
        8: (3, 1),
        9: (2, 1),
        10: (1, 1),
        11: (0, 1),
        12: (3, 0),  # Right Bottom
        13: (2, 0),
        14: (1, 0),
        15: (0, 0)  # Left Bottom
    }

    # Create an NFU
    nfu = NFU(nfu_id=0)

    # Which GNCs are full?
    full_gncs = set()

    # Build target->sources mapping
    target_to_sources = defaultdict(list)
    for src, tgt, _w in connections:
        target_to_sources[tgt].append(src)

    # Helper: BFS up to `relation` steps to find all ancestors of a node
    def get_ancestors(node, max_relation):
        """
        Returns the set of all ancestor neurons of `node` up to `max_relation` generations.
        For example, if max_relation=2, we gather parents (1st gen) + grandparents (2nd gen).
        """
        visited = set()
        queue = [(node, 0)]  # (current_node, current_depth)
        ancestors = set()

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_relation:
                continue
            # direct parents of curr
            parents = target_to_sources.get(curr, [])
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    ancestors.add(p)
                    # enqueue this parent to explore further, up to max_relation
                    queue.append((p, depth + 1))
        return ancestors

    # Manhattan distance
    def manhattan_distance(gnc1, gnc2):
        x1, y1 = gnc_coordinates[gnc1]
        x2, y2 = gnc_coordinates[gnc2]
        return abs(x1 - x2) + abs(y1 - y2)

    # Sort GNCs by min distance to a set of parent_gncs
    def get_sorted_gncs(parent_gncs):
        distance_dict = {}
        for gnc in range(16):
            if gnc in full_gncs:
                continue
            # find min dist from this gnc to any parent's GNC
            min_dist = min(manhattan_distance(gnc, pg) for pg in parent_gncs)
            distance_dict[gnc] = min_dist
        # sort by (distance, gnc_id)
        return sorted(distance_dict.keys(), key=lambda g: (distance_dict[g], g))

    # 1) Assign input neurons to GNCs [13, 14, 10, 9] in order
    input_neurons = neuron_id_map.get('input', [])
    input_gncs_order = [13, 14, 10, 9]
    current_gnc_input_idx = 0

    for neuron in input_neurons:
        assigned = False
        while current_gnc_input_idx < len(input_gncs_order):
            current_gnc = input_gncs_order[current_gnc_input_idx]
            if current_gnc in full_gncs:
                current_gnc_input_idx += 1
                continue
            # Assign neuron to this GNC
            nfu.get_GNC(current_gnc).add(neuron)
            # Update neuron_to_gnc mapping later
            # Check capacity
            if len(nfu.get_GNC(current_gnc).contains) >= 8192:
                full_gncs.add(current_gnc)
                current_gnc_input_idx += 1
            assigned = True
            break
        if not assigned:
            print("NFU 0 cannot accommodate all input neurons. Need more NFUs.")
            return None, None, None

    # 2) Build a dictionary: neuron -> GNC
    neuron_to_gnc = {}
    for gnc_id in range(16):
        for neuron in nfu.get_GNC(gnc_id).contains:
            neuron_to_gnc[neuron] = gnc_id

    # 3) Layer order
    layer_order = list(neuron_id_map.keys())  # assumed from input->...->output

    # 4) Assign neurons in subsequent layers
    for layer_idx in range(1, len(layer_order)):
        layer_name = layer_order[layer_idx]
        layer_neurons = neuron_id_map[layer_name]

        for neuron in layer_neurons:
            # Gather up to `relation` steps of ancestors
            ancestors = get_ancestors(neuron, relation)
            # Find which GNCs these ancestors belong to
            parent_gncs = set()
            for anc in ancestors:
                if anc in neuron_to_gnc:
                    parent_gncs.add(neuron_to_gnc[anc])

            # Special case: if we found no ancestors => just pick the smallest ID GNC not full
            if not parent_gncs:
                available_gncs = set(range(16)) - full_gncs
                if not available_gncs:
                    print("NFU 0 cannot accommodate all neurons. Need more NFUs.")
                    return None, None, None
                # Pick the smallest ID
                assigned_gnc = min(available_gncs)
                nfu.get_GNC(assigned_gnc).add(neuron)
                neuron_to_gnc[neuron] = assigned_gnc
                if len(nfu.get_GNC(assigned_gnc).contains) >= 8192:
                    full_gncs.add(assigned_gnc)
                continue

            # Otherwise, sort GNC by distance to these parent GNCs
            candidate_gncs = get_sorted_gncs(parent_gncs)
            # Place in the first available
            for cgnc in candidate_gncs:
                if cgnc in full_gncs:
                    continue
                nfu.get_GNC(cgnc).add(neuron)
                neuron_to_gnc[neuron] = cgnc
                if len(nfu.get_GNC(cgnc).contains) >= 8192:
                    full_gncs.add(cgnc)
                break
            else:
                print("NFU 0 cannot accommodate all neurons. Need more NFUs.")
                return None, None, None

    # 5) Input mapping
    input_mapping = {n: neuron_to_gnc.get(n, None) for n in input_neurons}
    # 6) Output mapping
    output_neurons = neuron_id_map.get('output', [])
    output_mapping = {n: neuron_to_gnc.get(n, None) for n in output_neurons}
    return input_mapping, output_mapping, nfu


def strict_layer_gnc_greedy_assign(neuron_id_map, connections, relation=1, *, CAP=8192, GNC_COUNT=16, gnc_coordinates=None):
    """
    严格逐层 + 严格逐GNC（未满绝不新开）的基解：
      - 按 layer_order 顺序遍历所有神经元；
      - 只要当前 active_gnc 未满，就必须继续放到 active_gnc；
      - 只有当 active_gnc 满了，才“启用”下一个 GNC；
      - 启用新 GNC 时，按“距父辈最近（曼哈顿），平局取编号小”的准则选址。
    返回:
      assign_dict: {neuron_id -> gnc_id}
      nfu: NFU 实例（每个 GNC 的 contains 已填充）
    """

    if gnc_coordinates is None:
        gnc_coordinates = globals().get('gnc_coordinates') or {
            0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
            4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
            8:(3,1), 9:(2,1),10:(1,1),11:(0,1),
           12:(3,0),13:(2,0),14:(1,0),15:(0,0),
        }

    def manhattan(g1, g2):
        (x1,y1),(x2,y2) = gnc_coordinates[g1], gnc_coordinates[g2]
        return abs(x1-x2)+abs(y1-y2)

    # target -> [sources]（祖先索引）
    tgt2src = defaultdict(list)
    for s,t,_ in connections:
        tgt2src[t].append(s)

    def get_ancestors(n, k):
        if k <= 0: return set()
        seen, q, acc = set(), [(n,0)], set()
        while q:
            cur,d = q.pop(0)
            if d >= k: continue
            for p in tgt2src.get(cur, []):
                if p not in seen:
                    seen.add(p); acc.add(p); q.append((p, d+1))
        return acc

    # 层序（尽量维持你原口径：input在最前，output在最后）
    layer_order = list(neuron_id_map.keys())
    if 'input'  in neuron_id_map: layer_order = ['input'] + [k for k in layer_order if k!='input']
    if 'output' in neuron_id_map: layer_order = [k for k in layer_order if k!='output'] + ['output']

    nfu = NFU(0)
    full_gncs   = set()
    opened_gncs = set()
    active_gnc  = None
    assign_dict = {}

    def pick_new_gnc(parent_gncs):
        # 优先未启用且未满；其次任一未满
        cand = [g for g in range(GNC_COUNT) if g not in full_gncs and g not in opened_gncs]
        if not cand:
            cand = [g for g in range(GNC_COUNT) if g not in full_gncs]
        if not cand:
            return None
        if parent_gncs:
            def score(g):
                return min(manhattan(g, pg) for pg in parent_gncs)
            return min(cand, key=lambda g:(score(g), g))
        return min(cand)

    # 严格逐层 + 严格逐GNC 放置
    for layer in layer_order:
        for neu in neuron_id_map[layer]:
            parents = get_ancestors(neu, relation)
            parent_gncs = {assign_dict[a] for a in parents if a in assign_dict}

            if active_gnc is None or len(nfu.get_GNC(active_gnc).contains) >= CAP:
                ng = pick_new_gnc(parent_gncs)
                if ng is None:
                    raise RuntimeError("No free GNC available under strict greedy policy.")
                active_gnc = ng
                opened_gncs.add(active_gnc)

            nfu.get_GNC(active_gnc).add(neu)
            assign_dict[neu] = active_gnc

            if len(nfu.get_GNC(active_gnc).contains) >= CAP:
                full_gncs.add(active_gnc)

    return assign_dict, nfu


def cluster_sa(neuron_id_map,
               connections,
               relation=1,
               cap=8192,
               max_iter=200,
               T_init=5.0,
               T_final=0.01,
               w_usage=1.0, w_edge=0.5, w_dist=0.1,
               w_ub=2.0, w_cut=1.5,              # 新增：UB与per-link cut权重
               retry_base=10):
    """
    SA（改进版）：
      - 基解：严格“逐层 + 逐GNC”的 greedy（strict_layer_gnc_greedy_assign）
      - 目标：原 cost + 归一化 UB（SCD-D_UB） + 归一化 per-link cut max
      - 约束：容量 / DAG + 硬门槛（UB不得比基解高>2；per-link不得高于基解10%）
      - 邻域：swap 或 relocate-to-topK-parent-near GNC（K=3）
    返回: input_mapping, output_mapping, nfu, longest_time_expression
    """
    import random, math, time
    from collections import defaultdict
    import networkx as nx
    from tqdm import tqdm

    # -------------- 网格坐标 & 基础工具 --------------
    GNC_COORD = {0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
                 4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
                 8:(3,1), 9:(2,1),10:(1,1),11:(0,1),
                12:(3,0),13:(2,0),14:(1,0),15:(0,0)}
    COLS = ROWS = 4
    def md(g1,g2):
        (x1,y1),(x2,y2)=GNC_COORD[g1],GNC_COORD[g2]
        return abs(x1-x2)+abs(y1-y2)

    # 神经元 -> 父辈索引（至多 relation 代）
    tgt2src = defaultdict(list)
    for s,t,_ in connections: tgt2src[t].append(s)
    def ancestors(n, k):
        if k<=0: return set()
        seen, q, acc = set(), [(n,0)], set()
        while q:
            cur,d = q.pop(0)
            if d>=k: continue
            for p in tgt2src.get(cur, []):
                if p not in seen:
                    seen.add(p); acc.add(p); q.append((p,d+1))
        return acc

    # 构 GNC 级图（边属性 hop = 曼哈顿距离；用于UB）
    def build_gnc_graph(assign):
        G = nx.DiGraph()
        for s,t,_ in connections:
            gs, gt = assign[s], assign[t]
            if gs != gt:
                G.add_edge(gs, gt, hop=md(gs,gt))
        return G

    # UB：DAG 上按 hop 加权的最长路（SCD-D_UB）
    def compute_ub(assign):
        G = build_gnc_graph(assign)
        if not nx.is_directed_acyclic_graph(G):
            return None  # 有环
        # networkx 的最长路径在有权图上是求“权重和最大的路径”
        path = nx.dag_longest_path(G, weight="hop")
        if not path or len(path)==1:
            return 0.0
        # 计算加权长度
        w = 0.0
        for u,v in zip(path[:-1], path[1:]):
            w += G[u][v].get("hop", 1.0)
        return float(w)

    # per-link cut（近似瓶颈）：统计每条“列间切割 / 行间切割”的穿越总量
    # 对于任意跨GNC边 (x1,y1)->(x2,y2)：
    #   它会穿越 |x1-x2| 个竖向 cut & |y1-y2| 个横向 cut。
    # 每个竖 cut 的单链路数=ROWS；每个横 cut 的单链路数=COLS。
    def compute_perlink_max(assign):
        vcuts = [0,0,0]  # between col 0-1, 1-2, 2-3
        hcuts = [0,0,0]  # between row 0-1, 1-2, 2-3
        for s,t,_ in connections:
            gs, gt = assign[s], assign[t]
            if gs == gt: continue
            (x1,y1) = GNC_COORD[gs]; (x2,y2) = GNC_COORD[gt]
            lo, hi = sorted((x1,x2))
            for c in range(lo, hi):   # c in {0,1,2}
                vcuts[c] += 1
            lo, hi = sorted((y1,y2))
            for r in range(lo, hi):
                hcuts[r] += 1
        v_perlink_max = max(vcuts)/ROWS if vcuts else 0.0
        h_perlink_max = max(hcuts)/COLS if hcuts else 0.0
        return max(v_perlink_max, h_perlink_max), v_perlink_max, h_perlink_max

    # -------------- 基解（严格逐层＋逐GNC） --------------
    base_assign = None
    for _ in range(retry_base):
        try:
            ba, nfu_tmp = strict_layer_gnc_greedy_assign(
                neuron_id_map, connections, relation=relation, CAP=cap, GNC_COUNT=16
            )
        except Exception as e:
            print(f"[GDV] strict-greedy failed: {e}")
            ba, nfu_tmp = None, None

        if ba is None:
            continue
        G0 = build_gnc_graph(ba)
        if nx.is_directed_acyclic_graph(G0):
            base_assign, nfu = ba, nfu_tmp
            break

    if base_assign is None:
        print("[GDV] Greedy mapping results in cycles for all retries.")
        return None, None, None, "Undefined (GDV)"

    # -------------- 基线指标（用于归一化与硬门槛） --------------
    ub0 = compute_ub(base_assign)
    cut0, vpl0, hpl0 = compute_perlink_max(base_assign)

    # 原三项（imb/e/dist）也取基线用于归一化
    def basic_terms(assign):
        usage=[0]*16
        for g in assign.values(): usage[g]+=1
        imb = math.sqrt(sum(((u/cap) - (sum(usage)/16/cap))**2 for u in usage)/16)
        G = build_gnc_graph(assign)
        e = len(G.edges())
        dist = sum(md(u,v) for u,v in G.edges())
        return imb, e, dist, G

    imb0, e0, dist0, _ = basic_terms(base_assign)
    # 防止除零
    def nz(x): return x if x>0 else 1.0

    # -------------- 归一化代价函数 + 硬门槛 --------------
    def cost(assign):
        imb, e, dist, G = basic_terms(assign)
        if not nx.is_directed_acyclic_graph(G):
            return 1e12
        # 容量硬约束
        usage=[0]*16
        for g in assign.values(): usage[g]+=1
        if any(u>cap for u in usage): return 1e12

        ub = compute_ub(assign)
        if ub is None:  # 有环
            return 1e12

        cut, _, _ = compute_perlink_max(assign)

        # 硬门槛（守住最坏时延与瓶颈）
        if ub > ub0 + 2:          # 最长路不得比基线高出 2 hops
            return 1e11 + (ub-ub0)*1e6
        if cut > 1.10*cut0:       # 单链路最大穿越量不得高于基线 10%
            return 1e11 + (cut-cut0)*1e4

        # 归一化组合：各项 ~ O(1) 量级
        c = (
            w_usage * (imb / nz(imb0)) +
            w_edge  * (e   / nz(e0))   +
            w_dist  * (dist/ nz(dist0))+
            w_ub    * (ub  / nz(ub0))  +
            w_cut   * (cut / nz(cut0))
        )
        return c

    # -------------- SA 初始化 --------------
    best = current = dict(base_assign)
    best_c = cur_c = cost(current)
    T = T_init
    neurons = list(current.keys())

    # -------------- 邻域操作 --------------
    def relocate_candidate_gncs(n):
        # 取父辈GNC集合，按距离排序挑前K个
        K = 3
        pgs = {current[a] for a in ancestors(n, relation) if a in current}
        if not pgs:
            # 无父辈：全局按编号从小到大挑前K
            return list(range(16))[:K]
        order = sorted(range(16), key=lambda g: (min(md(g,pg) for pg in pgs), g))
        return order[:K]

    def try_move(n, dst):
        """尝试把 n 放到 dst；返回 (accepted, prev_dst)"""
        src = current[n]
        if src == dst: return False, src
        # 容量约束
        if sum(1 for g in current.values() if g==dst) >= cap:
            return False, src
        current[n] = dst
        # DAG 快检
        if not nx.is_directed_acyclic_graph(build_gnc_graph(current)):
            current[n] = src
            return False, src
        new_c = cost(current)
        d = new_c - cur_c
        accept = (d < 0) or (random.random() < math.exp(-d / T))
        if accept:
            return True, src
        else:
            current[n] = src
            return False, src

    def try_swap(n1, n2):
        """交换两个不同GNC上的神经元"""
        g1, g2 = current[n1], current[n2]
        if g1 == g2: return False
        current[n1], current[n2] = g2, g1
        if not nx.is_directed_acyclic_graph(build_gnc_graph(current)):
            current[n1], current[n2] = g1, g2
            return False
        new_c = cost(current)
        d = new_c - cur_c
        if (d < 0) or (random.random() < math.exp(-d / T)):
            return True
        current[n1], current[n2] = g1, g2
        return False

    # -------------- 模拟退火主循环 --------------
    for _ in tqdm(range(max_iter), "SA"):
        if random.random() < 0.5:
            # relocate：往“父辈最近”的前K个核之一尝试
            n = random.choice(neurons)
            cand = relocate_candidate_gncs(n)
            dst = random.choice(cand)
            ok, src = try_move(n, dst)
            if ok:
                cur_c = cost(current)
                if cur_c < best_c:
                    best_c, best = cur_c, dict(current)
        else:
            # swap：更利于维持均衡且不激进
            n1, n2 = random.sample(neurons, 2)
            if try_swap(n1, n2):
                cur_c = cost(current)
                if cur_c < best_c:
                    best_c, best = cur_c, dict(current)

        # 降温
        T *= (T_final / T_init) ** (1 / max_iter)
        if T < T_final: break

    # -------------- 结束校验 & 输出 --------------
    if not nx.is_directed_acyclic_graph(build_gnc_graph(best)):
        print("[ERROR] SA finished with cycle - unexpected")
        return None, None, None, "Undefined"

    # 重建 NFU
    nfu = NFU(0)
    for g in range(16): nfu.get_GNC(g).contains.clear()
    for n,g in best.items(): nfu.get_GNC(g).add(n)

    # I/O 映射
    input_mapping  = {n:best[n] for n in neuron_id_map.get('input', [])}
    output_mapping = {n:best[n] for n in neuron_id_map.get('output', [])}

    # 最长路表达式（沿用你原有格式）
    ub_final = compute_ub(best)
    # 估算“路径中的节点数”=（hop总和/平均每边hop≈1）+ 1，这里仍按边数估；保留原表达口径：
    G_final = build_gnc_graph(best)
    longest_path_nodes = nx.dag_longest_path(G_final) if nx.is_directed_acyclic_graph(G_final) else []
    num_hops = len(longest_path_nodes) - 1
    num_gncs = len(longest_path_nodes)
    longest_time_expression = f"Longest_Time = ({num_hops} * T0) + ({num_gncs} * T1) + T2"

    # 可选：打印一下改进后的关键指标
    cut_final, vpl, hpl = compute_perlink_max(best)
    print(f"[SA+] base UB={ub0:.3f} -> final UB={ub_final:.3f} | base perlink={cut0:.2f} -> final {cut_final:.2f}")

    return input_mapping, output_mapping, nfu, longest_time_expression


def cluster_pso(neuron_id_map,
                connections,
                relation       = 1,
                cap            = 8192,
                pop_size       = 20,
                max_iter       = 200000,
                w_usage        = 1.0,
                w_edge         = 0.5,
                w_dist         = 0.1,
                retry_base     = 10):
    """
    PSO：以“严格逐层＋严格逐GNC”的基解为初始 gbest/pbest
    返回: input_mapping, output_mapping, nfu, longest_time_expression
    """

    GNC_COORD = {0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
                 4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
                 8:(3,1), 9:(2,1),10:(1,1),11:(0,1),
                12:(3,0),13:(2,0),14:(1,0),15:(0,0)}
    def md(g1,g2):
        (x1,y1),(x2,y2)=GNC_COORD[g1],GNC_COORD[g2]
        return abs(x1-x2)+abs(y1-y2)

    def build_graph(assign_dict):
        G=nx.DiGraph()
        for s,t,_ in connections:
            gs,gt=assign_dict[s],assign_dict[t]
            if gs!=gt: G.add_edge(gs,gt)
        return G

    # ---------- 基解 ----------
    base = None
    for _ in range(retry_base):
        try:
            ba, base_nfu = strict_layer_gnc_greedy_assign(
                neuron_id_map, connections, relation=relation, CAP=cap, GNC_COUNT=16
            )
        except Exception as e:
            print(f"[GDV] strict-greedy failed: {e}")
            ba, base_nfu = None, None

        if ba and nx.is_directed_acyclic_graph(build_graph(ba)):
            base, nfu = ba, base_nfu
            break
    if base is None:
        print("[GDV] greedy stage fails")
        return None, None, None, "Undefined"

    # ---------- 数据准备 ----------
    all_neurons = list(base.keys())
    idx = {n:i for i,n in enumerate(all_neurons)}
    total = len(all_neurons)

    def assignment_to_list(a_dict):
        lst=[0]*total
        for n,g in a_dict.items(): lst[idx[n]]=g
        return lst

    base_list = assignment_to_list(base)

    # ---------- 代价函数 ----------
    def cost(lst):
        usage=[0]*16
        for g in lst: usage[g]+=1
        mean=sum(usage)/16/cap
        imb=math.sqrt(sum(((u/cap)-mean)**2 for u in usage)/16)
        G=build_graph({n:lst[idx[n]] for n in all_neurons})
        if not nx.is_directed_acyclic_graph(G): return 1e9
        e=len(G.edges())
        dist=sum(md(u,v) for u,v in G.edges())
        for u in usage:
            if u>cap: return 1e9
        return w_usage*imb + w_edge*e + w_dist*dist

    # ---------- 初始化粒子 ----------
    pop=[]
    for _ in range(pop_size):
        pos=base_list[:]  # 以基解为起点
        random.shuffle(pos)  # 也可少量扰动
        c=cost(pos)
        pop.append({'position':pos,'pbest_position':pos[:],'pbest_cost':c})
    g_best_pos=min(pop, key=lambda p: p['pbest_cost'])['pbest_position'][:]
    g_best_cost=min(p['pbest_cost'] for p in pop)

    # ---------- PSO 主循环（保持你原来的离散式做法） ----------
    start_vel, end_vel = 5, 1
    for it in tqdm(range(max_iter), "PSO"):
        cur_vel = max(int(start_vel - (start_vel-end_vel)*it/max_iter), 1)
        for p in pop:
            new = p['position'][:]
            for _ in range(cur_vel):
                i = random.randrange(total)
                # 参考 pbest / gbest
                if random.random()<0.3: new[i] = p['pbest_position'][i]
                if random.random()<0.3: new[i] = g_best_pos[i]
                # 少量随机探索
                if random.random()<0.02: new[i] = random.randint(0,15)
            c = cost(new)
            if c < p['pbest_cost']:
                p['pbest_cost']=c
                p['pbest_position']=new[:]
                if c < g_best_cost:
                    g_best_cost=c; g_best_pos=new[:]
            p['position']=new

    print(f"[INFO] PSO done | best_cost={g_best_cost:.4f}")

    # ---------- 输出映射 ----------
    nfu = NFU(0)
    for g in range(16): nfu.get_GNC(g).contains.clear()
    final={n:g_best_pos[idx[n]] for n in all_neurons}
    for n,g in final.items(): nfu.get_GNC(g).add(n)

    input_mapping  ={n:final[n] for n in neuron_id_map.get('input',[])}
    output_mapping ={n:final[n] for n in neuron_id_map.get('output',[])}

    path = nx.dag_longest_path(build_graph(final))
    hops, lenp = len(path)-1, len(path)
    expr = f"Longest_Time = ({hops} * T0) + ({lenp} * T1) + T2"
    return input_mapping, output_mapping, nfu, expr


def cluster_pacman(neuron_id_map, connections):
    print("[DEBUG] Starting cluster_pacman function (PACMAN).")
    start_time = time.time()
    nfu = NFU(nfu_id=0)
    # 定义4x4 GNC网格坐标和每核容量
    gnc_coordinates = {
        0: (3, 3), 1: (2, 3), 2: (1, 3), 3: (0, 3),
        4: (3, 2), 5: (2, 2), 6: (1, 2), 7: (0, 2),
        8: (3, 1), 9: (2, 1), 10: (1, 1), 11: (0, 1),
        12: (3, 0), 13: (2, 0), 14: (1, 0), 15: (0, 0)
    }
    GNC_CAPACITY = 8192
    capacity = GNC_CAPACITY

    # 获取神经元组顺序（假定 neuron_id_map 键已按输入->...->输出 顺序）
    layer_order = list(neuron_id_map.keys())
    # 确保 'input' 最先, 'output' 最后
    if 'input' in layer_order:
        layer_order.insert(0, layer_order.pop(layer_order.index('input')))
    if 'output' in layer_order:
        layer_order.append(layer_order.pop(layer_order.index('output')))
    layer_order_unique = []
    for name in layer_order:
        if name not in layer_order_unique:
            layer_order_unique.append(name)
    layer_order = layer_order_unique

    current_core = 0
    neuron_to_gnc = {}
    # 按组顺序分配神经元到核心
    for group_name in layer_order:
        neurons = neuron_id_map.get(group_name, [])
        if not neurons:
            continue
        idx = 0
        # 为该组的每个神经元分配核心，必要时跨多个核心
        while idx < len(neurons):
            if current_core >= 16:
                print("NFU 0 cannot accommodate all neurons. Need more NFUs.")
                return None, None, None, None
            # 如果当前核心已有其他组的神经元，则换到下一个核心（不混合不同组）
            if len(nfu.get_GNC(current_core).contains) != 0:
                current_core += 1
                continue
            space = capacity - len(nfu.get_GNC(current_core).contains)
            if space <= 0:  # 当前核心满，尝试下一个核心
                current_core += 1
                continue
            # 将本组尽可能多的神经元放入当前核心
            assign_count = min(space, len(neurons) - idx)
            batch = neurons[idx: idx + assign_count]
            nfu.get_GNC(current_core).add(batch)
            for neuron in batch:
                neuron_to_gnc[neuron] = current_core
            idx += assign_count
            # 如果核心填满或该组已分配完，则切换到下一个核心
            if len(nfu.get_GNC(current_core).contains) >= capacity or idx >= len(neurons):
                current_core += 1

    # 构建输入/输出映射结果
    input_neurons = neuron_id_map.get('input', [])
    output_neurons = neuron_id_map.get('output', [])
    input_mapping = {nid: neuron_to_gnc.get(nid, None) for nid in input_neurons}
    output_mapping = {nid: neuron_to_gnc.get(nid, None) for nid in output_neurons}

    # 将每个核心中输出神经元移至列表末尾（保持与路由表构建顺序一致）
    for g in range(16):
        assigned = nfu.get_GNC(g).contains
        if assigned:
            non_output = [neu for neu in assigned if neu not in output_neurons]
            output_list = [neu for neu in assigned if neu in output_neurons]
            nfu.get_GNC(g).contains = non_output + output_list

    gnc_usage = {}
    for g in range(16):
        used = len(nfu.get_GNC(g).contains)
        gnc_usage[g] = min(used / GNC_CAPACITY, 1.0)  # 归一化到 0~1

    # 2) 调用已有的绘图函数
    plot_gnc_mapping(gnc_coordinates, gnc_usage)

    # 构建核心级有向图，采用最短路径原则连接不同核心间的通信
    def build_gnc_graph(assignment):
        G = nx.DiGraph()
        # 将有神经元分配的核心添加为节点，确保无边也能反映节点
        for core_id in range(16):
            if nfu.get_GNC(core_id).contains:
                G.add_node(core_id)
        for src, tgt, _w in connections:
            if src in assignment and tgt in assignment:
                g_src = assignment[src]
                g_tgt = assignment[tgt]
                if g_src != g_tgt:
                    G.add_edge(g_src, g_tgt)
        return G

    G_final = build_gnc_graph(neuron_to_gnc)
    if not nx.is_directed_acyclic_graph(G_final):
        print("[WARNING] The final GNC graph has cycles?")
        longest_time_expression = "Undefined (Graph has cycles?)"
    else:
        try:
            longest_path = nx.dag_longest_path(G_final)
            num_hops = max(0, len(longest_path) - 1)
            num_cores_in_path = len(longest_path) or 0
            longest_time_expression = f"Longest_Time = ({num_hops} * T0) + ({num_cores_in_path} * T1) + T2"
        except Exception as e:
            print(f"[ERROR] Longest path calculation failed: {e}")
            longest_time_expression = "Error in computing the longest path"

    end_time = time.time()
    print(f"[DEBUG] cluster_pacman completed in {end_time - start_time:.2f} seconds.")
    return input_mapping, output_mapping, nfu, longest_time_expression


def cluster_sneap(neuron_id_map, connections, max_iter=200000):
    print("[DEBUG] Starting cluster_sneap function (SNEAP).")
    start_time = time.time()
    nfu = NFU(nfu_id=0)
    gnc_coordinates = {
        0: (3, 3), 1: (2, 3), 2: (1, 3), 3: (0, 3),
        4: (3, 2), 5: (2, 2), 6: (1, 2), 7: (0, 2),
        8: (3, 1), 9: (2, 1), 10: (1, 1), 11: (0, 1),
        12: (3, 0), 13: (2, 0), 14: (1, 0), 15: (0, 0)
    }
    GNC_CAPACITY = 8192
    capacity = GNC_CAPACITY

    # 收集全部神经元
    all_neurons = {nid for ids in neuron_id_map.values() for nid in ids}
    total_nrn = len(all_neurons)
    if total_nrn == 0:
        return {}, {}, nfu, "Longest_Time = (0 * T0) + (0 * T1) + T2"

    # --- 多级图划分：将神经元按连接关系分簇 ---
    G = nx.Graph()
    G.add_nodes_from(all_neurons)
    # 添加无向边表示神经元间通信
    for src, tgt, _w in connections:
        if src in all_neurons and tgt in all_neurons:
            G.add_edge(src, tgt)
    # 目标簇数下限：满足容量限制
    min_clusters = math.ceil(total_nrn / capacity)
    if min_clusters > 16:
        print("NFU 0 cannot accommodate all neurons. Need more NFUs.")
        return None, None, None, None
    target_clusters = max(1, min_clusters)
    # 初始社区划分（异步流体社区算法）
    communities = list(nx.algorithms.community.asyn_fluidc(G, target_clusters))
    clusters = [set(comm) for comm in communities]
    # 如果划分数不足目标（可能图有多个分量），对最大簇进行拆分
    while len(clusters) < target_clusters:
        largest = max(clusters, key=len)
        clusters.remove(largest)
        half = len(largest) // 2 or 1
        lst = list(largest)
        setA = set(lst[:half]); setB = set(lst[half:])
        clusters.append(setA); clusters.append(setB)
    # 拆分所有超过容量的簇
    i = 0
    while i < len(clusters):
        if len(clusters[i]) > capacity:
            lst = list(clusters[i])
            half = len(lst) // 2 or 1
            setA = set(lst[:half]); setB = set(lst[half:])
            clusters[i] = setA
            clusters.insert(i+1, setB)
            # 不增加 i，以继续检查新的簇 setA 是否仍超容量
        else:
            i += 1
    cluster_count = len(clusters)
    if cluster_count > 16:
        print("NFU 0 cannot accommodate all neurons after clustering. Need more NFUs.")
        return None, None, None, None

    # 建立 neuron->cluster 映射
    neuron_to_cluster = {}
    for cid, cluster_nodes in enumerate(clusters):
        for neu in cluster_nodes:
            neuron_to_cluster[neu] = cid

    # 预计算簇大小和簇间边（通信）
    cluster_size = [len(cset) for cset in clusters]
    cluster_edges = set()
    for src, tgt, _w in connections:
        if src in neuron_to_cluster and tgt in neuron_to_cluster:
            ci, cj = neuron_to_cluster[src], neuron_to_cluster[tgt]
            if ci != cj:
                cluster_edges.add((ci, cj))

    # --- 模拟退火：优化簇到核心的映射 ---
    cluster_count = len(clusters)
    # 初始映射：随机将每个簇分配到不同核心
    cores = list(range(16))
    random.shuffle(cores)
    current_mapping = cores[:cluster_count]
    current_cost = None
    best_mapping = None
    best_cost = float('inf')

    def compute_cost(mapping):
        # (1) 负载均衡
        usage = [0] * 16
        for cid, core in enumerate(mapping):
            usage[core] += cluster_size[cid]
        usage_ratios = [u / capacity for u in usage]
        mean_usage = sum(usage_ratios) / 16
        usage_imb = math.sqrt(sum((r - mean_usage) ** 2 for r in usage_ratios) / 16)

        # (2) 跨核边数 + (3) 曼哈顿距离
        edge_cnt, dist_sum = 0, 0
        G_tmp = nx.DiGraph()  # ← 新增：临时图
        for ci, cj in cluster_edges:
            core_i, core_j = mapping[ci], mapping[cj]
            if core_i != core_j:
                edge_cnt += 1
                x1, y1 = gnc_coordinates[core_i];
                x2, y2 = gnc_coordinates[core_j]
                dist_sum += abs(x1 - x2) + abs(y1 - y2)
                G_tmp.add_edge(core_i, core_j)  # ← 收集方向边

        cost_val = (1.0 * usage_imb
                    + 0.5 * edge_cnt
                    + 0.1 * dist_sum)

        # (4) 容量惩罚
        if any(c > capacity for c in usage):
            cost_val += 1e9

        # (5) **DAG 惩罚**  —— 插在 return 之前
        if not nx.is_directed_acyclic_graph(G_tmp):
            cost_val += 1e9

        return cost_val

    current_cost = compute_cost(current_mapping)
    best_mapping = current_mapping[:]
    best_cost = current_cost

    # 退火参数初始化
    T_init, T_final = 5.0, 0.01
    cooling_rate = (T_final / T_init) ** (1.0 / max_iter) if max_iter > 0 else 1.0
    T = T_init

    for it in tqdm(range(max_iter), desc="Simulated Annealing", leave=False):
        # 每轮温度下尝试对每个簇进行一次随机移动
        for cid in range(cluster_count):
            old_core = current_mapping[cid]
            # 随机选一个不同核心
            target_core = random.randrange(16)
            if target_core == old_core:
                continue
            # 找到目标核心上目前的簇（如有）
            other_cid = None
            if target_core in current_mapping:
                other_cid = current_mapping.index(target_core)
            # 执行交换/迁移
            if other_cid is not None:
                # 与另一个簇交换核心
                current_mapping[cid], current_mapping[other_cid] = target_core, old_core
            else:
                # 将簇 cid 移到空闲的 target_core
                current_mapping[cid] = target_core
            new_cost = compute_cost(current_mapping)
            # 根据 Metropolis 准则决定是否接受新解
            delta = new_cost - current_cost
            if new_cost < current_cost or random.random() < math.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mapping = current_mapping[:]
            else:
                # 恢复原映射
                if other_cid is not None:
                    current_mapping[cid], current_mapping[other_cid] = old_core, target_core
                else:
                    current_mapping[cid] = old_core
        # 降低温度
        T *= cooling_rate
        if T < T_final:
            break

    G_check = nx.DiGraph()
    for (src, tgt, _w) in connections:
        core_src = best_mapping[neuron_to_cluster[src]]
        core_tgt = best_mapping[neuron_to_cluster[tgt]]
        if core_src != core_tgt:
            G_check.add_edge(core_src, core_tgt)

    if not nx.is_directed_acyclic_graph(G_check):  # 仍然成环
        for _ in range(10):  # 打印 10 次
            print("cycle")
        print("[ERROR] SNEAP abort: final mapping is cyclic.")
        return None, None, None, None

        # 最优映射结果 best_mapping 应用到 NFU
    for core_id in range(16):
        nfu.get_GNC(core_id).contains.clear()
    neuron_to_gnc = {}
    for neu, cid in neuron_to_cluster.items():
        core = best_mapping[cid]
        neuron_to_gnc[neu] = core
        nfu.get_GNC(core).add(neu)

    # 构建输入/输出映射
    input_neurons = neuron_id_map.get('input', [])
    output_neurons = neuron_id_map.get('output', [])
    input_mapping = {nid: neuron_to_gnc.get(nid, None) for nid in input_neurons}
    output_mapping = {nid: neuron_to_gnc.get(nid, None) for nid in output_neurons}

    # 将输出神经元移至各核心 contains 列表末尾
    for g in range(16):
        assigned = nfu.get_GNC(g).contains
        if assigned:
            non_output = [neu for neu in assigned if neu not in output_neurons]
            output_list = [neu for neu in assigned if neu in output_neurons]
            nfu.get_GNC(g).contains = non_output + output_list

    gnc_usage = {}
    for g in range(16):
        used = len(nfu.get_GNC(g).contains)
        gnc_usage[g] = min(used / GNC_CAPACITY, 1.0)  # 归一化到 0~1

    # 2) 调用已有的绘图函数
    plot_gnc_mapping(gnc_coordinates, gnc_usage)

    # 计算最长时间表达式（基于核心间最长路径）
    def build_gnc_graph(assignment):
        G = nx.DiGraph()
        for core_id in range(16):
            if nfu.get_GNC(core_id).contains:
                G.add_node(core_id)
        for src, tgt, _w in connections:
            if src in assignment and tgt in assignment:
                g_src, g_tgt = assignment[src], assignment[tgt]
                if g_src != g_tgt:
                    G.add_edge(g_src, g_tgt)
        return G

    G_final = build_gnc_graph(neuron_to_gnc)
    if not nx.is_directed_acyclic_graph(G_final):
        longest_time_expression = "Undefined (Graph has cycles?)"
    else:
        try:
            longest_path = nx.dag_longest_path(G_final)
            num_hops = max(0, len(longest_path) - 1)
            num_cores = len(longest_path) or 0
            longest_time_expression = f"Longest_Time = ({num_hops} * T0) + ({num_cores} * T1) + T2"
        except Exception as e:
            longest_time_expression = "Error in computing the longest path"

    end_time = time.time()
    print(f"[DEBUG] cluster_sneap completed in {end_time - start_time:.2f} seconds.")
    print(f"[DEBUG] Best cost found: {best_cost:.4f}")
    return input_mapping, output_mapping, nfu, longest_time_expression


# =========================================================
# ⬇⬇   顶 层  工 具  函 数（需置于模块最外层，便于 pickle）   ⬇⬇
# =========================================================
def build_gnc_graph_from_mapping(mapping, cluster_edges, gnc_coordinates):
    """只用 cluster->GNC 映射 & cluster_edges 构建临时 GNC 图"""
    G = nx.DiGraph()
    for (ci, cj) in cluster_edges:
        gi, gj = mapping[ci], mapping[cj]
        if gi != gj:
            G.add_edge(gi, gj)
    return G

def compute_cost_parallel(mapping,        # list[int]  长度 = cluster_cnt
                          cluster_size,   # list[int]
                          cluster_edges,  # set[(ci,cj)]
                          capacity,
                          gnc_coordinates):
    # ---------- (1) 负载均衡 ----------
    usage = [0] * 16
    for cid, core in enumerate(mapping):
        usage[core] += cluster_size[cid]

    ratios = [u / capacity for u in usage]
    mu = sum(ratios) / 16
    usage_imb = math.sqrt(sum((r - mu) ** 2 for r in ratios) / 16)

    # ---------- (2) 跨核边 + 曼哈顿距离 ----------
    edge_cnt, dist_sum = 0, 0
    G_tmp = nx.DiGraph()
    for ci, cj in cluster_edges:
        gi, gj = mapping[ci], mapping[cj]
        if gi != gj:
            edge_cnt += 1
            x1, y1 = gnc_coordinates[gi]
            x2, y2 = gnc_coordinates[gj]
            dist_sum += abs(x1 - x2) + abs(y1 - y2)
            G_tmp.add_edge(gi, gj)

    cost = 1.0 * usage_imb + 0.5 * edge_cnt + 0.1 * dist_sum

    # ---------- (3) 容量 & DAG 惩罚 ----------
    if any(u > capacity for u in usage) or not nx.is_directed_acyclic_graph(G_tmp):
        cost += 1e9
    return cost

def ensure_unique(mapping):
    """若多个簇映射同一核心，将后续重复簇换到空闲核心"""
    used, dups = set(), []
    for idx, core in enumerate(mapping):
        if core in used:
            dups.append(idx)
        else:
            used.add(core)
    free = [c for c in range(16) if c not in used]
    for idx in dups:
        mapping[idx] = free.pop() if free else mapping[idx]
    return mapping

def evaluate_particle(pos,
                      cluster_size, cluster_edges,
                      capacity, gnc_coordinates):
    """子进程调用：确保唯一 → 计算 cost → 返回"""
    pos = ensure_unique(pos)
    cost = compute_cost_parallel(pos, cluster_size, cluster_edges,
                                 capacity, gnc_coordinates)
    return pos, cost

def break_cycles(mapping,
                 cluster_edges, gnc_coordinates,
                 max_retry=100):
    """若映射仍有环，随机迁移环中一簇到空闲 GNC 直至无环"""
    for _ in range(max_retry):
        G = build_gnc_graph_from_mapping(mapping, cluster_edges, gnc_coordinates)
        if nx.is_directed_acyclic_graph(G):
            return mapping
        # 取一条回环边
        ci, cj = next(iter(nx.find_cycle(G)))
        # 把 cj 对应簇移到空闲核
        free = [g for g in range(16) if g not in mapping]
        if not free:
            free = list(range(16))
        # cj 可能是源也可能是目标 → 找任何映射到 cj 的簇
        victims = [idx for idx, core in enumerate(mapping) if core == cj]
        mapping[random.choice(victims)] = random.choice(free)
    return mapping  # 若失败，返回最后尝试结果
# =========================================================
# ⬆⬆   ⬆⬆   ⬆⬆   工 具  函 数  结  束   ⬆⬆   ⬆⬆   ⬆⬆
# =========================================================

def cluster_spinemap(neuron_id_map, connections,
                     pop_size=192,             # 并行粒子数
                     max_iter=200000,             # 迭代轮数
                     max_workers=24):          # 7900X → 24 线程
    print("[DEBUG] Starting parallel SpiNeMap…")
    t0 = time.time()

    # ---------- 0) NFU / 坐标 / 容量 ----------
    nfu = NFU(nfu_id=0)
    gnc_coordinates = {
        0: (3,3),  1: (2,3),  2: (1,3),  3: (0,3),
        4: (3,2),  5: (2,2),  6: (1,2),  7: (0,2),
        8: (3,1),  9: (2,1), 10: (1,1), 11: (0,1),
        12:(3,0), 13:(2,0), 14:(1,0), 15:(0,0)
    }
    capacity = 8192

    # ---------- 1) 收集神经元 ----------
    all_neurons = {nid for ids in neuron_id_map.values() for nid in ids}
    if not all_neurons:
        return {}, {}, nfu, "Longest_Time = (0*T0)+(0*T1)+T2"

    # ---------- 2) 递归 KL 二分 -> clusters ----------
    clusters = [set(all_neurons)]
    i = 0
    while i < len(clusters):
        if len(clusters[i]) <= capacity:
            i += 1; continue
        clu_nodes = clusters[i]
        subG = nx.Graph()
        subG.add_nodes_from(clu_nodes)
        for s, t, _ in connections:
            if s in clu_nodes and t in clu_nodes:
                subG.add_edge(s, t)
        # 随机初划分
        half = min(capacity, len(clu_nodes)//2 or 1)
        nodes = list(clu_nodes)
        initA, initB = set(nodes[:half]), clu_nodes - set(nodes[:half])
        try:
            A, B = nx.algorithms.community.kernighan_lin_bisection(subG, (initA, initB))
        except Exception:
            A, B = initA, initB
        clusters[i] = A
        clusters.insert(i+1, B)

    # 补足簇数 (小概率)
    min_clusters = math.ceil(len(all_neurons)/capacity)
    while len(clusters) < min_clusters:
        big = max(clusters, key=len); clusters.remove(big)
        nodes = list(big); mid = len(nodes)//2 or 1
        clusters.extend([set(nodes[:mid]), set(nodes[mid:])])

    if len(clusters) > 16:
        print("[FATAL] Need >16 GNC → abort")
        return None, None, None, None

    # ---------- 3) 建映射辅助结构 ----------
    neuron_to_cluster = {}
    for cid, cset in enumerate(clusters):
        for n in cset:
            neuron_to_cluster[n] = cid
    cluster_size = [len(c) for c in clusters]
    cluster_edges = {(neuron_to_cluster[s], neuron_to_cluster[t])
                     for s, t, _ in connections
                     if neuron_to_cluster[s] != neuron_to_cluster[t]}
    cluster_cnt = len(clusters)

    # ---------- 4) 初始化粒子 ----------
    particles = [{
        'position': ensure_unique(random.sample(range(16), cluster_cnt)),
        'pbest_pos': None,
        'pbest_cost': float('inf')
    } for _ in range(pop_size)]

    g_best_pos, g_best_cost = None, float('inf')

    # ---------- 5) 并行 PSO ----------
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for it in tqdm(range(max_iter), desc="PSO"):
            cur_vel = max(1, int(cluster_cnt - (cluster_cnt-1)*it/max_iter))
            # (a) 生成扰动后的 positions
            jobs = []
            for p in particles:
                pos = p['position'][:]
                for _ in range(cur_vel):
                    idx = random.randrange(cluster_cnt)
                    if p['pbest_pos'] and random.random()<0.3:
                        pos[idx] = p['pbest_pos'][idx]
                    if g_best_pos and random.random()<0.3:
                        pos[idx] = g_best_pos[idx]
                    if random.random() < 0.02:
                        pos[idx] = random.randrange(16)
                jobs.append(pos)

            # (b) 并行计算 cost
            results = list(pool.map(
                evaluate_particle,
                jobs,
                itertools.repeat(cluster_size),
                itertools.repeat(cluster_edges),
                itertools.repeat(capacity),
                itertools.repeat(gnc_coordinates)
            ))

            # (c) 更新 pbest / gbest
            for p, (new_pos, new_cost) in zip(particles, results):
                p['position'] = new_pos
                if new_cost < p['pbest_cost']:
                    p['pbest_cost'] = new_cost
                    p['pbest_pos']  = new_pos
                if new_cost < g_best_cost:
                    g_best_cost = new_cost
                    g_best_pos   = new_pos

    print(f"[INFO] PSO finished. g_best_cost = {g_best_cost:.4e}")

    # ---------- 6) 最终 break_cycles 兜底 ----------
    g_best_pos = break_cycles(g_best_pos, cluster_edges, gnc_coordinates)
    if not nx.is_directed_acyclic_graph(
            build_gnc_graph_from_mapping(g_best_pos, cluster_edges, gnc_coordinates)):
        for _ in range(10):
            print("cycle")
        print("[ERROR] break_cycles failed. Abort.")
        return None, None, None, None

    # ---------- 7) 写入 NFU ----------
    for g in range(16):
        nfu.get_GNC(g).contains.clear()
    neuron_to_gnc = {}
    for neu, cid in neuron_to_cluster.items():
        core = g_best_pos[cid]
        neuron_to_gnc[neu] = core
        nfu.get_GNC(core).add(neu)

    # ---------- 8) input / output map ----------
    input_mapping  = {n: neuron_to_gnc[n] for n in neuron_id_map.get('input', [])}
    output_mapping = {n: neuron_to_gnc[n] for n in neuron_id_map.get('output', [])}

    # ---------- 9) 可视化 ----------
    usage = {g: min(len(nfu.get_GNC(g).contains)/capacity, 1.0) for g in range(16)}
    plot_gnc_mapping(gnc_coordinates, usage)

    # ---------- 10) Longest path ----------
    G_final = build_gnc_graph_from_mapping(g_best_pos, cluster_edges, gnc_coordinates)
    lp = nx.dag_longest_path(G_final)
    longest_time_expr = f"Longest_Time = ({len(lp)-1}*T0) + ({len(lp)}*T1) + T2"

    print(f"[DEBUG] cluster_spinemap_parallel OK, elapsed {time.time()-t0:.2f}s")
    return input_mapping, output_mapping, nfu, longest_time_expr


def multi_cluster_greedy_mapping(
        neuron_id_map: Dict[str, list],
        connections: list
) -> Tuple[
        Dict[Any, int],                                      # best_input_mapping
        Dict[Any, int],                                      # best_output_mapping
        "NFU",                                               # best_nfu
        str,                                                 # expression tag
]:
    """
    Greedy multi‑cluster mapper with N∈{16,8,4,2,1}.  Always returns
    (best_in, best_out, best_nfu, 'Multi‑Cluster‑N=k', full_results_dict) full_results_dict目前未启用
    """

    capacity = 8192
    gnc_coordinates = {
        0: (3, 3), 1: (2, 3), 2: (1, 3), 3: (0, 3),
        4: (3, 2), 5: (2, 2), 6: (1, 2), 7: (0, 2),
        8: (3, 1), 9: (2, 1), 10: (1, 1), 11: (0, 1),
        12: (3, 0), 13: (2, 0), 14: (1, 0), 15: (0, 0)
    }
    POS16 = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    POS_DICT = {16: POS16, 8: POS16[:8], 4: POS16[:4], 2: POS16[:2], 1: POS16[:1]}

    def manhattan_distance(g1: int, g2: int) -> int:
        (x1, y1), (x2, y2) = gnc_coordinates[g1], gnc_coordinates[g2]
        return abs(x1 - x2) + abs(y1 - y2)

    def build_adj_weight(neuron2clu, N: int):
        w = [[0] * N for _ in range(N)]
        for src, tgt, _ in connections:
            c1, c2 = neuron2clu[src], neuron2clu[tgt]
            if c1 != c2:
                w[c1][c2] += 1
                w[c2][c1] += 1
        return w

    def place_clusters_distance_aware(N, neuron2clu):
        free_gnc = POS_DICT[N].copy()
        adj = build_adj_weight(neuron2clu, N)

        first = min(range(N), key=lambda c: (-sum(adj[c]), c))
        cluster2gnc = {first: free_gnc.pop(0)}
        placed = {first}

        while free_gnc:
            cand = [c for c in range(N) if c not in placed]
            target = min(cand, key=lambda c: (-sum(adj[c][p] for p in placed), c))
            best_gnc = min(
                free_gnc,
                key=lambda g: sum(manhattan_distance(g, cluster2gnc[p]) for p in placed)
            )
            cluster2gnc[target] = best_gnc
            free_gnc.remove(best_gnc)
            placed.add(target)
        return cluster2gnc

    def build_gnc_graph(neuron2clu, clu2gnc):
        G = nx.DiGraph()
        for src, tgt, _ in connections:
            g1, g2 = clu2gnc[neuron2clu[src]], clu2gnc[neuron2clu[tgt]]
            if g1 != g2:
                G.add_edge(g1, g2)
        return G

    # 缓存
    results: Dict[int, Tuple[Dict[Any, int], Dict[Any, int]]] = {}
    nfu_map, cluster2gnc_map, neuron2cluster_map = {}, {}, {}

    total_neurons = sum(len(v) for v in neuron_id_map.values())
    layer_order = list(neuron_id_map.keys())

    for N in [16, 8, 4, 2, 1]:
        avg_per_clu = math.ceil(total_neurons / N)
        if avg_per_clu > capacity:
            for m in [k for k in [16, 8, 4, 2, 1] if k >= N]:
                results.setdefault(m, (None, None))
            break

        clusters = [[] for _ in range(N)]
        c_idx = 0
        for layer in layer_order:
            for neuron in neuron_id_map[layer]:
                clusters[c_idx].append(neuron)
                if len(clusters[c_idx]) >= avg_per_clu and c_idx < N - 1:
                    c_idx += 1

        if any(len(clu) > capacity for clu in clusters):
            for m in [k for k in [16, 8, 4, 2, 1] if k >= N]:
                results.setdefault(m, (None, None))
            break

        neuron2cluster = {n: idx for idx, clu in enumerate(clusters) for n in clu}
        cluster2gnc = place_clusters_distance_aware(N, neuron2cluster)

        G = build_gnc_graph(neuron2cluster, cluster2gnc)
        if not nx.is_directed_acyclic_graph(G):
            for m in [k for k in [16, 8, 4, 2, 1] if k >= N]:
                results.setdefault(m, (None, None))
            break

        # 缓存 NFU
        nfu_tmp = NFU(nfu_id=0)
        for neu in neuron2cluster:
            g = cluster2gnc[neuron2cluster[neu]]
            nfu_tmp.get_GNC(g).add(neu)
        nfu_map[N] = nfu_tmp
        cluster2gnc_map[N] = cluster2gnc
        neuron2cluster_map[N] = neuron2cluster

        input_mapping = {n: cluster2gnc[neuron2cluster[n]]
                         for n in neuron_id_map.get('input', [])}
        output_mapping = {n: cluster2gnc[neuron2cluster[n]]
                          for n in neuron_id_map.get('output', [])}
        results[N] = (input_mapping, output_mapping)

        if avg_per_clu * 2 > capacity:
            for m in [k for k in [8, 4, 2, 1] if k < N]:
                results.setdefault(m, (None, None))
            break

    for N in [16, 8, 4, 2, 1]:
        results.setdefault(N, (None, None))

    bestN = next((n for n in [1, 2, 4, 8, 16] if results[n][0] is not None), None)
    if bestN is None:
        return {}, {}, None, "Undefined (all GDV)"

    best_in, best_out = results[bestN]
    best_nfu = nfu_map[bestN]
    expr = f"Multi‑Cluster‑N={bestN}"
    return best_in, best_out, best_nfu, expr


# ========== 1) 将单次退火逻辑封装成函数：run_single_sa(...) ==========

def run_single_sa(
        neuron_list,
        connections,
        gnc_coordinates,
        capacity=8192,
        w_usage=1.0,
        w_edge=0.5,
        w_dist=0.1,
        T_init=5.0,
        T_final=0.01,
        max_iter=20000,
        seed=0
):
    """
    只做一次模拟退火，返回:
       (best_assignment, best_cost, time_spent)

    - neuron_list: 要分配的所有神经元 (list)
    - connections: [(src, tgt, w), ...]
    - gnc_coordinates: {gnc_id: (x,y)}
    - capacity: 每个 GNC 的容量 (默认 8192)
    - w_usage, w_edge, w_dist: cost 函数各项权重
    - T_init, T_final, max_iter: 退火参数
    - seed: 随机数种子
    """

    start_time = time.time()
    random.seed(seed)  # 不同进程用不同种子，避免生成相同分配

    # ========== (1) 初始化随机分配 neuron->gnc ==========
    neuron_to_gnc = {}
    full_gncs = set()

    for neuron in neuron_list:
        assigned = False
        for _try in range(50):
            candidate_gnc = random.randint(0, 15)
            if candidate_gnc in full_gncs:
                continue
            current_count = sum(1 for n, g in neuron_to_gnc.items() if g == candidate_gnc)
            if current_count < capacity:
                neuron_to_gnc[neuron] = candidate_gnc
                if current_count + 1 >= capacity:
                    full_gncs.add(candidate_gnc)
                assigned = True
                break

        if not assigned:
            # 容量不够, 提前返回
            return None, float('inf'), 0.0

    # ========== (2) 定义辅助函数 ==========
    def build_gnc_graph(assignment):
        G = nx.DiGraph()
        for src, tgt, _w in connections:
            g_src = assignment[src]
            g_tgt = assignment[tgt]
            if g_src != g_tgt:
                G.add_edge(g_src, g_tgt)
        return G

    def manhattan_distance(gnc1, gnc2):
        x1, y1 = gnc_coordinates[gnc1]
        x2, y2 = gnc_coordinates[gnc2]
        return abs(x1 - x2) + abs(y1 - y2)

    def compute_cost(assignment):
        # usage imbalance
        usage_count = [0] * 16
        for n, g in assignment.items():
            usage_count[g] += 1
        usage_ratios = [uc / capacity for uc in usage_count]
        mean_usage = sum(usage_ratios) / 16
        usage_imbalance = math.sqrt(sum((r - mean_usage) ** 2 for r in usage_ratios) / 16)

        G_tmp = build_gnc_graph(assignment)
        edge_count = len(G_tmp.edges())

        sum_of_manhattan_dist = 0
        for (u, v) in G_tmp.edges():
            sum_of_manhattan_dist += manhattan_distance(u, v)

        cost_val = (w_usage * usage_imbalance) + \
                   (w_edge * edge_count) + \
                   (w_dist * sum_of_manhattan_dist)
        return cost_val

    # ========== (3) 模拟退火主循环 ==========
    current_assignment = dict(neuron_to_gnc)
    current_cost = compute_cost(current_assignment)
    best_assignment = dict(current_assignment)
    best_cost = current_cost

    cooling_rate = (T_final / T_init) ** (1.0 / max_iter)
    T = T_init

    for _ in range(max_iter):
        # 随机扰动
        neuron = random.choice(neuron_list)
        old_gnc = current_assignment[neuron]
        new_gnc = random.randint(0, 15)
        if new_gnc == old_gnc:
            continue

        # 容量检查
        count_new_gnc = sum(1 for n, g in current_assignment.items() if g == new_gnc)
        if count_new_gnc >= capacity:
            continue

        # 临时改动
        current_assignment[neuron] = new_gnc

        # 检查是否产生环
        G_tmp = build_gnc_graph(current_assignment)
        if not nx.is_directed_acyclic_graph(G_tmp):
            # 回滚
            current_assignment[neuron] = old_gnc
            continue

        # 计算 cost
        new_cost = compute_cost(current_assignment)
        delta = new_cost - current_cost

        if delta < 0:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_assignment = dict(current_assignment)
        else:
            accept_prob = math.exp(-delta / T)
            if random.random() < accept_prob:
                current_cost = new_cost
            else:
                current_assignment[neuron] = old_gnc

        # 退火
        T *= cooling_rate
        if T < T_final:
            break

    # 整个退火结束
    end_time = time.time()
    time_spent = end_time - start_time

    return best_assignment, best_cost, time_spent


# ========== 2) 在 cluster2complex(...) 中并行调用 run_single_sa(...) ==========

def cluster2complex(
        neuron_id_map,
        connections,
        relation=1,
        n_jobs=4,
        max_iter=20000,
        T_init=5.0,
        T_final=0.01
):
    """
    并行运行多次模拟退火, 并在主进程里通过 as_completed(...) 输出每个进程的完成时间。
    参数:
      - n_jobs: 并行数量 (CPU 核数)
      - max_iter, T_init, T_final: 退火参数
    """

    print(f"[DEBUG] Starting cluster2complex with parallel SA. n_jobs={n_jobs}")
    overall_start = time.time()

    # ---------- (1) 收集所有神经元 ----------
    neuron_list = []
    for layer_name, n_list in neuron_id_map.items():
        neuron_list.extend(n_list)
    neuron_list = list(set(neuron_list))

    # ---------- (2) GNC 坐标与容量 ----------
    gnc_coordinates = {
        0: (3, 3), 1: (2, 3), 2: (1, 3), 3: (0, 3),
        4: (3, 2), 5: (2, 2), 6: (1, 2), 7: (0, 2),
        8: (3, 1), 9: (2, 1), 10: (1, 1), 11: (0, 1),
        12: (3, 0), 13: (2, 0), 14: (1, 0), 15: (0, 0)
    }
    capacity = 8192

    # ---------- (3) 多进程并行 ----------
    #    用 concurrent.futures, 逐个等待任务完成并打印进度
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # 随机生成种子, 每个进程用一个
    seeds = [random.randint(1, 10_000_000) for _ in range(n_jobs)]

    futures = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i in range(n_jobs):
            futures.append(
                executor.submit(
                    run_single_sa,
                    neuron_list,
                    connections,
                    gnc_coordinates,
                    capacity,
                    1.0,  # w_usage
                    0.5,  # w_edge
                    0.1,  # w_dist
                    T_init,
                    T_final,
                    max_iter,
                    seeds[i]
                )
            )

        # ---------- (4) as_completed(...) 逐个收集结果 ----------
        best_assignment_global = None
        best_cost_global = float('inf')
        best_time_global = 0.0

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            single_assignment, single_cost, single_time = future.result()
            # 在这里输出此进程完成信息
            print(f"[DEBUG] -> Process {done_count}/{n_jobs} done: cost={single_cost:.4f}, time={single_time:.2f}s")

            if single_assignment is not None and single_cost < best_cost_global:
                best_cost_global = single_cost
                best_assignment_global = single_assignment
                best_time_global = single_time

    # ---------- (5) 如果所有进程都失败, 返回 None ----------
    if best_assignment_global is None:
        print("[ERROR] All parallel SA runs failed or no valid assignment found.")
        return None, None, None, None

    print(f"[DEBUG] Best cost among all parallel runs: {best_cost_global:.4f}, time={best_time_global:.2f}s")

    # ---------- (6) 构建最终 NFU 并加载 assignment ----------
    nfu = NFU(nfu_id=0)
    for gnc_id in range(16):
        nfu.get_GNC(gnc_id).contains.clear()

    for n, g in best_assignment_global.items():
        nfu.get_GNC(g).add(n)

    # ---------- (7) input_mapping / output_mapping ----------
    input_neurons = neuron_id_map.get('input', [])
    output_neurons = neuron_id_map.get('output', [])

    input_mapping = {nid: best_assignment_global[nid] for nid in input_neurons}
    output_mapping = {nid: best_assignment_global[nid] for nid in output_neurons}

    # ---------- (8) longest_time_expression ----------
    def build_gnc_graph(assignment):
        G = nx.DiGraph()
        for src, tgt, _w in connections:
            g_src = assignment[src]
            g_tgt = assignment[tgt]
            if g_src != g_tgt:
                G.add_edge(g_src, g_tgt)
        return G

    G_final = build_gnc_graph(best_assignment_global)
    if not nx.is_directed_acyclic_graph(G_final):
        print("[WARNING] Best assignment has cycles? (Shouldn't happen.)")
        longest_time_expression = "Undefined (Graph has cycles?)"
    else:
        try:
            longest_path = nx.dag_longest_path(G_final)
            num_hops = len(longest_path) - 1
            num_gncs_in_path = len(longest_path)
            longest_time_expression = f"Longest_Time = ({num_hops} * T0) + ({num_gncs_in_path} * T1) + T2"
        except Exception as e:
            print(f"[ERROR] DAG longest path calculation failed: {e}")
            longest_time_expression = "Error in computing the longest path"

    # ---------- (9) 可视化 GNC 利用率 ----------
    gnc_usage = {}
    for gnc_id in range(16):
        count = len(nfu.get_GNC(gnc_id).contains)
        gnc_usage[gnc_id] = min(count / capacity, 1.0)

    plot_gnc_mapping(gnc_coordinates, gnc_usage)

    overall_time = time.time() - overall_start
    print(f"[DEBUG] cluster2complex parallel SA finished. Total elapsed: {overall_time:.2f}s")

    return input_mapping, output_mapping, nfu, longest_time_expression


# -------------------------------------  输出逻辑  -----------------------------------------------


def check_save_eq(save_file_name, net, sample_input_size):
    """
    检查是否存在 save_file_name 文件，并尝试加载。
    如果文件存在，就比较网络结构和输入尺寸是否与当前一致。
    一致则返回加载的所有数据；不一致或异常则返回 None。

    参数:
    - save_file_name: 保存文件的名称。
    - net: 当前的神经网络 (nn.Module) 实例。
    - sample_input_size: 当前推断的输入尺寸 (tuple)。

    返回:
    - loaded_data (dict 或者 None):
      如果存在并且网络结构 & 输入尺寸一致，则返回字典，其中包含之前保存的内容；
      否则返回 None。
    """
    if not os.path.exists(save_file_name):
        print(f"\n未检测到 {save_file_name} 文件，将进行首次分配与构建连接...")
        return None

    try:
        with open(save_file_name, "rb") as f:
            saved_data = pickle.load(f)

        saved_net_structure = saved_data["net_structure"]  # 上次保存时的网络结构(str形式)
        saved_input_size = saved_data["input_size"]  # 上次保存时的输入尺寸

        current_net_structure = str(net)
        if current_net_structure == saved_net_structure and sample_input_size == saved_input_size:
            print(f"\n检测到已有 {save_file_name}，且网络结构与输入尺寸一致，"
                  f"跳过重新分配与构建连接。")
            return saved_data
        else:
            print(f"\n检测到已有 {save_file_name}，但网络结构或输入尺寸不一致，"
                  f"将重新进行分配与构建连接...")
            return None

    except Exception as e:
        print(f"\n加载 {save_file_name} 出错: {e}，将重新进行分配与构建连接...")
        return None


def save_or_not(
        save_file_name,
        net,
        sample_input_size,
        neuron_id_map,
        total_neurons,
        connections,
        input_mapping,
        output_mapping,
        nfu,
        longest_time_expression
):
    """
    询问用户是否需要将当前映射结果保存到本地文件 (save_file_name)。
    如果用户选择 "y"，则覆盖/更新原有的保存文件。

    参数:
    - save_file_name: 保存文件的名称 (str)。
    - net: 当前的神经网络 (nn.Module) 实例。
    - sample_input_size: 当前推断的输入尺寸 (tuple)。
    - neuron_id_map: 映射表，记录每层对应的神经元ID列表。
    - total_neurons: 神经元总数 (int)。
    - connections: 所有连接关系的列表 [(src, tgt, weight), ...]。
    - input_mapping: 输入层神经元到 GNC 的映射 (dict)。
    - output_mapping: 输出层神经元到 GNC 的映射 (dict)。
    - nfu: NFU 实例。
    - longest_time_expression: 表示最长耗时路径的字符串表达式 (str)。
    """
    choice = input("\n是否需要将映射结果(包括net, input_size, input_mapping等)保存到本地? (y/n): ").strip().lower()
    if choice == "y":
        data_to_save = {
            "net_structure": str(net),  # 仅比较结构，不比较具体权重
            "input_size": sample_input_size,
            "neuron_id_map": neuron_id_map,
            "total_neurons": total_neurons,
            "connections": connections,
            "input_mapping": input_mapping,
            "output_mapping": output_mapping,
            "nfu": nfu,
            "longest_time_expression": longest_time_expression
        }
        with open(save_file_name, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"已将当前分配结果保存到 {save_file_name}。下次运行时net结构与input_size一致，将直接加载此映射。")
    else:
        print("用户选择不保存映射结果。")


def output_transfer(
        neuron_id_map: dict,
        nfu: NFU,
        connections: list,  # 传入 connections
        output_filename_1: str = "outputAPI.pkl",
        output_filename_2: str = "connectionAPI.pkl"
):
    """
    将每一层的神经元在 GNC 上的映射位置保存到 .pkl 文件中，并且将连接信息转换为 GNC 的映射形式，保存在另一个文件中。
    """

    # ---------- 1) 先从 nfu 里取出所有 (gnc_id -> neuron_list) 的映射 ----------
    nfu_map = {}
    for gnc_idx, gnc_obj in nfu.GNCs.items():
        nfu_map[gnc_idx] = list(gnc_obj.contains)  # 转成 list

    # ---------- 2) 反向索引：全局 neuron_id -> (gnc_idx, local_idx_in_this_GNC) ----------
    global_to_gnc_map = {}
    for gnc_idx, neuron_list in nfu_map.items():
        for local_idx, global_neuron_id in enumerate(neuron_list):
            global_to_gnc_map[global_neuron_id] = (gnc_idx, local_idx)

    # ---------- 3) 按照 neuron_id_map 的顺序，生成每层的映射列表 ----------
    layer_dict = {}
    transferred_input = []
    transferred_output = []
    for layer_idx, (layer_name, neuron_list) in enumerate(neuron_id_map.items()):
        location_list = []
        for g_neuron_id in neuron_list:
            if g_neuron_id in global_to_gnc_map:
                gnc_idx, local_idx = global_to_gnc_map[g_neuron_id]
                location_list.append(f"{gnc_idx}_{local_idx}")
                # 在这里区分输入层和输出层的转换结果
                if layer_name == 'input':
                    transferred_input.append(f"{gnc_idx}_{local_idx}")
                elif layer_name == 'output':
                    transferred_output.append(f"{gnc_idx}_{local_idx}")
            else:
                # 如果出现没有被分配到任何 GNC 的神经元，可自定义处理逻辑
                location_list.append("NA_NA")
        layer_dict[layer_idx] = location_list

    # ---------- 4) 保存到第一个 .pkl 文件 ----------
    with open(output_filename_1, 'wb') as f:
        pickle.dump(layer_dict, f)

    print(f"输出文件已保存至: {output_filename_1}")

    # ---------- 5) 转换连接数据并保存到第二个 .pkl 文件 ----------
    connection_dict = []
    for src_id, tgt_id, weight in connections:
        if src_id in global_to_gnc_map and tgt_id in global_to_gnc_map:
            src_gnc_idx, src_local_idx = global_to_gnc_map[src_id]
            tgt_gnc_idx, tgt_local_idx = global_to_gnc_map[tgt_id]
            # 转换成 GNC 格式
            connection_dict.append((f"{src_gnc_idx}_{src_local_idx}",
                                    f"{tgt_gnc_idx}_{tgt_local_idx}",
                                    weight))
        else:
            # 如果某些连接的神经元没有对应的 GNC 映射，可以选择跳过或者处理成 "NA_NA"
            connection_dict.append(("NA_NA", "NA_NA", weight))

    # 保存转换后的连接信息
    with open(output_filename_2, 'wb') as f:
        pickle.dump(connection_dict, f)

    print(f"连接信息已保存至: {output_filename_2}")

    # 返回转换后的输入和输出
    return transferred_input, transferred_output


def ask_user_with_timeout(prompt, timeout=10, default='1'):
    """
    在主线程里打印 prompt，然后开启一个后台线程等待用户输入。
    如果在 timeout 秒内用户没有输入，则返回 default。
    返回用户输入(去除空格)或 default。
    """
    # 存放结果的容器
    user_input_box = {'value': None}

    def _input_reader():
        try:
            # 读取一行输入
            usr = input(prompt).strip()
            user_input_box['value'] = usr
        except EOFError:
            pass

    # 创建并启动后台线程
    t = threading.Thread(target=_input_reader, daemon=True)
    t.start()

    # 等待线程 t 完成或 timeout 秒
    t.join(timeout)
    if t.is_alive():
        # 用户还没输入，放弃读取
        print(f"\n[INFO] {timeout}秒内无输入，默认选 {default}\n")
        return default
    else:
        # 用户输入了
        ans = user_input_box['value']
        if ans == '':
            print(f"[INFO] 用户输入为空，默认选 {default}\n")
            return default
        return ans

# ------------------------ Evaluation ---------------------------------------------------------------------------------
# PowerEfficiencyScore 函数
def PowerEfficiencyScore(results, connections):
    scores = []
    for idx, (input_map, output_map, nfu, longest_time_expr) in enumerate(results, start=1):
        # Build neuron_to_gnc mapping from the NFU object
        neuron_to_gnc = {}
        for gnc_id in range(16):
            for neuron in nfu.get_GNC(gnc_id).contains:
                neuron_to_gnc[neuron] = gnc_id

        # Calculate total Manhattan distance of inter-GNC connections
        total_distance = 0
        for (src_neuron, tgt_neuron, _w) in connections:
            src_g = neuron_to_gnc.get(src_neuron)
            tgt_g = neuron_to_gnc.get(tgt_neuron)
            if src_g is None or tgt_g is None:
                continue  # Skip if neuron not found (shouldn't happen in valid mapping)
            if src_g != tgt_g:
                (x1, y1), (x2, y2) = gnc_coordinates[src_g], gnc_coordinates[tgt_g]
                dist = abs(x1 - x2) + abs(y1 - y2)
                total_distance += dist

        score = total_distance if total_distance > 0 else -1
        scores.append(score)
        print(f"Mapping result {idx}: Total Manhattan distance = {score}")

    return scores  # Return scores without plotting

# UtilizationBalanceScore 函数
def UtilizationBalanceScore(results):
    scores = []
    for idx, (input_map, output_map, nfu, longest_time_expr) in enumerate(results, start=1):
        # Count neurons in each of the 16 clusters
        usage_count = [0] * 16
        for gnc_id in range(16):
            count = len(nfu.get_GNC(gnc_id).contains)
            usage_count[gnc_id] = count
        total_neurons = sum(usage_count)
        if total_neurons == 0:
            score = -1  # no neurons mapped
            scores.append(score)
            print(f"Mapping result {idx}: No neurons mapped (score = -1)")
            continue
        # Check capacity constraint
        capacity = 8192
        if any(count > capacity for count in usage_count):
            score = 1e9  # heavy penalty for overload
            scores.append(score)
            print(f"Mapping result {idx}: Cluster overload detected (score = {score})")
            continue
        # Compute standard deviation of usage ratios
        usage_ratios = [count / capacity for count in usage_count]
        mean_ratio = sum(usage_ratios) / len(usage_ratios)
        variance = sum((r - mean_ratio) ** 2 for r in usage_ratios) / len(usage_ratios)
        std_dev = math.sqrt(variance)
        score = std_dev
        scores.append(score)
        print(f"Mapping result {idx}: Neurons per GNC = {usage_count}")
        print(f" -> Utilization std. dev. = {std_dev:.4f}")
    return scores  # Return scores without plotting


# LatencyEfficiencyScore 函数
def LatencyEfficiencyScore(results, connections):
    scores = []
    for idx, (input_map, output_map, nfu, longest_time_expr) in enumerate(results, start=1):
        # print(f"\n[INFO] Processing mapping result {idx}...")

        # Build neuron_to_gnc map
        neuron_to_gnc = {}
        for gnc_id in range(16):
            for neuron in nfu.get_GNC(gnc_id).contains:
                neuron_to_gnc[neuron] = gnc_id

        # print(f"[DEBUG] neuron_to_gnc map: {neuron_to_gnc}")

        # Build directed graph (adjacency list) with Manhattan distance weights
        graph = {}
        indegree = {}
        for (src, tgt, _w) in connections:
            src_g = neuron_to_gnc.get(src)
            tgt_g = neuron_to_gnc.get(tgt)
            if src_g is None or tgt_g is None:
                continue  # skip if neuron mapping missing (should not happen)
            (x1, y1), (x2, y2) = gnc_coordinates[src_g], gnc_coordinates[tgt_g]
            weight = abs(x1 - x2) + abs(y1 - y2)
            if src not in graph:
                graph[src] = []
            graph[src].append((tgt, weight))
            indegree[src] = indegree.get(src, 0)
            indegree[tgt] = indegree.get(tgt, 0) + 1

        # print(f"[DEBUG] Graph: {graph}")
        # print(f"[DEBUG] In-degrees: {indegree}")

        # Initialize distance for longest path computation
        distance = {}
        for neuron, indeg in indegree.items():
            distance[neuron] = 0 if indeg == 0 else float('-inf')
        topo_queue = [node for node, indeg in indegree.items() if indeg == 0]

        # print(f"[DEBUG] Initial distance map: {distance}")
        # print(f"[DEBUG] Initial topo_queue: {topo_queue}")

        # Topological order traversal to compute longest distances
        while topo_queue:
            u = topo_queue.pop(0)
            if u in graph:
                for v, w in graph[u]:
                    if distance[u] + w > distance.get(v, float('-inf')):
                        distance[v] = distance[u] + w
                    indegree[v] -= 1
                    if indegree[v] == 0:
                        topo_queue.append(v)

        # print(f"[DEBUG] Distance after topological sorting: {distance}")

        # Determine longest path to any output neuron
        longest_distance = 0
        for out_neuron in output_map.keys():
            if out_neuron in distance and distance[out_neuron] > longest_distance:
                longest_distance = distance[out_neuron]

        print(f"[DEBUG] Longest distance: {longest_distance}")
        score = longest_distance if longest_distance > 0 else -1
        scores.append(score)
        print(f"Mapping result {idx}: Longest Manhattan path = {score}")

    return scores  # Return scores without plotting

def LatencyEfficiencyScore_NoPrune(
    results,
    connections,
    gamma=0.9,             # 强度加权指数（0.7~1.0 推荐做灵敏度）
    compute_time=None      # dict: 节点计算时延；不传则 0
):
    """
    返回两列分数（顺序与 results 一一对应）：
      - ub_list : 纯结构上界（最长曼哈顿路径 + 可选计算时延）
      - ew_list : 基于“正权占比”的强度加权最长路（非概率、无剪枝）

    说明：
      * UB 无任何假设，最稳妥。
      * EW 通过入边正权占比衰减边时延：w_eff = t_comm * (frac_pos_weight^gamma)，
        仅借助静态权重做“倾向性”区分；不代表发放概率。
      * 图要求近似无环（前馈）；若存在环，将给出告警并返回 -1（无法在有向有环图上定义最长路）。
    """
    compute_time = compute_time or {}

    def build_graph(neuron_to_gnc):
        graph = defaultdict(list)      # u -> [(v, t_comm, w)]
        in_edges = defaultdict(list)   # v -> [(u, t_comm, w)]
        indeg = {}
        for (src, tgt, w) in connections:
            gs = neuron_to_gnc.get(src)
            gt = neuron_to_gnc.get(tgt)
            if gs is None or gt is None:
                continue
            (x1, y1), (x2, y2) = gnc_coordinates[gs], gnc_coordinates[gt]
            t_comm = abs(x1 - x2) + abs(y1 - y2)
            graph[src].append((tgt, t_comm, w))
            in_edges[tgt].append((src, t_comm, w))
            indeg[src] = indeg.get(src, 0)
            indeg[tgt] = indeg.get(tgt, 0) + 1
        return graph, in_edges, indeg

    def topo_order(graph, indeg):
        indeg_tmp = indeg.copy()
        Q = deque([n for n, d in indeg_tmp.items() if d == 0])
        order = []
        while Q:
            u = Q.popleft()
            order.append(u)
            for (v, _t, _w) in graph.get(u, []):
                indeg_tmp[v] -= 1
                if indeg_tmp[v] == 0:
                    Q.append(v)
        # 有环：无法得到完整拓扑序
        cyclic = len(order) < len(indeg_tmp)
        return order, cyclic

    def edge_time(u, t_comm):
        return t_comm + float(compute_time.get(u, 0.0))

    ub_list, ew_list = [], []

    for idx, (input_map, output_map, nfu, _longest_unused) in enumerate(results, start=1):
        # 1) neuron -> GNC
        neuron_to_gnc = {}
        for g in range(16):
            for n in nfu.get_GNC(g).contains:
                neuron_to_gnc[n] = g

        # 2) 构图
        graph, in_edges, indeg = build_graph(neuron_to_gnc)
        order, cyclic = topo_order(graph, indeg)

        out_neurons = set(output_map.keys()) if isinstance(output_map, dict) else set(output_map)

        if cyclic:
            print(f"[WARN] Mapping {idx}: detected cycles; longest-path on general digraph is ill-defined. Return -1.")
            ub_list.append(-1.0); ew_list.append(-1.0); continue

        # 3) 纯结构上界（UB）
        dist = {n: (0.0 if indeg.get(n, 0) == 0 else float('-inf')) for n in indeg}
        indeg_tmp = indeg.copy()
        Q = deque([n for n, d in indeg_tmp.items() if d == 0])
        while Q:
            u = Q.popleft()
            for (v, t_comm, _w) in graph.get(u, []):
                w_eff = edge_time(u, t_comm)
                if dist[u] + w_eff > dist.get(v, float('-inf')):
                    dist[v] = dist[u] + w_eff
                indeg_tmp[v] -= 1
                if indeg_tmp[v] == 0:
                    Q.append(v)
        ub = max([dist.get(o, float('-inf')) for o in out_neurons] or [0.0])
        if ub == float('-inf'):
            ub = -1.0

        # 4) 强度加权临界路径（EW，非概率）
        #   对每个目标 v，计算其入边正权和 Spos_v；入边强度占比 frac = max(0,w_uv)/Spos_v
        #   边有效时延 = edge_time * (frac ^ gamma)
        #   * 若 Spos_v=0（只有抑制），则该 v 的入边强度为 0（不贡献路径）
        Spos = {}
        for v, ins in in_edges.items():
            s = 0.0
            for (_u, _t, w) in ins:
                if w > 0.0:
                    s += w
            Spos[v] = s

        # 用“强度过滤”的图与入度（不剪枝，只是 0 强度的边不会贡献）
        indegE = {n: 0 for n in indeg.keys()}
        graphE = defaultdict(list)
        for u, outs in graph.items():
            for (v, t_comm, w) in outs:
                if Spos.get(v, 0.0) <= 0.0 or w <= 0.0:
                    # 目标没有任何正权输入，或此边非正权：不给强度贡献
                    continue
                frac = w / Spos[v]   # ∈ (0,1]
                w_eff = edge_time(u, t_comm) * (frac ** gamma)
                graphE[u].append((v, w_eff))
                indegE[v] = indegE.get(v, 0) + 1
                indegE[u] = indegE.get(u, 0)

        # 若由于全是非正权入边导致 graphE 为空，退化为 UB（给出 -1 或 UB）
        if not graphE:
            ew = -1.0
        else:
            distE = {n: (0.0 if indegE.get(n, 0) == 0 else float('-inf')) for n in indegE}
            Q = deque([n for n, d in indegE.items() if d == 0])
            indeg_tmp = indegE.copy()
            while Q:
                u = Q.popleft()
                for (v, w_eff) in graphE.get(u, []):
                    if distE[u] + w_eff > distE.get(v, float('-inf')):
                        distE[v] = distE[u] + w_eff
                    indeg_tmp[v] -= 1
                    if indeg_tmp[v] == 0:
                        Q.append(v)
            ew = max([distE.get(o, float('-inf')) for o in out_neurons] or [0.0])
            if ew == float('-inf'):
                ew = -1.0

        ub_list.append(ub)
        ew_list.append(ew)
        print(f"[NoPrune] Mapping {idx}: UB={ub:.3f}, EW={ew:.3f}")

    return ub_list, ew_list

def PathHopStatsAndUB(
    results,
    connections,
    compute_time=None,           # 可选：dict{node: t_comp}；不传则 0
    save_hist=True,
    hist_fname_prefix="hops_hist",
    model_tag=None,              # 把 MODEL_TAG 作为参数传入
    include_intra_core=False     # 仅用于“直方图/统计”：是否包含 hop=0 的同核边
):
    """
    对每个 mapping 输出：
      - ub_list : 纯结构 UB（最长曼哈顿路径；可叠加源节点计算时延）
      - hop_stats : list[dict]，含 {'mean','median','max','count'}
      - cut_stats : list[dict]，含 {'cut_total_vert','cut_total_horz',
                                     'cut_perlink_vert','cut_perlink_horz','W','H'}
      - 并保存每个 mapping 的“边 hop 分布”条形图（文件名与工程风格一致）

    重要修复：
      * UB 永远在“完整依赖图”（包含 hop=0 的同核边）上计算，以避免断开连通性导致 UB=-1 的问题。
      * 直方图/均值/中位数仍按 include_intra_core 控制是否排除 hop=0。
    """
    compute_time = compute_time or {}

    # ---- 统一拿 GNC 坐标（与工程一致） ----
    def _get_gnc_coords():
        coords = globals().get('gnc_coordinates') or globals().get('GNC_COORD')
        if coords is None:
            # 与项目中相同的 4×4 mesh 坐标
            coords = {
                0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
                4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
                8:(3,1), 9:(2,1), 10:(1,1), 11:(0,1),
                12:(3,0), 13:(2,0), 14:(1,0), 15:(0,0),
            }
        return coords

    def edge_time(u, hop):  # 可把源节点计算时延并到出边
        return float(hop) + float(compute_time.get(u, 0.0))

    def build_maps(nfu):
        neuron_to_gnc = {}
        for g in range(16):
            for n in nfu.get_GNC(g).contains:
                neuron_to_gnc[n] = g
        return neuron_to_gnc

    # 在一次 build_graph 中：
    # - 搭建图 & 入度（可选择是否包含 hop=0）
    # - 统计每条边的 hop 列表
    # - 统计“几何割”计数（Lx/Ly）
    # 返回：graph, indeg, hops, Lx, Ly, W, H
    def build_graph(neuron_to_gnc, include_intra):
        graph = defaultdict(list)   # u -> [(v, hop)]
        indeg = {}
        hops = []

        coords = _get_gnc_coords()
        xs = [xy[0] for xy in coords.values()]
        ys = [xy[1] for xy in coords.values()]
        W = max(xs) + 1
        H = max(ys) + 1

        Lx = [0]*(W-1)   # 竖直割计数：列 0|1, 1|2, 2|3
        Ly = [0]*(H-1)   # 水平割计数：行 0|1, 1|2, 2|3

        for (u, v, _w) in connections:
            gu, gv = neuron_to_gnc.get(u), neuron_to_gnc.get(v)
            if gu is None or gv is None:
                continue
            (x1, y1), (x2, y2) = coords[gu], coords[gv]
            hop = abs(x1 - x2) + abs(y1 - y2)  # ★ 4×4 mesh 的曼哈顿“跳数”

            # 图结构 & hop 收集：是否包含 hop=0 由 include_intra 控制
            if include_intra or hop > 0:
                graph[u].append((v, hop))
                indeg[u] = indeg.get(u, 0)
                indeg[v] = indeg.get(v, 0) + 1
                hops.append(hop)

            # ★ 几何“割”计数（路由无关）：端点在割两侧 ⇒ 必跨割一次
            # 竖直割：介于列 k 与 k+1（k=0..W-2）
            if x1 < x2:
                for k in range(x1, x2):
                    Lx[k] += 1
            elif x2 < x1:
                for k in range(x2, x1):
                    Lx[k] += 1
            # 水平割：介于行 l 与 l+1（l=0..H-2）
            if y1 < y2:
                for l in range(y1, y2):
                    Ly[l] += 1
            elif y2 < y1:
                for l in range(y2, y1):
                    Ly[l] += 1

        return graph, indeg, hops, Lx, Ly, W, H

    def _gnc_longest_hop(input_map, output_map, nfu, connections):
        # 1) 神经元 -> GNC
        neuron_to_gnc = {}
        for g in range(16):
            for n in nfu.get_GNC(g).contains:
                neuron_to_gnc[n] = g

        # 2) 建 GNC 级 DAG（带 hop 权重）
        coords = _get_gnc_coords()
        import networkx as nx
        Gm = nx.DiGraph()
        for (u, v, _w) in connections:
            gu = neuron_to_gnc.get(u);
            gv = neuron_to_gnc.get(v)
            if gu is None or gv is None or gu == gv:  # 同核边不计 hop
                continue
            hop = abs(coords[gu][0] - coords[gv][0]) + abs(coords[gu][1] - coords[gv][1])
            # 多重边在最长路上没区别，保留一条即可；若想保留可叠加到属性里
            if Gm.has_edge(gu, gv):
                # 可选：取更大的 hop（但在 mesh 上 hop 固定），或计数到 'mult'
                pass
            else:
                Gm.add_edge(gu, gv, hop=hop)

        # 3) S/T：来自 input_map / output_map 的 GNC 集合，兜底用入度0/出度0
        def to_gnc_set(map_or_iter):
            if isinstance(map_or_iter, dict):
                it = map_or_iter.keys()
            else:
                it = map_or_iter
            return {neuron_to_gnc[n] for n in it if n in neuron_to_gnc}

        S = to_gnc_set(input_map)
        T = to_gnc_set(output_map)
        if not S:
            S = {u for u in Gm.nodes if Gm.in_degree(u) == 0}
        if not T:
            T = {u for u in Gm.nodes if Gm.out_degree(u) == 0}

        # 4) 仅保留 S 可达且能达 T 的子图（防止孤点干扰）
        reachable_from_S = set()
        for s in S:
            reachable_from_S |= nx.descendants(Gm, s) | {s}
        can_reach_T = set()
        Grev = Gm.reverse()
        for t in T:
            can_reach_T |= nx.descendants(Grev, t) | {t}
        sub_nodes = reachable_from_S & can_reach_T
        Gs = Gm.subgraph(sub_nodes).copy()

        if not nx.is_directed_acyclic_graph(Gs):
            return -1.0  # 与原约定一致：有环返回 -1

        # 5) 加权最长路（hop 权）
        path = nx.dag_longest_path(Gs, weight="hop")
        if len(path) <= 1:
            return 0.0
        ub = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ub += Gs[u][v]['hop']
        return float(ub)

    def median_of(lst):
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        mid = n // 2
        return float(s[mid]) if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])

    # ==== 主循环 ====
    ub_list = []
    hop_stats = []
    cut_stats = []

    for idx, (input_map, output_map, nfu, _longest_unused) in enumerate(results, start=1):
        neuron_to_gnc = build_maps(nfu)

        # A) 直方图/统计视角：可选排除 hop=0
        graph_hist, indeg_hist, hops, Lx, Ly, W, H = build_graph(neuron_to_gnc, include_intra_core)

        # B) UB 视角：始终包含 hop=0，保证连通性完整
        graph_ub,   indeg_ub,   _hops2, _Lx2, _Ly2, _W2, _H2 = build_graph(neuron_to_gnc, True)

        # 选出 UB 用的输出节点集合：优先使用 output_map 中存在于 UB 图里的节点
        out_nodes_given = set(output_map.keys()) if isinstance(output_map, dict) else set(output_map)
        out_nodes_ub = [o for o in out_nodes_given if o in indeg_ub]
        if not out_nodes_ub:
            # 回退：把 UB 图里的汇点（无出边）当作输出
            out_nodes_ub = [n for n in indeg_ub.keys() if n not in graph_ub or len(graph_ub.get(n, [])) == 0]

        # 1) UB（最长路）
        ub = _gnc_longest_hop(input_map, output_map, nfu, connections)
        ub_list.append(ub)

        # 2) 边 hop 统计（来自 graph_hist / hops）
        count = len(hops)
        mean_hop = (sum(hops) / count) if count else 0.0
        med_hop = median_of(hops)
        max_hop = max(hops) if hops else 0
        hop_stats.append({"mean": mean_hop, "median": med_hop, "max": max_hop, "count": count})

        print(f"[Edge-Hop Stats] Mapping {idx}: "
              f"UB={ub:.3f}, mean={mean_hop:.3f}, median={med_hop:.3f}, max={max_hop}, edges={count}")

        # 3) 几何割统计（SCD-C 相关；与 include_intra_core 无关）
        cut_total_vert = max(Lx) if Lx else 0
        cut_total_horz = max(Ly) if Ly else 0
        cut_perlink_vert = (cut_total_vert / H) if H > 0 else 0.0  # 一条竖直割上并行链路 = H
        cut_perlink_horz = (cut_total_horz / W) if W > 0 else 0.0  # 一条水平割上并行链路 = W

        cut_stats.append({
            "cut_total_vert": int(cut_total_vert),
            "cut_total_horz": int(cut_total_horz),
            "cut_perlink_vert": float(cut_perlink_vert),
            "cut_perlink_horz": float(cut_perlink_horz),
            "W": int(W), "H": int(H)
        })

        print(f"[Cut Stats] Mapping {idx}: "
              f"V-cut max={cut_total_vert} (per-link={cut_perlink_vert:.2f}), "
              f"H-cut max={cut_total_horz} (per-link={cut_perlink_horz:.2f})")

        # 4) 画条形图（每个跳数的边数量）——使用“统计视角”的 hops
        if save_hist and count > 0:
            # 统计每个整数 hop 的计数
            freq = defaultdict(int)
            for h in hops:
                freq[int(h)] += 1
            xs = sorted(freq.keys())
            ys = [freq[x] for x in xs]

            # 文件名：与工程风格一致；优先使用 out_path(ts_name(...))
            base_name = f"{hist_fname_prefix}_{model_tag}_map{idx}" if model_tag else f"{hist_fname_prefix}_map{idx}"
            try:
                fname = str(out_path(ts_name(base_name)))
            except Exception:
                fname = f"{base_name}.png"

            plt.figure(figsize=(8, 4.5))
            plt.bar(xs, ys, width=0.8)
            plt.xlabel("GNC Manhattan hops")
            plt.ylabel("#Edges")
            title_tag = f" · {model_tag}" if model_tag else ""
            plt.title(f"Edge Hop Distribution · Mapping {idx}{title_tag}")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"[INFO] Edge-hop histogram saved: {fname}")

    return ub_list, hop_stats, cut_stats

def AvgPathHops(results, connections, *, merge_parallel_edges=False):
    """
    对每个映射 M 计算 GNC 级 DAG 上的“路径均值”（AvgPathHops）：
      AvgPathHops = (所有 S->T 有向路径的总 hop 和) / (S->T 路径条数)
    其中 hop 为 GNC 网格的曼哈顿距离；同核连接不计入（仅统计跨 GNC）。

    参数:
      - results: [(input_map, output_map, nfu, _), ...]（与你现有结果结构一致）
      - connections: [(u, v, w), ...] 原始神经元级边
      - merge_parallel_edges: 是否把同一 (GNC->GNC) 的多条神经元边去重合并（推荐 True）

    返回:
      - avg_list: 每个映射的 AvgPathHops（float）组成的列表；若无路径则为 0.0；
                  若 GNC 级图出现环则返回 -1.0（和 UB 的约定保持一致）
    """

    def _get_gnc_coords():
        coords = globals().get('gnc_coordinates') or globals().get('GNC_COORD')
        if coords is None:
            coords = {
                0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
                4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
                8:(3,1), 9:(2,1), 10:(1,1), 11:(0,1),
                12:(3,0), 13:(2,0), 14:(1,0), 15:(0,0),
            }
        return coords

    def _build_neuron_to_gnc(nfu):
        m = {}
        for g in range(16):
            for n in nfu.get_GNC(g).contains:
                m[n] = g
        return m

    def _build_gnc_dag(neuron_to_gnc):
        """构造 GNC 级 DAG：返回 (adj, radj, indeg, nodes, edge_list, is_dag)。"""
        coords = _get_gnc_coords()
        nodes = set(neuron_to_gnc.values())
        indeg = {g: 0 for g in nodes}
        adj, radj = {g: [] for g in nodes}, {g: [] for g in nodes}
        edge_set, edge_list = set(), []

        for (u, v, _w) in connections:
            gu = neuron_to_gnc.get(u); gv = neuron_to_gnc.get(v)
            if gu is None or gv is None or gu == gv:
                continue
            hop = abs(coords[gu][0] - coords[gv][0]) + abs(coords[gu][1] - coords[gv][1])
            if merge_parallel_edges:
                key = (gu, gv)
                if key in edge_set:
                    continue
                edge_set.add(key)
            edge_list.append((gu, gv, hop))

        for (gu, gv, hop) in edge_list:
            adj.setdefault(gu, []).append((gv, hop))
            radj.setdefault(gv, []).append((gu, hop))
            if gu not in indeg: indeg[gu] = 0
            if gv not in indeg: indeg[gv] = 0
            indeg[gv] += 1
            nodes.add(gu); nodes.add(gv)
        for g in nodes:
            adj.setdefault(g, []); radj.setdefault(g, [])

        # 拓扑检查（检测环）
        indeg_tmp = indeg.copy()
        Q = deque([g for g, d in indeg_tmp.items() if d == 0])
        seen = 0
        while Q:
            x = Q.popleft(); seen += 1
            for (y, _h) in adj.get(x, []):
                indeg_tmp[y] -= 1
                if indeg_tmp[y] == 0:
                    Q.append(y)
        is_dag = (seen == len(indeg))
        return adj, radj, indeg, nodes, edge_list, is_dag

    def _to_gnc_set(neuron_set_or_map, neuron_to_gnc):
        if isinstance(neuron_set_or_map, dict):
            neurons = list(neuron_set_or_map.keys())
        elif isinstance(neuron_set_or_map, (list, set, tuple)):
            neurons = list(neuron_set_or_map)
        else:
            neurons = []
        return {neuron_to_gnc[n] for n in neurons if n in neuron_to_gnc}

    avg_list = []

    for (input_map, output_map, nfu, _unused) in results:
        neuron_to_gnc = _build_neuron_to_gnc(nfu)
        adj, radj, indeg, nodes, edge_list, is_dag = _build_gnc_dag(neuron_to_gnc)

        # S/T：优先用原始输入/输出神经元所在 GNC；兜底为入度0 / 出度0 的 GNC
        S = _to_gnc_set(input_map, neuron_to_gnc)
        T = _to_gnc_set(output_map, neuron_to_gnc)
        if not S:
            S = {g for g in nodes if indeg.get(g, 0) == 0}
        if not T:
            outdeg = {g: len(adj.get(g, [])) for g in nodes}
            T = {g for g in nodes if outdeg.get(g, 0) == 0}

        if not S or not T:
            avg_list.append(0.0); continue
        if not is_dag:
            avg_list.append(-1.0); continue  # 与 UB 的约定一致：有环时返回 -1

        # —— 计算路径计数 F/B（一次拓扑）——
        indeg_tmp = indeg.copy()
        from collections import deque as _dq
        Q = _dq([g for g, d in indeg_tmp.items() if d == 0])
        topo = []
        while Q:
            x = Q.popleft(); topo.append(x)
            for (y, _h) in adj.get(x, []):
                indeg_tmp[y] -= 1
                if indeg_tmp[y] == 0:
                    Q.append(y)

        F = {g: 0 for g in nodes}
        for s in S: F[s] = 1
        for u in topo:
            for (v, _h) in adj.get(u, []):
                F[v] = F.get(v, 0) + F.get(u, 0)

        B = {g: 0 for g in nodes}
        for t in T: B[t] = 1
        for u in reversed(topo):
            for (v, _h) in adj.get(u, []):
                B[u] = B.get(u, 0) + B.get(v, 0)

        N_paths = sum(F.get(t, 0) for t in T)
        if N_paths == 0:
            avg_list.append(0.0); continue

        Sigma_len = 0
        for (u, v, hop) in edge_list:
            Sigma_len += hop * F.get(u, 0) * B.get(v, 0)

        avg_list.append(Sigma_len / N_paths)

    return avg_list


def EdgeHopHistogramFlex(results, connections, *,
                         weight='multiplicity',    # 'multiplicity' | 'unique' | 'payload'
                         payload_default=1.0,      # 当 weight='payload' 且 w 无法解析时的兜底
                         include_zero_hop=False,   # 是否把同核 0-hop 计入直方图（默认不计）
                         grid_coords=None):
    """
    统计每个映射 M 的跨 GNC 边 hop 分布（计数与占比），支持三种权重：
      - 'multiplicity'：每条神经元边计 1（不去重，最不稀疏，反映通信“次数/机会”）
      - 'unique'      ：同一 (GNC->GNC) 只计 1（结构本地化，不受并行边影响）
      - 'payload'     ：把 connections 的第三列当作负载（字节/比特）权重后再按 hop 聚合

    返回：与 results 对齐的列表，每项为 dict：
      {
        'counts': OrderedDict{hop: weight_sum, ...},        # 各 hop 的权重和
        'proportions': OrderedDict{hop: ratio, ...},        # 各 hop 占比，sum=1（当 total>0）
        'total_weight': float,                              # 全部 hop 的总权重
        'hops': [hop1, hop2, ...],                          # 升序 hop 列表
      }
    """
    def _get_coords():
        if grid_coords is not None:
            return grid_coords
        coords = globals().get('gnc_coordinates') or globals().get('GNC_COORD')
        if coords is None:
            # 默认 4x4（按你项目的坐标习惯来；与 UB 计算一致即可）
            coords = {
                0:(3,3), 1:(2,3), 2:(1,3), 3:(0,3),
                4:(3,2), 5:(2,2), 6:(1,2), 7:(0,2),
                8:(3,1), 9:(2,1), 10:(1,1), 11:(0,1),
                12:(3,0), 13:(2,0), 14:(1,0), 15:(0,0),
            }
        return coords

    def _build_neuron_to_gnc(nfu):
        m = {}
        for g in range(16):
            for n in nfu.get_GNC(g).contains:
                m[n] = g
        return m

    data_list = []
    coords = _get_coords()

    for (_in_map, _out_map, nfu, _unused) in results:
        neuron_to_gnc = _build_neuron_to_gnc(nfu)
        hop_weight = defaultdict(float)
        seen_unique = set()

        for edge in connections:
            # 允许 connections 是 (u,v) 或 (u,v,w)；w 作为 payload
            if len(edge) == 2:
                u, v = edge
                w = payload_default
            else:
                u, v, w = edge[0], edge[1], edge[2]

            gu = neuron_to_gnc.get(u); gv = neuron_to_gnc.get(v)
            if gu is None or gv is None:
                continue
            hop = abs(coords[gu][0] - coords[gv][0]) + abs(coords[gu][1] - coords[gv][1])

            if hop == 0 and not include_zero_hop:
                continue  # 默认不统计同核 0-hop

            if weight == 'unique':
                key = (gu, gv)
                if key in seen_unique:
                    continue
                seen_unique.add(key)
                hop_weight[hop] += 1.0
            elif weight == 'payload':
                # 尝试把 w 解析为数值（比特/字节），不行就用兜底
                try:
                    val = float(w)
                except Exception:
                    val = float(payload_default)
                hop_weight[hop] += val
            else:  # 'multiplicity'
                hop_weight[hop] += 1.0

        if hop_weight:
            hops_sorted = sorted(hop_weight.keys())
            total = sum(hop_weight[h] for h in hops_sorted)
            proportions = OrderedDict((h, hop_weight[h]/total) for h in hops_sorted)
            counts = OrderedDict((h, hop_weight[h]) for h in hops_sorted)
        else:
            hops_sorted = []
            total = 0.0
            proportions = OrderedDict()
            counts = OrderedDict()

        data_list.append({
            'counts': counts,
            'proportions': proportions,
            'total_weight': total,
            'hops': hops_sorted,
        })

    return data_list

def plot_edge_hop_hist_per_mapping(names, hist_data_list, model_tag="", out_dir_func=None, ylabel="Proportion (%)"):
    """ 每个映射单独一张占比柱状图（不稀疏；支持任何权重口味） """
    for name, data in zip(names, hist_data_list):
        hops = list(data['hops'])
        props = [data['proportions'][h] for h in hops] if hops else []
        total = data['total_weight']

        plt.figure(figsize=(5.5, 3.6), dpi=160)
        if hops:
            plt.bar([str(h) for h in hops], [p*100 for p in props])
            for x, p in zip(range(len(hops)), props):
                plt.text(x, p*100, f"{p*100:.1f}%", ha='center', va='bottom', fontsize=9)
        else:
            plt.text(0.5, 0.5, "No cross-GNC edges", ha='center', va='center', transform=plt.gca().transAxes)

        plt.title(f"Hop Histogram · {name} · {model_tag}")
        plt.xlabel("Hop length (Manhattan)")
        plt.ylabel(ylabel)
        plt.ylim(0, 100)
        plt.tight_layout()

        fname = f"hop_hist_{name.replace(' ','_')}_{model_tag}.png" if model_tag else f"hop_hist_{name.replace(' ','_')}.png"
        path = out_dir_func(fname) if out_dir_func else os.path.join(os.getcwd(), fname)
        plt.savefig(path); plt.close()
        print(f"[INFO] Saved hop histogram: {path} (total_weight={total:.0f})")

def plot_edge_hop_hist_combined(names, hist_data_list, model_tag="", out_dir_func=None, ylabel="Proportion (%)"):
    """ 多映射合并对比（分组柱状图，占比%） """
    all_hops = sorted(set(h for d in hist_data_list for h in d['hops']))
    if not all_hops:
        print("[WARN] No cross-GNC edges across all mappings; skip combined plot.")
        return

    props_mat = np.array([[d['proportions'].get(h, 0.0) for h in all_hops] for d in hist_data_list])
    num_maps, num_bins = props_mat.shape
    x = np.arange(num_bins)
    bw = min(0.8 / max(1, num_maps), 0.25)

    plt.figure(figsize=(max(6.0, 1.1*num_bins), 3.8), dpi=160)
    for i, name in enumerate(names):
        plt.bar(x + (i - (num_maps-1)/2)*bw, props_mat[i]*100, width=bw, label=name)

    plt.xticks(x, [str(h) for h in all_hops])
    plt.xlabel("Hop length (Manhattan)")
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.title(f"Hop Histogram · Combined · {model_tag}")
    plt.legend(fontsize=9, ncol=min(num_maps, 3))
    plt.tight_layout()

    fname = f"hop_hist_combined_{model_tag}.png" if model_tag else "hop_hist_combined.png"
    path = out_dir_func(fname) if out_dir_func else os.path.join(os.getcwd(), fname)
    plt.savefig(path); plt.close()
    print(f"[INFO] Saved combined hop histogram: {path}")


# 绘制图表
def plot_results(scores, title, xlabel, ylabel, filename):
    if len(scores) > 1:
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(scores)), scores, color='orange')  # 这里可以选择其他颜色
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(range(len(scores)), [f"Result{i + 1}" for i in range(len(scores))])
        plt.savefig(filename)  # Save the figure as a .png file
        plt.close()  # Close the plot to free up memory


import time, math, random
from statistics import mean, pstdev
from typing import Callable, Dict, Any, List, Tuple

def test_convergence(alg_fn: Callable,
                     neuron_id_map: Dict[str, List[int]],
                     connections: List[Tuple[int, int, float]],
                     *,
                     max_iter_list=(500, 1000, 1500, 2000, 2500),
                     repeats: int = 5,
                     epsilon: float = 1e-4,
                     plateau_gap: int = 50) -> Dict[str, Any]:
    """
    评估单个映射算法的收敛性（已去除 TLE 判定）:
      alg_fn        : 要测试的函数对象 (例如 cluster, cluster_pso)
      neuron_id_map : 网络结构
      connections   : 连接列表
      max_iter_list : 对带迭代算法, 依次给它们不同迭代上限
      repeats       : 随机算法每个 max_iter 重复运行次数
      epsilon       : plateau 判定阈值
      plateau_gap   : 取 Δ 步窗口
    返回: metrics 字典
    """
    results = {}

    # ---------- 1) baseline runtime ----------
    t0 = time.time()
    _ = alg_fn(neuron_id_map, connections)
    greedy_time = time.time() - t0
    results["t_baseline"] = greedy_time

    # ---------- 2) stochastic stability & plateau ----------
    cost_samples = []
    plateau_iter_samples = []

    for K in max_iter_list:
        for _ in range(repeats):
            out = alg_fn(neuron_id_map, connections, max_iter=K) \
                  if "max_iter" in alg_fn.__code__.co_varnames else \
                  alg_fn(neuron_id_map, connections)

            *_, longest = out  # longest_time_expression
            try:
                cost_val = float(longest.split()[2])  # 示例解析
            except Exception:
                cost_val = random.random()            # 占位
            cost_samples.append(cost_val)

            # 如果返回 trace，计算 plateau 迭代数
            if isinstance(out[-1], list):
                trace = out[-1]
                plateau_idx = len(trace) - 1
                for i in range(len(trace) - plateau_gap - 1, -1, -1):
                    if abs(trace[plateau_idx] - trace[i]) / max(trace[i], 1e-9) > epsilon:
                        break
                    plateau_idx = i
                plateau_iter_samples.append(plateau_idx)

    # 2a 方差稳定
    μ = mean(cost_samples)
    σ = pstdev(cost_samples) if len(cost_samples) > 1 else 0.0
    results["stochastic_ratio"] = σ / μ if μ else float("inf")

    # 2b plateau
    if plateau_iter_samples:
        results["mean_plateau_iter"] = mean(plateau_iter_samples)
        results["plateau_ok"] = all(it <= 0.8 * max(max_iter_list) for it in plateau_iter_samples)
    else:
        results["plateau_ok"] = None

    # ---------- 3) 合格判定 ----------
    total_units = len(sum(neuron_id_map.values(), [])) + len(connections)
    results["runtime_ok"] = greedy_time < 1e-3 * total_units
    results["stable_ok"]  = results["stochastic_ratio"] < 0.05
    results["converged"]  = results["runtime_ok"] and \
                            (results["stable_ok"] if σ else True) and \
                            (results["plateau_ok"] if plateau_iter_samples else True)
    return results


# ------------------------ Runner --------------------------------------------------------------------------------------


def main():
    # ---------------------- 1) 创建网络实例 ----------------------
    # net = ComplexSNN()
    # net = run_dnn_gui_and_get_network(
    #     total_nrn=1000,    # 你想要的参数
    #     input_nrn=784,
    #     output_nrn=10,
    #     min_layers=1,
    #     max_layers=10
    # )
    # model1 = LeNet_MNIST
    # AlexNet, MobileNet, InceptionV3
    # model1 = ResNet18
    #[AlexNet,ResNet18,LeNet_MNIST,MobileNet,InceptionV3]
    model1 = ResNet18
    # print(f"net 结构:\n{net}\n")
    print(f"net 结构:\n{model1}\n")

    MODEL_TAG = get_model_tag(model1)

    # ---------------------- 2) 自动推断输入尺寸 ----------------------
    print("assign_neuron_ids函数需要指定input_size大小并传入assign_neuron_ids，"
          "目前默认为input_size=(1, 28, 28)")
    print("根据input数据推断input size中...")

    # 4.1) 为神经元分配ID
    # neuron_id_map, total_neurons = assign_select(net, sample_input_size) # 测试阶段注释掉
    neuron_id_map, total_neurons = assign_select(model1, input_size=(1, 28, 28))
    input_neurons_count = len(neuron_id_map['input'])
    print(f"总神经元数: {total_neurons}")
    print(f"Input层神经元数量: {input_neurons_count}")
    print(f"Output层神经元数量: {len(neuron_id_map['output'])}")

    # 4.2) 构建连接关系
    print("build_connections函数需要指定input_size大小并传入assign_neuron_ids，"
          "目前默认为input_size=(1, 28, 28)")
    try:
        # connections = build_connections(net, neuron_id_map, input_size=sample_input_size)
        # connections = build_connections_1d(net, neuron_id_map, input_size=sample_input_size)
        connections = build_connections(model1, neuron_id_map, input_size=(1, 28, 28))
    except IndexError as e:
        print(f"构建连接关系时出错: {e}")
        return
    print(f"总连接数: {len(connections)}")

    # 4.3) 计算 GNC 和 NFU 需求
    minGncInput = math.ceil(input_neurons_count / 8192)
    minGncAll = math.ceil(total_neurons / 8192)
    print(f"minGncInput (ceil(Input Neurons / 8192)): {minGncInput}")
    print(f"minGncAll (ceil(Total Neurons / 8192)): {minGncAll}")

    # 4.4) 检查 NFU 容量
    if minGncAll > 16:
        print("NFU 0 容纳不下所有神经元，需要更多的NFU。")
        return

    # 4.5) 执行聚类
    # ---------------------- 4.5) 执行聚类 ----------------------
    print("\n请选择映射策略：")
    print("  1) 傻瓜法 -- 快速映射，34M神经元2分钟 (cluster)")
    print("  2) 单线程模拟退火 -- 5000次 较慢，34M神经元约4小时 (cluster2complex_simple)")
    print("  3) 多线程模拟退火 -- 慢 (cluster2complex)")
    print("  4) PSO对照组 -- 未知 (cluster_pso)")

    # 10秒不输入，默认选 '1'
    user_choice = ask_user_with_timeout(
        prompt="请输入数字 [1/2/3/4/5/6/7/8/9/10] (10秒后默认选10): ",
        timeout=10,
        default='10'
    )

    if user_choice == '1':
        print("\n[INFO] 选择: 傻瓜法 -- 快速映射")
        profiler = cProfile.Profile()  # 创建 cProfile 实例
        profiler.enable()  # 开始性能分析
        result = cluster(neuron_id_map, connections, relation=1)
        profiler.disable()
        profiler.dump_stats('profile_output.prof')
        # 使用 SnakeViz 可视化分析文件
        print("运行结束，正在启动 SnakeViz 可视化...")
        subprocess.Popen(['python', '-m', 'snakeviz', 'profile_output.prof'])  # 启动 SnakeViz

    elif user_choice == '2':
        print("\n[INFO] 选择: 单线程模拟退火 -- 5000次")
        # result = cluster2complex_single(neuron_id_map, connections, relation=1)
        # result = cluster_sa(neuron_id_map, connections)
        metrics = test_convergence(cluster_pso, neuron_id_map, connections)
        print(metrics)

    elif user_choice == '3':
        print("\n[INFO] 选择: 多线程模拟退火 -- 默认使用6个核心(n_jobs), max_iter=5000, T_init=10.0, T_final=0.01")
        # 再询问用户是否要更改设定
        yes_no = input("是否要修改这些默认设定？(yes/no): ").strip().lower()
        if yes_no == 'yes':
            # 按顺序让用户输入
            try:
                n_jobs = int(input("请输入 n_jobs (默认6): ").strip())
            except:
                n_jobs = 6
            try:
                max_iter = int(input("请输入 max_iter (默认5000): ").strip())
            except:
                max_iter = 5000
            try:
                T_init = float(input("请输入 T_init (默认10.0): ").strip())
            except:
                T_init = 10.0
            try:
                T_final = float(input("请输入 T_final (默认0.01): ").strip())
            except:
                T_final = 0.01

            print(
                f"\n[INFO] 使用自定义并行退火参数: n_jobs={n_jobs}, max_iter={max_iter}, T_init={T_init}, T_final={T_final}")
            result = cluster2complex(
                neuron_id_map,
                connections,
                relation=1,
                n_jobs=n_jobs,
                max_iter=max_iter,
                T_init=T_init,
                T_final=T_final
            )
        else:
            # 不修改，直接用默认
            print("\n[INFO] 不修改默认设定，开始并行模拟退火...")
            result = cluster2complex(
                neuron_id_map,
                connections,
                relation=1,
                n_jobs=6,
                max_iter=5000,
                T_init=10.0,
                T_final=0.01
            )
    elif user_choice == '4':
        print("\n[INFO] 选择: PSO -- 150次")
        result = cluster_pso(neuron_id_map, connections, pop_size=20, max_iter=150, w_usage=1.0, w_edge=0.5, w_dist=0.1)

    elif user_choice == '5':
        print("\n[INFO] 选择: Pacman")
        result = cluster_pacman(neuron_id_map, connections)

    elif user_choice == '6':
        print("\n[INFO] 选择: Sneap")
        result = cluster_sneap(neuron_id_map, connections, max_iter=200000)

    elif user_choice == '7':
        print("\n[INFO] 选择: Spinemap")
        # result = cluster_spinemap(neuron_id_map, connections, pop_size=20, max_iter=200000)
        result = cluster_spinemap(neuron_id_map, connections)

    elif user_choice == '8':
        print("\n[INFO] 选择: Dfsynthesizer")
        result = multi_cluster_greedy_mapping(neuron_id_map, connections)
        print(result)

    elif user_choice == '9':

        print("\n[INFO] 全选(并行执行6种映射算法)...")

        # 1) 将所有映射函数及其参数打包成列表

        tasks = [

            ("cluster", cluster, (neuron_id_map, connections), {'relation': 1}),

            ("cluster_sa", cluster_sa, (neuron_id_map, connections)),

            # ("cluster2complex_single", cluster2complex_single, (neuron_id_map, connections), {'relation': 1}),

            # ("cluster_pso", cluster_pso, (neuron_id_map, connections),
            #
            #  {'pop_size': 20, 'max_iter': 200, 'w_usage': 1.0, 'w_edge': 0.5, 'w_dist': 0.1}),

            # ("cluster_pacman", cluster_pacman, (neuron_id_map, connections), {}),

            # ("cluster_sneap", cluster_sneap, (neuron_id_map, connections), {'max_iter': 100}),

            # ("cluster_spinemap", cluster_spinemap, (neuron_id_map, connections), {'pop_size': 20, 'max_iter': 100}),



        ]

        # 2) 使用ThreadPoolExecutor或ProcessPoolExecutor

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:

            # 定义一个小函数包装一下调用

            def run_mapping_func(name, func, args, kwargs):

                print(f"[DEBUG] Starting {name}...")

                start_t = time.time()

                try:

                    res = func(*args, **kwargs)  # 调用真正的映射函数

                    print(f"[DEBUG] {name} completed successfully.")

                except Exception as e:

                    print(f"[ERROR] {name} raised an exception: {e}")

                    res = None

                end_t = time.time()

                print(f"[DEBUG] {name} finished in {end_t - start_t:.2f} sec.")

                return (name, res)

            # 3) 提交全部映射任务

            future_to_name = {}

            for (name, func, args, kwargs) in tasks:
                fut = executor.submit(run_mapping_func, name, func, args, kwargs)

                future_to_name[fut] = name

            # 4) 收集结果

            results = {}

            for fut in concurrent.futures.as_completed(future_to_name):

                nm = future_to_name[fut]

                try:

                    (name, result_val) = fut.result()

                    results[name] = result_val

                except Exception as e:

                    print(f"[ERROR] {nm} raised an exception: {e}")

                    results[nm] = None

        # 5) 现在 results 字典包含每个算法的result

        #     例如 results["cluster"] => (input_map, output_map, nfu, longest_time_expr)

        print("\n=== 并行映射全部结束, 各算法结果 ===")

        for nm, val in results.items():

            if val is None:

                print(f"{nm} => failed or error.")

            else:

                print(f"{nm} => success. (type={type(val)})")

        # 6) 获取所有有效结果并评分

        all_results_list = [results[nm] for nm in ["cluster", "cluster2complex_single", "cluster_pso", "cluster_pacman",

                                                   "cluster_sneap", "cluster_spinemap"] if results[nm] is not None]

        if all_results_list:

            pscores = PowerEfficiencyScore(all_results_list, connections)

            uscores = UtilizationBalanceScore(all_results_list)

            lscorse = LatencyEfficiencyScore(all_results_list, connections)

            # 7) 保存评分结果并绘制图像

            plot_results(pscores, 'Power Efficiency Comparison', 'Mapping Result', 'Total Manhattan Distance',
                         'pscore0723.png')

            plot_results(uscores, 'Utilization Balance Comparison', 'Mapping Result', 'Std. Dev. of Utilization',
                         'uscore0723.png')

            plot_results(lscorse, 'Latency Efficiency Comparison', 'Mapping Result', 'Longest Path Distance',
                         'lscore0723.png')

        else:

            print("[INFO] No valid mapping results available for scoring.")


    elif user_choice == '10':

        print("\n[INFO] 串行执行6种映射算法...")

        # 1) 将所有映射函数及其参数打包成列表

        tasks = [

            ("cluster", cluster, (neuron_id_map, connections), {'relation': 1}),

            ("cluster_sa", cluster_sa, (neuron_id_map, connections), {}),

            # ("cluster2complex_single", cluster2complex_single, (neuron_id_map, connections), {'relation': 1}),

            # ("cluster_pso", cluster_pso, (neuron_id_map, connections),
            #
            #  {'pop_size': 20, 'max_iter': 200, 'w_usage': 1.0, 'w_edge': 0.5, 'w_dist': 0.1}),

            # ("cluster_pacman", cluster_pacman, (neuron_id_map, connections), {}),

            # ("cluster_sneap", cluster_sneap, (neuron_id_map, connections), {'max_iter': 100}),

            # ("cluster_spinemap", cluster_spinemap, (neuron_id_map, connections), {'pop_size': 20, 'max_iter': 100}),
            ("multi_cluster_greedy_mapping", multi_cluster_greedy_mapping, (neuron_id_map, connections), {})
        ]

        # 2) 串行执行每个映射算法

        results, runtime_dict = {}, {}

        for name, func, args, kwargs in tasks:

            print(f"[DEBUG] Starting {name}...")

            start_t = time.time()

            try:

                result = func(*args, **kwargs)

                results[name] = result

                print(f"[DEBUG] {name} completed successfully.")

            except Exception as e:

                print(f"[ERROR] {name} raised an exception: {e}")

                results[name] = None

            end_t = time.time()
            elapsed = end_t - start_t
            runtime_dict[name] = elapsed
            print(f"[DEBUG] {name} finished in {end_t - start_t:.2f} sec.")

        # 3) 现在 results 字典包含每个算法的result

        #     例如 results["cluster"] => (input_map, output_map, nfu, longest_time_expr)

        print("\n=== 串行映射全部结束, 各算法结果 ===")

        for nm, val in results.items():

            if val is None:

                print(f"{nm} => failed or error.")

            else:

                print(f"{nm} => success. (type={type(val)})")

        # ----------  将耗时写入 exetime.csv  ----------
        # csv_path = Path("exetime_ResNet18_0723.csv")
        csv_path = out_path(ts_name(f"exetime_{MODEL_TAG}", ".csv"))
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Algorithm", "ExecTime_sec"])
                for algo, t in runtime_dict.items():
                    writer.writerow([algo, f"{t:.6f}"])
            print(f"[INFO] 运行时统计已保存到 {csv_path.resolve()}")
        except Exception as e:
            print(f"[ERROR] 写 exetime.csv 失败: {e}")

        # 4) 获取所有有效结果并评分

        valid_results = []  # [(algo_name, result_tuple), ...]
        for nm in ["cluster", "cluster_sa", "cluster_pso",
                   "multi_cluster_greedy_mapping"]:
            res = results.get(nm)
            if res is None:
                print(f"[INFO] {nm} failed — skipped.")
                continue
            _, _, nfu, _ = res
            if nfu is None:
                print(f"[INFO] {nm} produced no valid NFU — skipped.")
                continue
            valid_results.append((nm, res))

        if not valid_results:
            print("[INFO] No valid mapping results available for scoring.")
        else:
            # 把 result_tuple 抽出来供评分函数使用
            result_list = [r for _, r in valid_results]
            names = [name for name, _ in valid_results]

            # PowerEfficiencyScore
            try:
                pscores = PowerEfficiencyScore(result_list, connections)
            except Exception as e:
                print(f"[WARN] PowerEfficiencyScore failed: {e}")
                pscores = None

            # UtilizationBalanceScore
            try:
                uscores = UtilizationBalanceScore(result_list)
            except Exception as e:
                print(f"[WARN] UtilizationBalanceScore failed: {e}")
                uscores = None
            try:
                ub_list, hop_stats, cut_stats = PathHopStatsAndUB(
                    result_list, connections,
                    compute_time=None,
                    save_hist=True,
                    hist_fname_prefix="hops_hist",
                    model_tag=MODEL_TAG,
                    include_intra_core=False
                )
            except Exception as e:
                print(f"[WARN] PathHopStatsAndUB failed: {e}")
                ub_list, hop_stats, cut_stats = None, None, None

            # 5) 保存评分结果并绘制图像（只对非空评分画图）
            try:
                if pscores:
                    plot_results(
                        pscores,
                        f'Power Efficiency Comparison · {MODEL_TAG}',
                        'Mapping Result', 'Total Manhattan Distance',
                        str(out_path(ts_name(f'pscore_{MODEL_TAG}')))
                    )
                    print("[INFO] Power Efficiency 图已输出")
                else:
                    print("[INFO] Power Efficiency 数据为空，跳过绘图")
            except Exception as e:
                print(f"[WARN] 绘制 PowerEfficiency 图失败: {e}")

            try:
                if uscores:
                    plot_results(
                        uscores,
                        f'Utilization Balance Comparison · {MODEL_TAG}',
                        'Mapping Result', 'Std. Dev. of Utilization',
                        str(out_path(ts_name(f'uscore_{MODEL_TAG}')))
                    )
                    print("[INFO] Utilization Balance 图已输出")
                else:
                    print("[INFO] Utilization Balance 数据为空，跳过绘图")
            except Exception as e:
                print(f"[WARN] 绘制 UtilizationBalance 图失败: {e}")

            try:
                avg_path_hops = AvgPathHops(result_list, connections)  # 与 results/UB 使用的结构一致
                # 控制台输出（带算法名）
                for (name, _), val in zip(valid_results, avg_path_hops):
                    print(f"[AvgPathHops] {name}: {val:.3f}")
                # 可视化（柱状图）
                if avg_path_hops:
                    plot_results(
                        avg_path_hops,
                        f'Avg Path Hops · {MODEL_TAG}',
                        'Mapping Result',
                        'Avg S→T Path Hops',
                        str(out_path(ts_name(f'avg_path_hops_{MODEL_TAG}')))
                    )
                    print("[INFO] Avg Path Hops 图已输出")
            except Exception as e:
                print(f"[WARN] AvgPathHops failed: {e}")

            try:
                out_file = lambda f: str(out_path(ts_name(f)))
                # 1) 计重：最不稀疏，贴近“通信次数/机会”
                hop_hist_mult = EdgeHopHistogramFlex(result_list, connections, weight='multiplicity',
                                                     include_zero_hop=False)
                for name, d in zip(names, hop_hist_mult):
                    line = "  ".join(
                        [f"{h}:{d['proportions'][h] * 100:.1f}%({d['counts'][h]:.0f})" for h in d['hops']]) if d[
                        'hops'] else "N/A"
                    print(f"[HopHist-multiplicity] {name}  total={d['total_weight']:.0f}  |  {line}")
                plot_edge_hop_hist_per_mapping(names, hop_hist_mult, model_tag=f"{MODEL_TAG}_mult",
                                               out_dir_func=out_file)
                plot_edge_hop_hist_combined(names, hop_hist_mult, model_tag=f"{MODEL_TAG}_mult", out_dir_func=out_file)

            except Exception as e:
                print(f"[WARN] Hop histogram failed: {e}")

            try:
                any_scd = False
                # 1) SCD-D：UB
                if ub_list:
                    plot_results(
                        ub_list,
                        f'Latency (SCD-D / UB) · {MODEL_TAG}',
                        'Mapping Result', 'Manhattan Critical Path (hops)',
                        str(out_path(ts_name(f'lats_ub_{MODEL_TAG}')))
                    )
                    any_scd = True

                # 2) SCD-H：mean hop（从 hop_stats 派生）
                if hop_stats:
                    scd_h_mean = [s['mean'] for s in hop_stats]
                    plot_results(
                        scd_h_mean,
                        f'SCD-H (Mean hop) · {MODEL_TAG}',
                        'Mapping Result', 'Mean Manhattan Hops per Edge',
                        str(out_path(ts_name(f'scd_h_mean_{MODEL_TAG}')))
                    )
                    # 同时打印表格信息
                    for i, s in enumerate(hop_stats, start=1):
                        print(f"[Table] Map{i}: mean={s['mean']:.3f}, median={s['median']:.3f}, "
                              f"max={s['max']}, edges={s['count']}")
                    any_scd = True

                # 3) SCD-C：cut per-link（从 cut_stats 派生）
                if cut_stats:
                    scd_c_perlink = [
                        max(cs['cut_perlink_vert'], cs['cut_perlink_horz']) for cs in cut_stats
                    ]
                    plot_results(
                        scd_c_perlink,
                        f'SCD-C (Cut per-link) · {MODEL_TAG}',
                        'Mapping Result', 'Mandatory Crossings per Physical Link',
                        str(out_path(ts_name(f'scd_c_{MODEL_TAG}')))
                    )
                    # 打印割统计摘要
                    for i, cs in enumerate(cut_stats, start=1):
                        print(f"[Cut] Map{i}: "
                              f"V-total={cs['cut_total_vert']}, H-total={cs['cut_total_horz']}; "
                              f"V-perlink={cs['cut_perlink_vert']:.2f}, H-perlink={cs['cut_perlink_horz']:.2f}")
                    any_scd = True

                if any_scd:
                    print("[INFO] SCD 指标图已输出（UB / Mean hop / Cut per-link）")
                else:
                    print("[INFO] SCD 数据为空，跳过绘图")

            except Exception as e:
                print(f"[WARN] 绘制 SCD 图失败: {e}")

    else:
        print("\n[INFO] 输入无效，默认使用 傻瓜法(cluster)")
        result = cluster(neuron_id_map, connections, relation=1)

    if len(result) == 4:
        input_mapping, output_mapping, nfu, longest_time_expression = result
    else:
        # 兼容 cluster 的其他版本可能只返回 3 个值
        print("cluster 函数返回内容与预期不符。")
        return

    # output_mapping 为 None 说明 NFU 不足
    if output_mapping is None:
        print("聚类失败，由于NFU容量不足。")
        return

if __name__ == "__main__":
    main()
