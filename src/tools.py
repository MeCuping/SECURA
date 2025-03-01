import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def Turningdim(tensor,dimU = 1,dimV = 2,mode = "Regu"):
    """
    dimU = ...代表转换为多少维度,最后输出的0位置向量,=-1为原始长度
    dimV = ...代表转换为多少维度,最后输出的1位置向量
    mode = Regu/Self 为标准模式（可以提炼低纬信息）和自定义模式（可以更多保留整体信息)
    Regu模式下dimU为最终压缩维度,dimV不会用到
    """
    tensor = tensor.to(float)
    U,S,V = torch.svd(tensor)
    if(mode == "Regu"):
        if(dimU != -1):
            U = U[:,:dimU]
        else:
            pass
        S = torch.diag(S[:dimU])
        return torch.matmul(U,S)
    elif(mode == "Self"):
        if(dimU != -1):
            U = U[:dimU,:]
        else:
            pass
        S = torch.diag(S)
        V = V[:dimV,:]
        return torch.matmul(torch.matmul(U,S),V.T)

def plot_tensor_list(tensor_list, modeDim=2, colors=None):
    """
    tensor_list (list): 张量列表，每个元素是 torch.Tensor 或 numpy.ndarray，形状为 (n_samples, n_features)。
    modeDim (int): 降维模式，2 表示二维绘图，3 表示三维绘图。
    colors (list): 自定义颜色列表，长度与 tensor_list 相同。如果为 None，自动生成颜色。
    """

    if not isinstance(tensor_list, list) or len(tensor_list) == 0:
        raise ValueError("Input must be a non-empty list of tensors.")

    data_list = [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensor_list]

    dim = data_list[0].shape[1]
    if not all(t.shape[1] == dim for t in data_list):
        raise ValueError("All tensors must have the same number of dimensions.")
    if modeDim not in [2, 3]:
        raise ValueError("modeDim must be 2 or 3.")
    if dim < modeDim:
        raise ValueError(f"Tensors must have at least {modeDim} dimensions for plotting.")

    if colors is None:
        num_colors = len(tensor_list)
        colors = [plt.cm.hsv(i / num_colors) for i in range(num_colors)] 
    if len(colors) < len(tensor_list):
        raise ValueError("Number of colors must match the number of tensors in tensor_list.")

    x_min, x_max = min(t[:, 0].min() for t in data_list) - 1, max(t[:, 0].max() for t in data_list) + 1
    y_min, y_max = min(t[:, 1].min() for t in data_list) - 1, max(t[:, 1].max() for t in data_list) + 1
    if modeDim == 3:
        z_min, z_max = min(t[:, 2].min() for t in data_list) - 1, max(t[:, 2].max() for t in data_list) + 1

    plt.figure(figsize=(8, 6))
    if modeDim == 2:
        for i, data in enumerate(data_list):
            plt.scatter(data[:, 0], data[:, 1], c=colors[i], alpha=0.7, edgecolors='k', label=f"Group {i+1}")
        plt.title("2D Tensor List Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    elif modeDim == 3:
        ax = plt.axes(projection='3d')
        for i, data in enumerate(data_list):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors[i], alpha=0.7, edgecolors='k', label=f"Group {i+1}")
        ax.set_title("3D Tensor List Visualization")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    plt.legend()
    plt.grid(True)
    plt.show()

def cos(tensorA,tensorB):
    return nn.functional.cosine_similarity(tensorA,tensorB)

def readGrad(path):
    dicGrad = torch.load(path)
    return dicGrad

#错的梯度算法
def CombineGrad(dicGrad):
    tensorCache = []
    combineDic = {}
    i = 1
    for name,item in dicGrad.items():
        tensorCache.append(item)
        if(i%2 == 0):
            svName = name[:-7]+"Bias"
            combineDic[svName] = torch.matmul(tensorCache[0],tensorCache[1])
            tensorCache = []
        i += 1
    return combineDic

#由于bias会是一个向量所以做一个对角线升维，这样就可以放到多维对比
def DiagUpdim(Vector):
    tensor = torch.diag(Vector)
    return tensor

# #测试
# test = torch.randn([768,768])
# c = torch.randn([768,768])
# a = Turningdim(test,dimU=-1,dimV=3,mode="Self")
# b = Turningdim(c,dimU=3,dimV=2,mode="Regu")
# d = cos(test,c)
# print(b)
# plot_tensor_list([a,b],3)

def GradBiasOut(A, Grad_B):
    # Create a mask for non-zero elements in A
    mask = (A != 0).float()  # Shape: [5]

    # Replace zeros in A with ones to prevent division by zero
    A_safe = A.clone()
    A_safe[A_safe == 0] = 1  # Replace zeros with ones

    # Safely perform element-wise division and zero out invalid positions
    temp = Grad_B / A_safe[:, None]   # Element-wise division
    temp = temp * mask[:, None]      # Zero out positions where A was zero

    # Compute dL/dbiasOut by averaging over valid contributions
    count = torch.sum(mask)          # Number of non-zero elements in A
    Grad_biasOut = torch.sum(temp, dim=0) / count
    return Grad_biasOut
