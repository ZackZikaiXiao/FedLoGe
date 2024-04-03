import numpy as np
import torch
# 构造满秩矩阵
while True:
    matrix = np.random.rand(512, 512)
    rank = np.linalg.matrix_rank(matrix)
    if rank == 512:
        break
# Q就是我们的旋转矩阵，它可以用来旋转500维空间中的向量# 假设X是一个[100，500]的矩阵，包含100个500维向量
Q,_= np.linalg.qr(matrix)
rotation_matrix = torch.from_numpy(Q).to(torch.float32)  # 指定device为cuda
# transpose_matrix = np.transpose(Q)
# product = np.matmul(transpose_matrix, Q)
# identity_matrix = np.eye(512)

# # 检查矩阵是否是正交矩阵
# is_orthogonal = np.allclose(product, identity_matrix)
X = torch.rand(100, 512)
X_rotated =torch.matmul(X, rotation_matrix)
angle_degs = []
for x in range(100):
    norm1 = np.linalg.norm(X[x,:])

    # 计算向量2的模长
    norm2 = np.linalg.norm(X_rotated[x,:])

    # 计算向量1和向量2的点积
    dot_product = np.dot(X[x,:], X_rotated[x,:])

    # 计算夹角（弧度）
    angle_rad = np.arccos(dot_product / (norm1 * norm2))
    angle_deg = np.degrees(angle_rad)
    angle_degs.append(angle_deg)
    print(angle_deg)
# 将弧度转换为角度
print(np.mean(angle_degs))
print(np.var(angle_degs))
