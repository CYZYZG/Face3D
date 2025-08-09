import math
import numpy as np
import random

def householder_vector(x):
    """
    计算向量 x 的 Householder 变换向量 v 和系数 beta，
    使得 (I - beta * v v^T) x = [±||x||, 0, 0, ...]^T。
    """
    n = len(x)
    # 计算 x[1:] 的平方和
    tail_norm2 = sum(x[i]*x[i] for i in range(1, n))
    v = list(x)  # 复制 x
    if tail_norm2 == 0.0:
        beta = 0.0
    else:
        norm_x = math.sqrt(x[0]*x[0] + tail_norm2)
        # 改良的数值稳定性方案
        if x[0] <= 0:
            v0 = x[0] - norm_x
        else:
            v0 = -tail_norm2 / (x[0] + norm_x)
        beta = 2.0 * v0 * v0 / (tail_norm2 + v0*v0)
        v = [1.0] + [xi / v0 for xi in x[1:]]
    return v, beta

def householder_bidiagonalization(A):
    """
    对 A (m×n, m>=n) 执行 Householder 双对角化。
    返回 U (m×m), B (m×n bidiagonal), Vt (n×n) 满足 U B Vt = A。
    """
    m, n = len(A), len(A[0])
    # 初始化 U = I_m, Vt = I_n，B 为 A 的拷贝
    U = [[1.0 if i==j else 0.0 for j in range(m)] for i in range(m)]
    Vt = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    B = [row[:] for row in A]  # 深拷贝

    for col in range(n):
        # --- 左侧 Householder 反射，消去 B[col+1:, col] ---
        x = [B[i][col] for i in range(col, m)]
        v, beta = householder_vector(x)
        if beta != 0.0:
            # 更新 B 的 col..m, col..n 子矩阵
            for j in range(col, n):
                proj = 0.0
                for i in range(len(v)):
                    proj += v[i] * B[col+i][j]
                proj *= beta
                for i in range(len(v)):
                    B[col+i][j] -= v[i] * proj
            # 更新 U = U * (I - beta v v^T)
            for i in range(m):
                proj = 0.0
                for k in range(col, m):
                    proj += U[i][k] * v[k-col]
                proj *= beta
                for k in range(col, m):
                    U[i][k] -= proj * v[k-col]
        # 强制下三角为 0
        for i in range(col+1, m):
            B[i][col] = 0.0

        # --- 右侧 Householder 反射，消去 B[col, col+2:] ---
        if col < n-1:
            x = [B[col][j] for j in range(col+1, n)]
            v, beta = householder_vector(x)
            if beta != 0.0:
                # 更新 B 的 col..m, col+1..n 子矩阵
                for i in range(col, m):
                    proj = 0.0
                    for k in range(len(v)):
                        proj += B[i][col+1+k] * v[k]
                    proj *= beta
                    for k in range(len(v)):
                        B[i][col+1+k] -= proj * v[k]
                # 更新 Vt = (I - beta v v^T) * Vt
                for i in range(n):
                    proj = 0.0
                    for k in range(len(v)):
                        proj += v[k] * Vt[k+col+1][i]
                    proj *= beta
                    for k in range(len(v)):
                        Vt[k+col+1][i] -= proj * v[k]
            # 强制超对角上方为 0
            for j in range(col+2, n):
                B[col][j] = 0.0

    return U, B, Vt

def jacobi_eigen_decomposition(A, tol=1e-12, max_iter=1000):
    """
    使用 Jacobi 迭代法对对称矩阵 A (n×n) 求特征值和特征向量。
    返回特征值列表和对应的特征向量矩阵（列为特征向量）。
    """
    n = len(A)
    # 初始化 V = I_n
    V = [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    # 工作矩阵 M
    M = [row[:] for row in A]

    for _ in range(max_iter):
        # 找到当前最大绝对值的非对角元素 M[p][q]
        max_val = 0.0
        p = q = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(M[i][j]) > max_val:
                    max_val = abs(M[i][j])
                    p, q = i, j
        if max_val < tol:
            break  # 充分对角化

        # 计算 Jacobi 旋转角度
        if M[p][q] != 0.0:
            tau = (M[q][q] - M[p][p]) / (2.0 * M[p][q])
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau*tau))
            if tau < 0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t*t)
            s = t * c
        else:
            c = 1.0
            s = 0.0

        # 更新对角元素
        app = M[p][p]; aqq = M[q][q]; apq = M[p][q]
        M[p][p] = c*c*app - 2.0*c*s*apq + s*s*aqq
        M[q][q] = s*s*app + 2.0*c*s*apq + c*c*aqq
        M[p][q] = M[q][p] = 0.0

        # 更新其他元素和特征向量矩阵 V
        for i in range(n):
            if i != p and i != q:
                aip = M[i][p]; aiq = M[i][q]
                M[i][p] = M[p][i] = c*aip - s*aiq
                M[i][q] = M[q][i] = c*aiq + s*aip
        for i in range(n):
            vip = V[i][p]; viq = V[i][q]
            V[i][p] = c*vip - s*viq
            V[i][q] = c*viq + s*vip

    eigvals = [M[i][i] for i in range(n)]
    # **确保特征向量列规范化（数值稳定）**
    for j in range(n):
        norm = math.sqrt(sum(V[i][j]*V[i][j] for i in range(n)))
        if norm > 0:
            for i in range(n):
                V[i][j] /= norm
    return eigvals, V

def svd_golub_reinsch(A, full_matrices=True):
    """
    纯 Python 实现 SVD（Golub-Reinsch 算法），返回 U, S, Vt 使得 A = U * diag(S) * Vt。
    full_matrices=True 返回完整 U (m×m)、Vt (n×n)；False 返回精简形 U (m×k)、Vt (k×n)。
    """
    # 转为浮点
    A = [[float(x) for x in row] for row in A]
    m, n = len(A), len(A[0])

    # 如果 m<n，先对 A^T 求 SVD，然后交换 U,V
    if m < n:
        # 转置 A
        At = [[A[i][j] for i in range(m)] for j in range(n)]
        U_t, S_vals, Vt_t = svd_golub_reinsch(At, full_matrices=True)
        # 原 U = Vt_t^T (m×m)
        U = [[Vt_t[j][i] for j in range(len(Vt_t))] for i in range(len(Vt_t))]
        # 原 Vt = U_t^T (n×n)
        Vt = [[U_t[j][i] for j in range(len(U_t))] for i in range(len(U_t))]
        if not full_matrices:
            # 经济形：U (m×m) 不变，Vt 取前 m 行 (m×n)
            Vt = Vt[:m]
        return U, S_vals, Vt

    # 1. 双对角化
    U_bi, B, Vt_bi = householder_bidiagonalization(A)

    # 2. 求 B^T B 的特征值分解
    # 构造对称矩阵 T = B^T B (n×n)
    T = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += B[k][i] * B[k][j]
            T[i][j] = s
    eigvals, eigvecs = jacobi_eigen_decomposition(T)

    # 特征值排序（按绝对值降序）并构造 V_B 矩阵
    eig_pairs = sorted([(eigvals[i], [eigvecs[row][i] for row in range(n)]) for i in range(n)],
                       key=lambda x: abs(x[0]), reverse=True)
    # 对小的负值截断为 0
    eig_pairs = [(val if val>0 else 0.0, vec) for val, vec in eig_pairs]

    # 奇异值为特征值开根号
    S_vals = [math.sqrt(val) for val, _ in eig_pairs]
    # 构造 V_B 矩阵：列向量为特征向量
    V_B = [vec for _, vec in eig_pairs]        # V_B[j] 是第 j 个奇异值对应的右奇异向量
    V_B_mat = [ [V_B[j][i] for j in range(n)] for i in range(n) ]

    # 计算 B 的左奇异向量列：u_i = B * v_i / sigma_i
    U_B_cols = []
    for i, (val, v) in enumerate(eig_pairs):
        if val == 0.0:
            # 对应 sigma=0 的情形，先填零（稍后正交补齐）
            U_B_cols.append([0.0]*m)
        else:
            sigma = math.sqrt(val)
            # B * v 计算
            u_col = []
            for r in range(m):
                s = 0.0
                for c in range(n):
                    s += B[r][c] * v[c]
                u_col.append(s / sigma)
            U_B_cols.append(u_col)

    # 对 U_B_cols 进行正交化（Gram-Schmidt）
    U_B = []
    for u in U_B_cols:
        # 投影扣除
        u_proj = list(u)
        for prev in U_B:
            dotp = sum(prev[k]*u_proj[k] for k in range(m))
            for k in range(m):
                u_proj[k] -= dotp * prev[k]
        norm = math.sqrt(sum(x*x for x in u_proj))
        if norm < 1e-12:
            continue
        U_B.append([x/norm for x in u_proj])

    # 若 full 矩阵，则补足 U_B 的正交基至 m 列
    U_B_full = list(U_B)
    if full_matrices:
        # 使用标准基向量补充
        for idx in range(m):
            if len(U_B_full) >= m:
                break
            e = [0.0]*m
            e[idx] = 1.0
            for prev in U_B_full:
                proj = sum(prev[k]*e[k] for k in range(m))
                for k in range(m):
                    e[k] -= proj * prev[k]
            norm = math.sqrt(sum(x*x for x in e))
            if norm < 1e-12:
                continue
            U_B_full.append([x/norm for x in e])

    # 3. 构造最终 U 和 Vt
    if full_matrices:
        U_final = [[0.0]*m for _ in range(m)]
        # 注意：U_B_full 是“列向量的列表”，U_B_full[j][k] = 第 j 列第 k 个分量
        for i in range(m):
            for j in range(m):
                s = 0.0
                for kk in range(m):
                    s += U_bi[i][kk] * U_B_full[j][kk]   # <--- 这里用 U_B_full[j][kk]
                U_final[i][j] = s
    else:
        kkcols = n  # 经济形取前 n 列
        U_final = [[0.0]*kkcols for _ in range(m)]
        for i in range(m):
            for j in range(kkcols):
                s = 0.0
                for kk in range(m):
                    s += U_bi[i][kk] * U_B_full[j][kk]   # <--- 同样修正
                U_final[i][j] = s

    # V_final^T = V_B^T * Vt_bi  （原代码这部分是正确的）
    Vt_final = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += V_B_mat[k][i] * Vt_bi[k][j]
            Vt_final[i][j] = s

    # ----- 确定性符号约定（有助于和 NumPy 一致） -----
    # 对每个奇异值列，确保 U 在其最大绝对分量处为非负（否则翻转该列 & 对应的 Vt 行）
    for j in range(len(S_vals)):
        # 找到 U_final[:, j] 中绝对值最大的下标
        if full_matrices:
            col = [U_final[i][j] for i in range(m)]
            imax = max(range(m), key=lambda r: abs(col[r]))
            if col[imax] < 0:
                for i in range(m):
                    U_final[i][j] = -U_final[i][j]
                for k in range(n):
                    Vt_final[j][k] = -Vt_final[j][k]
        else:
            col = [U_final[i][j] for i in range(m)]
            imax = max(range(m), key=lambda r: abs(col[r]))
            if col[imax] < 0:
                for i in range(m):
                    U_final[i][j] = -U_final[i][j]
                for k in range(n):
                    Vt_final[j][k] = -Vt_final[j][k]

    return U_final, S_vals, Vt_final


def random_matrix_3x3(min_val=-100000, max_val=100000):
    """生成一个 3x3 的随机浮点矩阵"""
    return [[round(random.uniform(min_val, max_val), 2) for _ in range(3)] for _ in range(3)]

# --- 示例用法 ---
if __name__ == "__main__":
    
    for i in range(10000):
        A = random_matrix_3x3()
        
        # # 示例矩阵 A (3×4)
        # A = [
        #     [100351.40695714, 12910.65833372, -10252.58161795],
        #     [418.45231489, -13982.11551773, 11103.44467584],
        #     [-82718.93688419, 15591.97843369, -12381.86522675]
        # ]
        # print("A\n")
        # print(A)
        #计算 SVD（full_matrices=True）
        U, S, Vt = svd_golub_reinsch(A, full_matrices=True)
        U = np.array(U)
        Vt = np.array(Vt)
        # print("U =")
        # print(U)
        # # print("S =", S)
        # print("Vt =")
        # print(Vt)
        #print("UV =")
        #print(U@Vt)
        UV = U@Vt
        # 验证重构精度
        # （如有 NumPy 环境可验证，此处仅演示输出）
        # print("A\n")
        # print(A)
        u, s, vh = np.linalg.svd(A)
        # print("U =")
        # print(u)
        # # print("S =", s)
        # print("Vt =")
        # print(vh)
        #print("UV =")
        
        #print(u@vh)
        uv = u@vh
        count = 0
        for i in range(3):
            for j in range(3):
                if abs(UV[i,j]) - abs(uv[i,j]) > 1e-10:
                    print(UV[i,j] , uv[i,j])
                    count+=1
        
        if count == 0:
            # print("结果比对一致")
            pass
        else:
            print("Error!", count)
        
