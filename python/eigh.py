import numpy as np
import math

def eig_2x2_analytic_algebraic(A):
    """
    使用 Jacobi 迭代法的代数解析解求解 2x2 对称矩阵的特征值和特征向量。
    
    :param A: 2x2 对称矩阵，**必须是 NumPy 数组**。
    :return: (eigenvalues, eigenvectors)
    """
    # 确保输入是 NumPy 数组，并且维度是 2x2
    if not isinstance(A, np.ndarray) or A.shape != (2, 2) or A[0, 1] != A[1, 0]:
        raise ValueError("Input A must be a 2x2 symmetric NumPy array.")

    # 🌟 修复后的变量提取：直接使用 NumPy 索引
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 1]
    
    # --- 1. 处理特殊情况：已是对角矩阵或标量矩阵 ---
    if abs(b) < 1e-12:
        eigenvalues = np.array([a, c])
        eigenvectors = np.identity(2)
        
    else:
        # --- 2. 求解 t = tan(theta) (代数法) ---
        diff_ac = a - c
        S = diff_ac / (2 * b)
        
        # 选择 t 的绝对值最小的根
        if S >= 0:
            t = -S + math.sqrt(S * S + 1)
        else:
            t = -S - math.sqrt(S * S + 1)
        
        # --- 3. 求解 c = cos(theta) 和 s = sin(theta) ---
        c_denom = math.sqrt(1 + t * t)
        c_val = 1.0 / c_denom
        s_val = t * c_val 

        # --- 4. 计算特征值 (代数关系式) ---
        h = c_val * s_val * b
        lambda1 = a * c_val**2 + c * s_val**2 + 2 * h
        lambda2 = a * s_val**2 + c * c_val**2 - 2 * h
        
        eigenvalues = np.array([lambda1, lambda2])

        # --- 5. 构造特征向量 (旋转矩阵 Q) ---
        eigenvectors = np.array([
            [c_val, -s_val],
            [s_val, c_val]
        ])

    # --- 6. 排序 ---
    if eigenvalues[0] < eigenvalues[1]:
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
    return eigenvalues, eigenvectors


# --- 验证代码 ---
if __name__ == '__main__':
    A_list = [[4, -3], 
              [-3, 8]]
    A_np = np.array(A_list, dtype=float)
    
    # 1. 我们的代数解析解
    eig_vals_my, eig_vecs_my = eig_2x2_analytic_algebraic(A_np)
    
    # 2. NumPy 参考解
    eig_vals_np, eig_vecs_np = np.linalg.eigh(A_np)
    
    # NumPy 通常升序，我们调整为降序
    eig_vals_np = eig_vals_np[::-1]
    eig_vecs_np = eig_vecs_np[:, ::-1]
    
    # --- 验证 ---
    # 重建 A = Q * Lambda * Q.T
    Lambda_my = np.diag(eig_vals_my)
    A_reconstructed = eig_vecs_my @ Lambda_my @ eig_vecs_my.T
    err_recon = np.linalg.norm(A_np - A_reconstructed)
    
    print("--- 2x2 对称矩阵特征分解（代数解）---")
    print("原始矩阵 A:\n", A_np)
    
    print("\n--- 结果 ---")
    print("特征值 (我的):\n", eig_vals_my)
    print("特征向量 (我的):\n", eig_vecs_my)

    print("\n--- 验证 ---")
    print(f"NumPy 特征值 (参考): {eig_vals_np}")
    print(f"重建误差 (||A - Q*Lambda*Q.T||): {err_recon:.2e}")

    # 特征值直接误差
    err_vals = np.linalg.norm(eig_vals_my - eig_vals_np)
    print(f"特征值误差 (L2 Norm): {err_vals:.2e}")
    
    if err_recon < 1e-10:
        print("\n✅ 代数解析解成功，精度极高，且无三角函数依赖。")
    else:
        print("\n❌ 求解失败。")
