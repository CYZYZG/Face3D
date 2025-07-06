#include "solvepnp_2.h"

// ------------------ 基本矩阵操作 ------------------
void matrix_transpose2(const double* A, double* At, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            At[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_multiply2(const double* A, const double* B, double* C,
    int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += MATRIX_AT(A, i, j, n) * MATRIX_AT(B, j, k, p);
            }
            MATRIX_AT(C, i, k, p) = sum;
        }
    }
}

double matrix_frobenius_norm2(const double* A, int rows, int cols) {
    double norm = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        norm += A[i] * A[i];
    }
    return sqrt(norm);
}

// ------------------ SVD 实现 (完整版) ------------------
// 基于Golub-Reinsch算法实现完整SVD
int svd(double* A, int m, int n, double* U, double* S, double* V) {
    const int max_iter = 100;
    const double eps = DBL_EPSILON;
    const double tol = 1e-12;

    double* work = (double*)malloc(n * sizeof(double));
    double* e = (double*)malloc(n * sizeof(double));
    double* temp = (double*)malloc(m * sizeof(double));

    // 双对角化初始化
    int i, j, k, iter;
    double c, f, g, h, s, scale, t, x, y, z;
    int l = 0; // 初始化l

    g = scale = 0.0;
    x = 0.0;

    for (i = 0; i < n; i++) {
        e[i] = g;
        s = 0.0;
        l = i + 1;

        for (j = i; j < m; j++) {
            s += A[j * n + i] * A[j * n + i];
        }

        if (s < tol) {
            g = 0.0;
        }
        else {
            f = A[i * n + i];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            A[i * n + i] = f - g;

            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = i; k < m; k++) {
                    s += A[k * n + i] * A[k * n + j];
                }
                f = s / h;
                for (k = i; k < m; k++) {
                    A[k * n + j] += f * A[k * n + i];
                }
            }
        }

        S[i] = g;
        s = 0.0;

        for (j = l; j < n; j++) {
            s += A[i * n + j] * A[i * n + j];
        }

        if (s < tol) {
            g = 0.0;
        }
        else {
            f = A[i * n + i + 1];
            g = (f < 0.0) ? sqrt(s) : -sqrt(s);
            h = f * g - s;
            A[i * n + i + 1] = f - g;

            for (j = l; j < n; j++) {
                e[j] = A[i * n + j] / h;
            }

            for (j = l; j < m; j++) {
                s = 0.0;
                for (k = l; k < n; k++) {
                    s += A[j * n + k] * A[i * n + k];
                }
                for (k = l; k < n; k++) {
                    A[j * n + k] += s * e[k];
                }
            }
        }

        y = fabs(S[i]) + fabs(e[i]);
        if (y > scale) scale = y;
    }

    // 累加右变换
    for (i = n - 1; i >= 0; i--) {
        if (g != 0.0) {
            h = A[i * n + i + 1] * g;
            for (j = l; j < n; j++) {
                V[j * n + i] = A[i * n + j] / h;
            }
            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = l; k < n; k++) {
                    s += A[i * n + k] * V[k * n + j];
                }
                for (k = l; k < n; k++) {
                    V[k * n + j] += s * V[k * n + i];
                }
            }
        }
        for (j = l; j < n; j++) {
            V[i * n + j] = V[j * n + i] = 0.0;
        }
        V[i * n + i] = 1.0;
        g = e[i];
        l = i;
    }

    // 累加左变换
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = S[i];
        for (j = l; j < n; j++) {
            A[i * n + j] = 0.0;
        }
        if (g != 0.0) {
            g = 1.0 / g;
            for (j = l; j < n; j++) {
                s = 0.0;
                for (k = l; k < m; k++) {
                    s += A[k * n + i] * A[k * n + j];
                }
                f = (s / A[i * n + i]) * g;
                for (k = i; k < m; k++) {
                    A[k * n + j] += f * A[k * n + i];
                }
            }
            for (j = i; j < m; j++) {
                A[j * n + i] *= g;
            }
        }
        else {
            for (j = i; j < m; j++) {
                A[j * n + i] = 0.0;
            }
        }
        A[i * n + i] += 1.0;
    }

    // 对角化双对角形式
    for (k = n - 1; k >= 0; k--) {
        for (iter = 0; iter < max_iter; iter++) {
            int cancel = 0; // 初始化cancel
            for (l = k; l >= 0; l--) {
                if (l == 0) { // 防止越界
                    cancel = 1;
                    break;
                }
                if (fabs(e[l]) <= eps * scale) {
                    cancel = 1;
                    break;
                }
                if (fabs(S[l - 1]) <= eps * scale) {
                    break;
                }
            }

            if (!cancel) {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * e[i];
                    e[i] = c * e[i];
                    if (fabs(f) <= eps * scale) break;
                    g = S[i];
                    h = hypot(f, g);
                    S[i] = h;
                    c = g / h;
                    s = -f / h;

                    for (j = 0; j < m; j++) {
                        y = A[j * n + l - 1];
                        z = A[j * n + i];
                        A[j * n + l - 1] = y * c + z * s;
                        A[j * n + i] = -y * s + z * c;
                    }
                }
            }

            z = S[k];
            if (l == k) {
                if (z < 0.0) {
                    S[k] = -z;
                    for (j = 0; j < n; j++) {
                        V[j * n + k] = -V[j * n + k];
                    }
                }
                break;
            }

            if (iter == max_iter - 1) {
                free(work);
                free(e);
                free(temp);
                return 0; // 未收敛
            }

            x = S[l];
            y = S[k - 1];
            g = e[k - 1];
            h = e[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = hypot(f, 1.0);
            f = ((x - z) * (x + z) + h * (y / (f + SIGN(g, f)) - h)) / x;

            c = s = 1.0;
            for (j = l; j < k; j++) {
                i = j + 1;
                g = e[i];
                y = S[i];
                h = s * g;
                g = c * g;
                z = hypot(f, h);
                e[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y = y * c;

                for (int jj = 0; jj < n; jj++) {
                    x = V[jj * n + j];
                    z = V[jj * n + i];
                    V[jj * n + j] = x * c + z * s;
                    V[jj * n + i] = -x * s + z * c;
                }

                z = hypot(f, h);
                S[j] = z;
                if (fabs(z) > eps) {
                    c = f / z;
                    s = h / z;
                }
                f = c * g + s * y;
                x = -s * g + c * y;

                for (int jj = 0; jj < m; jj++) {
                    y = A[jj * n + j];
                    z = A[jj * n + i];
                    A[jj * n + j] = y * c + z * s;
                    A[jj * n + i] = -y * s + z * c;
                }
            }
            e[l] = 0.0;
            e[k] = f;
            S[k] = x;
        }
    }

    // 复制结果到U和V
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            U[i * n + j] = A[i * n + j];
        }
    }

    free(work);
    free(e);
    free(temp);
    return 1;
}

// ------------------ 3x3 SVD 专用实现 ------------------
void svd_3x3(const double* M, double* U, double* S, double* V) {
    double A[9];
    memcpy(A, M, 9 * sizeof(double));

    double Ut[9], Vt[9];
    double S_temp[3];

    // 使用完整SVD实现
    int result = svd(A, 3, 3, Ut, S_temp, Vt);
    if (!result) {
        // 简单回退：单位矩阵
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                U[i * 3 + j] = (i == j) ? 1.0 : 0.0;
                V[i * 3 + j] = (i == j) ? 1.0 : 0.0;
            }
            S[i] = 1.0;
        }
        return;
    }

    // 转置以获得正确的U和V
    matrix_transpose2(Ut, U, 3, 3);
    matrix_transpose2(Vt, V, 3, 3);

    // 确保旋转矩阵是右手系
    double detU = U[0] * (U[4] * U[8] - U[5] * U[7]) - U[1] * (U[3] * U[8] - U[5] * U[6]) + U[2] * (U[3] * U[7] - U[4] * U[6]);
    double detV = V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);

    if (detU < 0) {
        for (int i = 0; i < 9; i++) U[i] = -U[i];
        S_temp[2] = -S_temp[2];
    }

    if (detV < 0) {
        for (int i = 0; i < 9; i++) V[i] = -V[i];
        S_temp[2] = -S_temp[2];
    }

    // 复制奇异值
    S[0] = S_temp[0];
    S[1] = S_temp[1];
    S[2] = S_temp[2];
}

// ------------------ PnP 主函数 ------------------
int solvePnPbyDLT2(const Matrix3d* K,
    const Vector3d* pts3d,
    const Vector2d* pts2d,
    int num_points,
    Matrix3d* R,
    Vector3d* t) {
    // 检查输入
    if (num_points < 6) {
        fprintf(stderr, "Error: At least 6 points are required\n");
        return 0;
    }

    // 获取相机内参
    const double fx = K->data[0];
    const double fy = K->data[4];
    const double cx = K->data[2];
    const double cy = K->data[5];

    // 构建矩阵 A (2n x 12)
    const int A_rows = 2 * num_points;
    const int A_cols = 12;
    double* A = (double*)malloc(A_rows * A_cols * sizeof(double));

    // 初始化为0
    memset(A, 0, A_rows * A_cols * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        const Vector3d pt3d = pts3d[i];
        const Vector2d pt2d = pts2d[i];

        const double x = pt3d.x;
        const double y = pt3d.y;
        const double z = pt3d.z;
        const double u = pt2d.x;
        const double v = pt2d.y;

        // 第一行 (u 方程)
        int row = 2 * i;
        MATRIX_AT(A, row, 0, A_cols) = x * fx;
        MATRIX_AT(A, row, 1, A_cols) = y * fx;
        MATRIX_AT(A, row, 2, A_cols) = z * fx;
        MATRIX_AT(A, row, 3, A_cols) = fx;
        MATRIX_AT(A, row, 8, A_cols) = x * cx - u * x;
        MATRIX_AT(A, row, 9, A_cols) = y * cx - u * y;
        MATRIX_AT(A, row, 10, A_cols) = z * cx - u * z;
        MATRIX_AT(A, row, 11, A_cols) = cx - u;

        // 第二行 (v 方程)
        row = 2 * i + 1;
        MATRIX_AT(A, row, 4, A_cols) = x * fy;
        MATRIX_AT(A, row, 5, A_cols) = y * fy;
        MATRIX_AT(A, row, 6, A_cols) = z * fy;
        MATRIX_AT(A, row, 7, A_cols) = fy;
        MATRIX_AT(A, row, 8, A_cols) = x * cy - v * x;
        MATRIX_AT(A, row, 9, A_cols) = y * cy - v * y;
        MATRIX_AT(A, row, 10, A_cols) = z * cy - v * z;
        MATRIX_AT(A, row, 11, A_cols) = cy - v;
    }

    // 执行完整SVD分解: A = U * S * V^T
    double* U_svd = (double*)malloc(A_rows * A_rows * sizeof(double));
    double* S_svd = (double*)malloc(A_cols * sizeof(double));
    double* V_svd = (double*)malloc(A_cols * A_cols * sizeof(double));

    // 复制A矩阵，因为svd会修改输入矩阵
    double* A_copy = (double*)malloc(A_rows * A_cols * sizeof(double));
    memcpy(A_copy, A, A_rows * A_cols * sizeof(double));

    int svd_result = svd(A_copy, A_rows, A_cols, U_svd, S_svd, V_svd);
    free(A_copy);
    free(A);

    if (!svd_result) {
        fprintf(stderr, "SVD computation failed\n");
        free(U_svd);
        free(S_svd);
        free(V_svd);
        return 0;
    }

    // 提取最小奇异值对应的右奇异向量（最后一列）
    double solution[12];
    for (int i = 0; i < 12; i++) {
        solution[i] = MATRIX_AT(V_svd, i, 11, A_cols);
    }

    free(U_svd);
    free(S_svd);
    free(V_svd);

    // 提取投影矩阵参数
    double a1 = solution[0], a2 = solution[1], a3 = solution[2], a4 = solution[3];
    double a5 = solution[4], a6 = solution[5], a7 = solution[6], a8 = solution[7];
    double a9 = solution[8], a10 = solution[9], a11 = solution[10], a12 = solution[11];

    // 构建 R_bar
    double R_bar[9] = {
        a1, a2, a3,
        a5, a6, a7,
        a9, a10, a11
    };

    // 对 R_bar 进行完整的SVD分解
    double U_R[9], S_R[3], V_R[9];
    svd_3x3(R_bar, U_R, S_R, V_R);

    // 计算正交旋转矩阵 R = U_R * V_R^T
    double Vt_R[9];
    matrix_transpose2(V_R, Vt_R, 3, 3);

    double R_temp[9];
    matrix_multiply2(U_R, Vt_R, R_temp, 3, 3, 3);

    // 复制到输出矩阵
    memcpy(R->data, R_temp, 9 * sizeof(double));

    // 计算尺度因子 beta
    double beta = 3.0 / (S_R[0] + S_R[1] + S_R[2]);

    // 计算平移向量
    t->x = beta * a4;
    t->y = beta * a8;
    t->z = beta * a12;

    // 深度一致性检查
    int num_positive = 0;
    int num_negative = 0;

    for (int i = 0; i < num_points; i++) {
        const Vector3d pt3d = pts3d[i];
        double lambda = beta * (a9 * pt3d.x + a10 * pt3d.y + a11 * pt3d.z + a12);

        if (lambda >= 0) num_positive++;
        else num_negative++;
    }

    // 符号校正
    if (num_positive < num_negative) {
        for (int i = 0; i < 9; i++) R->data[i] = -R->data[i];
        t->x = -t->x;
        t->y = -t->y;
        t->z = -t->z;
    }

    return 1;
}