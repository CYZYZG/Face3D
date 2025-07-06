#include "solvepnp_3.h"


void print_matrix(int rows, int cols, const double* mat) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%10.4f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void print_vector3(const Vector3d* v) {
    printf("[%.4f, %.4f, %.4f]\n", v->data[0], v->data[1], v->data[2]);
}

void print_matrix3(const Matrix3d* m) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%.4f ", m->data[i][j]);
        }
        printf("\n");
    }
}


#define MAX_ITER 100
#define EPS 1e-8

// 计算 Frobenius 范数
double frob_norm(int m, int n, const double* A) {
    double norm = 0.0;
    for (int i = 0; i < m * n; ++i)
        norm += A[i] * A[i];
    return sqrt(norm);
}

// Jacobi SVD: A[m x n] -> U[m x m], S[min(m,n)], V[n x n]
void jacobi_svd(int m, int n, const double* A, double* U, double* S, double* V) {
    // 初始化 U = I, V = I
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            U[i * m + j] = (i == j) ? 1.0 : 0.0;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;

    // Copy A to W
    double* W = (double*)malloc(m * n * sizeof(double));
    memcpy(W, A, m * n * sizeof(double));

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        int converged = 1;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double ap = 0.0, aq = 0.0, apq = 0.0;
                for (int k = 0; k < m; ++k) {
                    ap += W[k * n + p] * W[k * n + p];
                    aq += W[k * n + q] * W[k * n + q];
                    apq += W[k * n + p] * W[k * n + q];
                }

                if (fabs(apq) < EPS) continue;

                // 计算旋转角度
                double theta = (aq - ap) / (2.0 * apq);
                double t;
                if (theta >= 0)
                    t = 1.0 / (theta + sqrt(1.0 + theta * theta));
                else
                    t = -1.0 / (-theta + sqrt(1.0 + theta * theta));
                double c = 1.0 / sqrt(1 + t * t);
                double s = t * c;

                // Rotate columns p and q of W
                for (int k = 0; k < m; ++k) {
                    double a_kp = W[k * n + p];
                    double a_kq = W[k * n + q];
                    W[k * n + p] = a_kp * c - a_kq * s;
                    W[k * n + q] = a_kp * s + a_kq * c;
                }

                // Update V
                for (int k = 0; k < n; ++k) {
                    double v_kp = V[k * n + p];
                    double v_kq = V[k * n + q];
                    V[k * n + p] = v_kp * c - v_kq * s;
                    V[k * n + q] = v_kp * s + v_kq * c;
                }

                // Update U
                for (int k = 0; k < m; ++k) {
                    double u_kp = U[k * m + p];
                    double u_kq = U[k * m + q];
                    U[k * m + p] = u_kp * c - u_kq * s;
                    U[k * m + q] = u_kp * s + u_kq * c;
                }

                converged = 0;
            }
        }

        if (converged) break;
    }

    // Compute singular values from final W
    for (int i = 0; i < (m < n ? m : n); ++i) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k)
            sum += W[k * n + i] * W[k * n + i];
        S[i] = sqrt(sum);
    }

    free(W);
}

void build_A_matrix(int n, const Vector3d pts3d[], const Vector2d pts2d[],
    double fx, double fy, double cx, double cy, double* A) {
    for (int i = 0; i < n; ++i) {
        double X = pts3d[i].data[0], Y = pts3d[i].data[1], Z = pts3d[i].data[2];
        double u = pts2d[i].data[0], v = pts2d[i].data[1];

        int row = i * 2;

        // 第一行
        A[row * 12 + 0] = X * fx;
        A[row * 12 + 1] = Y * fx;
        A[row * 12 + 2] = Z * fx;
        A[row * 12 + 3] = fx;
        A[row * 12 + 4] = 0.0;
        A[row * 12 + 5] = 0.0;
        A[row * 12 + 6] = 0.0;
        A[row * 12 + 7] = 0.0;
        A[row * 12 + 8] = X * cx - u * X;
        A[row * 12 + 9] = Y * cx - u * Y;
        A[row * 12 + 10] = Z * cx - u * Z;
        A[row * 12 + 11] = cx - u;

        // 第二行
        row += 1;
        A[row * 12 + 0] = 0.0;
        A[row * 12 + 1] = 0.0;
        A[row * 12 + 2] = 0.0;
        A[row * 12 + 3] = 0.0;
        A[row * 12 + 4] = X * fy;
        A[row * 12 + 5] = Y * fy;
        A[row * 12 + 6] = Z * fy;
        A[row * 12 + 7] = fy;
        A[row * 12 + 8] = X * cy - v * X;
        A[row * 12 + 9] = Y * cy - v * Y;
        A[row * 12 + 10] = Z * cy - v * Z;
        A[row * 12 + 11] = cy - v;
    }
}


void orthogonalize_rotation(const double* R_bar, double* R) {
    double U[9], S[3], V[9];
    jacobi_svd(3, 3, R_bar, U, S, V);

    // R = U * V^T
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k)
                R[i * 3 + j] += U[i * 3 + k] * V[j * 3 + k];
        }
    }
}

void adjust_sign_based_on_depth(int n, const Vector3d pts3d[], double beta,
    double* R, double* t) {
    int num_positive = 0, num_negative = 0;

    for (int i = 0; i < n; ++i) {
        double X = pts3d[i].data[0];
        double Y = pts3d[i].data[1];
        double Z = pts3d[i].data[2];

        // lambda = beta * (X * R[6] + Y * R[7] + Z * R[8] + t[2])
        double lambda = beta * (X * R[6] + Y * R[7] + Z * R[8] + t[2]);

        if (lambda > 0) num_positive++;
        else num_negative++;
    }

    if (num_negative > num_positive) {
        for (int i = 0; i < 9; ++i) R[i] *= -1; // 反转 R 的所有元素
        for (int i = 0; i < 3; ++i) t[i] *= -1; // 反转 t 的所有元素
    }
}

int solvePnPbyDLT3(const double K[3][3],
    const Vector3d pts3d[6],
    const Vector2d pts2d[6],
    Matrix3d* R,
    Vector3d* t) {

    if (!pts3d || !pts2d || !R || !t) return 0;

    double fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];

    // Step 1: 构建 A 矩阵
    double A[12 * 12] = { 0 };
    build_A_matrix(6, pts3d, pts2d, fx, fy, cx, cy, A);

    // Step 2: SVD 分解 Ax = 0
    double U[12 * 12], S[12], V[12 * 12];
    jacobi_svd(12, 12, A, U, S, V);

    // Step 3: 提取最小奇异值对应的向量作为解
    double sol[12];
    for (int i = 0; i < 12; ++i)
        sol[i] = V[i * 12 + 11];  // 最后一个右奇异向量

    // Step 4: 提取 R_bar 和 t_bar
    double R_bar[9] = {
        sol[0], sol[1], sol[2],
        sol[4], sol[5], sol[6],
        sol[8], sol[9], sol[10]
    };
    double t_bar[3] = { sol[3], sol[7], sol[11] };

    // Step 5: 正交化 R_bar 得到 R
    orthogonalize_rotation(R_bar, R->data[0]);

    // Step 6: 计算 beta
    double beta = 1.0 / ((S[0] + S[1] + S[2]) / 3.0);
    t->data[0] = beta * t_bar[0];
    t->data[1] = beta * t_bar[1];
    t->data[2] = beta * t_bar[2];

    // Step 7: 深度符号判断
    adjust_sign_based_on_depth(6, pts3d, beta, R->data[0], t->data);

    return 1;
}