#include "solvepnp_4.h"

// ------------------ ����������� ------------------
void matrix_transpose(const double* A, double* At, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            At[j * rows + i] = A[i * cols + j];
        }
    }
}

void matrix_multiply(const double* A, const double* B, double* C,
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

double matrix_frobenius_norm(const double* A, int rows, int cols) {
    double norm = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        norm += A[i] * A[i];
    }
    return sqrt(norm);
}

// ------------------ Jacobi SVD ʵ�� ------------------
void jacobi_svd(double* A, int m, int n, double* U, double* S, double* V) {
    // ��ʼ��U��VΪ��λ����
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            U[i * m + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // ����A����������
    double* B = (double*)malloc(m * n * sizeof(double));
    memcpy(B, A, m * n * sizeof(double));

    double prev_norm = 0.0;
    int converged = 0;
    int iter = 0;

    while (iter < MAX_ITER && !converged) {
        double off_norm = 0.0;

        // ��������������Ԫ�أ��������Խ��ߣ�
        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                // ����2x2�Ӿ���
                double apq = 0.0, app = 0.0, aqq = 0.0;
                for (int i = 0; i < m; i++) {
                    double bip = MATRIX_AT(B, i, p, n);
                    double biq = MATRIX_AT(B, i, q, n);
                    apq += bip * biq;
                    app += bip * bip;
                    aqq += biq * biq;
                }

                off_norm += 2 * apq * apq;

                // ���Ԫ���㹻С��������ת
                if (fabs(apq) < TOLERANCE) continue;

                // ������ת�Ƕ�
                double beta = (app - aqq) / (2.0 * apq);
                double t = SIGN(1.0, beta) / (fabs(beta) + sqrt(beta * beta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;

                // ��BӦ����ת
                for (int i = 0; i < m; i++) {
                    double bip = MATRIX_AT(B, i, p, n);
                    double biq = MATRIX_AT(B, i, q, n);
                    MATRIX_AT(B, i, p, n) = c * bip - s * biq;
                    MATRIX_AT(B, i, q, n) = s * bip + c * biq;
                }

                // ��VӦ����ת
                for (int i = 0; i < n; i++) {
                    double vip = MATRIX_AT(V, i, p, n);
                    double viq = MATRIX_AT(V, i, q, n);
                    MATRIX_AT(V, i, p, n) = c * vip - s * viq;
                    MATRIX_AT(V, i, q, n) = s * vip + c * viq;
                }
            }
        }

        // ���������
        if (iter > 0 && fabs(prev_norm - off_norm) < TOLERANCE) {
            converged = 1;
        }
        prev_norm = off_norm;
        iter++;
    }

    // ��������ֵ
    for (int j = 0; j < n; j++) {
        double norm = 0.0;
        for (int i = 0; i < m; i++) {
            norm += MATRIX_AT(B, i, j, n) * MATRIX_AT(B, i, j, n);
        }
        S[j] = sqrt(norm);
    }

    // ����U
    for (int j = 0; j < n; j++) {
        if (fabs(S[j]) > TOLERANCE) {
            double inv_s = 1.0 / S[j];
            for (int i = 0; i < m; i++) {
                MATRIX_AT(U, i, j, n) = MATRIX_AT(B, i, j, n) * inv_s;
            }
        }
    }

    free(B);
}

// ------------------ 3x3 Jacobi SVD ר��ʵ�� ------------------
void jacobi_svd_3x3(const double* M, double* U, double* S, double* V) {
    double A[9];
    memcpy(A, M, 9 * sizeof(double));

    // ʹ��Jacobi SVDʵ��
    jacobi_svd(A, 3, 3, U, S, V);

    // ȷ����ת����������ϵ
    double detU = U[0] * (U[4] * U[8] - U[5] * U[7]) - U[1] * (U[3] * U[8] - U[5] * U[6]) + U[2] * (U[3] * U[7] - U[4] * U[6]);
    double detV = V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);

    if (detU < 0) {
        for (int i = 0; i < 9; i++) U[i] = -U[i];
        S[2] = -S[2];
    }

    if (detV < 0) {
        for (int i = 0; i < 9; i++) V[i] = -V[i];
        S[2] = -S[2];
    }
}

// ------------------ PnP ������ ------------------
int solvePnPbyDLT4(const Matrix3d* K,
    const Vector3d* pts3d,
    const Vector2d* pts2d,
    int num_points,
    Matrix3d* R,
    Vector3d* t) {
    // �������
    if (num_points < 6) {
        fprintf(stderr, "Error: At least 6 points are required\n");
        return 0;
    }

    // ��ȡ����ڲ�
    const double fx = K->data[0];
    const double fy = K->data[4];
    const double cx = K->data[2];
    const double cy = K->data[5];

    // �������� A (2n x 12)
    const int A_rows = 2 * num_points;
    const int A_cols = 12;
    double* A = (double*)malloc(A_rows * A_cols * sizeof(double));

    // ��ʼ��Ϊ0
    memset(A, 0, A_rows * A_cols * sizeof(double));

    for (int i = 0; i < num_points; i++) {
        const Vector3d pt3d = pts3d[i];
        const Vector2d pt2d = pts2d[i];

        const double x = pt3d.x;
        const double y = pt3d.y;
        const double z = pt3d.z;
        const double u = pt2d.x;
        const double v = pt2d.y;

        // ��һ�� (u ����)
        int row = 2 * i;
        MATRIX_AT(A, row, 0, A_cols) = x * fx;
        MATRIX_AT(A, row, 1, A_cols) = y * fx;
        MATRIX_AT(A, row, 2, A_cols) = z * fx;
        MATRIX_AT(A, row, 3, A_cols) = fx;
        MATRIX_AT(A, row, 8, A_cols) = x * cx - u * x;
        MATRIX_AT(A, row, 9, A_cols) = y * cx - u * y;
        MATRIX_AT(A, row, 10, A_cols) = z * cx - u * z;
        MATRIX_AT(A, row, 11, A_cols) = cx - u;

        // �ڶ��� (v ����)
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

    // ִ��Jacobi SVD�ֽ�
    double* U_svd = (double*)malloc(A_rows * A_rows * sizeof(double));
    double* S_svd = (double*)malloc(A_cols * sizeof(double));
    double* V_svd = (double*)malloc(A_cols * A_cols * sizeof(double));

    jacobi_svd(A, A_rows, A_cols, U_svd, S_svd, V_svd);
    free(A);

    // ��ȡ��С����ֵ��Ӧ�����������������һ�У�
    double solution[12];
    for (int i = 0; i < 12; i++) {
        solution[i] = MATRIX_AT(V_svd, i, 11, A_cols);
    }

    free(U_svd);
    free(S_svd);
    free(V_svd);

    // ��ȡͶӰ�������
    double a1 = solution[0], a2 = solution[1], a3 = solution[2], a4 = solution[3];
    double a5 = solution[4], a6 = solution[5], a7 = solution[6], a8 = solution[7];
    double a9 = solution[8], a10 = solution[9], a11 = solution[10], a12 = solution[11];

    // ���� R_bar
    double R_bar[9] = {
        a1, a2, a3,
        a5, a6, a7,
        a9, a10, a11
    };

    // �� R_bar ����Jacobi SVD�ֽ�
    double U_R[9], S_R[3], V_R[9];
    jacobi_svd_3x3(R_bar, U_R, S_R, V_R);

    // ����������ת���� R = U_R * V_R^T
    double Vt_R[9];
    matrix_transpose(V_R, Vt_R, 3, 3);

    double R_temp[9];
    matrix_multiply(U_R, Vt_R, R_temp, 3, 3, 3);

    // ���Ƶ��������
    memcpy(R->data, R_temp, 9 * sizeof(double));

    // ����߶����� beta
    double beta = 3.0 / (S_R[0] + S_R[1] + S_R[2]);

    // ����ƽ������
    t->x = beta * a4;
    t->y = beta * a8;
    t->z = beta * a12;

    // ���һ���Լ��
    int num_positive = 0;
    int num_negative = 0;

    for (int i = 0; i < num_points; i++) {
        const Vector3d pt3d = pts3d[i];
        double lambda = beta * (a9 * pt3d.x + a10 * pt3d.y + a11 * pt3d.z + a12);

        if (lambda >= 0) num_positive++;
        else num_negative++;
    }

    // ����У��
    if (num_positive < num_negative) {
        for (int i = 0; i < 9; i++) R->data[i] = -R->data[i];
        t->x = -t->x;
        t->y = -t->y;
        t->z = -t->z;
    }

    return 1;
}
