// svd_golub_reinsch.c
// Pure C implementation of Golub-Reinsch SVD (Householder bidiagonalization + Jacobi eigen)
// Includes a deterministic LCG RNG that matches the Python example.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ----------------- Utilities for matrix allocation / deallocation ----------------- */

double *alloc_vector(int n)
{
    double *p = (double *)calloc((size_t)n, sizeof(double));
    if (!p)
    {
        fprintf(stderr, "alloc_vector failed\n");
        exit(1);
    }
    return p;
}

double **alloc_matrix(int rows, int cols)
{
    double **A = (double **)malloc(rows * sizeof(double *));
    if (!A)
    {
        fprintf(stderr, "alloc_matrix failed\n");
        exit(1);
    }
    double *data = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (!data)
    {
        fprintf(stderr, "alloc_matrix data failed\n");
        exit(1);
    }
    for (int i = 0; i < rows; ++i)
        A[i] = data + (size_t)i * cols;
    return A;
}

void free_matrix(double **A)
{
    if (!A)
        return;
    free(A[0]); // data
    free(A);
}

void copy_matrix_into(double **src, double **dst, int rows, int cols)
{
    // assumes dst allocated with same layout
    memcpy(dst[0], src[0], sizeof(double) * (size_t)rows * cols);
}

/* ----------------- Deterministic LCG RNG (matches Python implementation) ------------- */
static uint32_t next_rand = 0;
void srand_custom(uint32_t seed) { next_rand = seed; }
int rand_custom(void)
{
    next_rand = next_rand * 1103515245u + 12345u;
    return (int)((next_rand >> 16) & 0x7FFF);
}
double rand_uniform(void) { return rand_custom() / 32768.0; }

/* ----------------- Householder vector computation ----------------- */
/*
 Given x (length n), compute v and beta s.t. (I - beta v v^T) x = [±||x||, 0, 0, ...]^T.
 Return newly allocated v (length n). Caller must free v. Beta returned by pointer.
*/
double *householder_vector(const double *x, int n, double *beta)
{
    double *v = alloc_vector(n);
    double tail_norm2 = 0.0;
    for (int i = 1; i < n; ++i)
        tail_norm2 += x[i] * x[i];
    if (tail_norm2 == 0.0)
    {
        *beta = 0.0;
        for (int i = 0; i < n; ++i)
            v[i] = x[i];
    }
    else
    {
        double norm_x = sqrt(x[0] * x[0] + tail_norm2);
        double v0;
        if (x[0] <= 0.0)
            v0 = x[0] - norm_x;
        else
            v0 = -tail_norm2 / (x[0] + norm_x);
        *beta = 2.0 * v0 * v0 / (tail_norm2 + v0 * v0);
        v[0] = 1.0;
        for (int i = 1; i < n; ++i)
            v[i] = x[i] / v0;
    }
    return v;
}

/* ----------------- Householder bidiagonalization -----------------
 Produces U (m x m), B (m x n), Vt (n x n) such that U * B * Vt = A
 All outputs are newly allocated matrices (with contiguous storage); caller must free them.
*/
void householder_bidiagonalization(double **A, int m, int n,
                                   double ***outU, double ***outB, double ***outVt)
{
    double **U = alloc_matrix(m, m);
    double **Vt = alloc_matrix(n, n);
    double **B = alloc_matrix(m, n);
    // initialize
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            U[i][j] = (i == j) ? 1.0 : 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            Vt[i][j] = (i == j) ? 1.0 : 0.0;
    copy_matrix_into(A, B, m, n);

    for (int col = 0; col < n; ++col)
    {
        // Left Householder: zero out B[col+1..m-1][col]
        int len = m - col;
        double *x = alloc_vector(len);
        for (int i = 0; i < len; ++i)
            x[i] = B[col + i][col];
        double beta;
        double *v = householder_vector(x, len, &beta);
        if (beta != 0.0)
        {
            // Update B's submatrix rows col..m-1, cols col..n-1
            for (int j = col; j < n; ++j)
            {
                double proj = 0.0;
                for (int i = 0; i < len; ++i)
                    proj += v[i] * B[col + i][j];
                proj *= beta;
                for (int i = 0; i < len; ++i)
                    B[col + i][j] -= v[i] * proj;
            }
            // Update U = U * (I - beta v v^T) on columns col..m-1
            for (int i = 0; i < m; ++i)
            {
                double proj = 0.0;
                for (int k = 0; k < len; ++k)
                    proj += U[i][col + k] * v[k];
                proj *= beta;
                for (int k = 0; k < len; ++k)
                    U[i][col + k] -= proj * v[k];
            }
        }
        for (int i = col + 1; i < m; ++i)
            B[i][col] = 0.0;
        free(x);
        free(v);

        // Right Householder: zero out B[col][col+2 .. n-1]
        if (col < n - 1)
        {
            int len2 = n - col - 1;
            double *x2 = alloc_vector(len2);
            for (int j = 0; j < len2; ++j)
                x2[j] = B[col][col + 1 + j];
            double beta2;
            double *v2 = householder_vector(x2, len2, &beta2);
            if (beta2 != 0.0)
            {
                // Update B rows col..m-1, cols col+1..n-1
                for (int i = col; i < m; ++i)
                {
                    double proj = 0.0;
                    for (int k = 0; k < len2; ++k)
                        proj += B[i][col + 1 + k] * v2[k];
                    proj *= beta2;
                    for (int k = 0; k < len2; ++k)
                        B[i][col + 1 + k] -= proj * v2[k];
                }
                // Update Vt = (I - beta2 v2 v2^T) * Vt (apply from rows col+1..)
                for (int i = 0; i < n; ++i)
                {
                    double proj = 0.0;
                    for (int k = 0; k < len2; ++k)
                        proj += v2[k] * Vt[col + 1 + k][i];
                    proj *= beta2;
                    for (int k = 0; k < len2; ++k)
                        Vt[col + 1 + k][i] -= proj * v2[k];
                }
            }
            for (int j = col + 2; j < n; ++j)
                B[col][j] = 0.0;
            free(x2);
            free(v2);
        }
    }

    *outU = U;
    *outB = B;
    *outVt = Vt;
}

/* ----------------- Jacobi eigen decomposition for symmetric matrix -----------------
 Returns eigvals (allocated array length n) and eigvecs matrix (n x n, columns are eigenvectors).
 The caller gets eigvals (array) and eigvecs (matrix) and should free them.
*/
void jacobi_eigen_decomposition(double **A, int n, double **outeigvals, double ***outeigvecs, double tol, int max_iter)
{
    double **V = alloc_matrix(n, n); // columns are eigenvectors in standard row-major layout
    double **M = alloc_matrix(n, n);
    // init
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            V[i][j] = (i == j) ? 1.0 : 0.0;
            M[i][j] = A[i][j];
        }

    for (int iter = 0; iter < max_iter; ++iter)
    {
        double max_val = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                double v = fabs(M[i][j]);
                if (v > max_val)
                {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_val < tol)
            break;
        double app = M[p][p], aqq = M[q][q], apq = M[p][q];
        double c, s;
        if (apq != 0.0)
        {
            double tau = (aqq - app) / (2.0 * apq);
            double t = 1.0 / (fabs(tau) + sqrt(1.0 + tau * tau));
            if (tau < 0.0)
                t = -t;
            c = 1.0 / sqrt(1.0 + t * t);
            s = t * c;
        }
        else
        {
            c = 1.0;
            s = 0.0;
        }
        M[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        M[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        M[p][q] = M[q][p] = 0.0;

        for (int i = 0; i < n; ++i)
        {
            if (i != p && i != q)
            {
                double aip = M[i][p], aiq = M[i][q];
                M[i][p] = M[p][i] = c * aip - s * aiq;
                M[i][q] = M[q][i] = c * aiq + s * aip;
            }
        }
        for (int i = 0; i < n; ++i)
        {
            double vip = V[i][p], viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = c * viq + s * vip;
        }
    }

    double *eigvals = alloc_vector(n);
    for (int i = 0; i < n; ++i)
        eigvals[i] = M[i][i];

    // normalize eigenvector columns for stability
    for (int j = 0; j < n; ++j)
    {
        double norm = 0.0;
        for (int i = 0; i < n; ++i)
            norm += V[i][j] * V[i][j];
        norm = sqrt(norm);
        if (norm > 0.0)
            for (int i = 0; i < n; ++i)
                V[i][j] /= norm;
    }

    free_matrix(M);
    *outeigvals = eigvals;
    *outeigvecs = V;
}

/* ----------------- Modified Gram-Schmidt (columns orthonormalization) -------------- */
/* Input: cols = number of input column vectors (each length m), colvecs pointer to array of pointers
   Output: orthonormal columns stored into orth_cols (array of pointers). Caller must allocate arrays.
   We will allocate orth_cols pointers inside the function (returned as double**), each column is newly
   allocated length m. Returns number of orthonormal columns produced via out_len.
*/
int modified_gram_schmidt(double **colvecs, int cols, int m, double ***out_cols)
{
    double **Q = (double **)malloc(cols * sizeof(double *));
    int q_len = 0;
    for (int i = 0; i < cols; ++i)
    {
        double *u = alloc_vector(m);
        for (int r = 0; r < m; ++r)
            u[r] = colvecs[i][r];
        for (int j = 0; j < q_len; ++j)
        {
            double dot = 0.0;
            for (int r = 0; r < m; ++r)
                dot += Q[j][r] * u[r];
            for (int r = 0; r < m; ++r)
                u[r] -= dot * Q[j][r];
        }
        double norm = 0.0;
        for (int r = 0; r < m; ++r)
            norm += u[r] * u[r];
        norm = sqrt(norm);
        if (norm < 1e-12)
        {
            free(u);
            continue;
        }
        for (int r = 0; r < m; ++r)
            u[r] /= norm;
        Q[q_len++] = u;
    }
    *out_cols = Q;
    return q_len;
}

/* ----------------- SVD: Golub-Reinsch main routine ------------------
   Inputs: A (m x n), m >= n preferred but code handles m < n via transposition branch.
   Outputs: U (m x m), S (length min(m,n) but we return n values for full), Vt (n x n).
   All outputs are newly allocated. Caller should free them.
*/
void svd_golub_reinsch(double **A, int m, int n, double ***outU, double **outS, double ***outVt)
{
    // If m < n, compute SVD of A^T and swap U/V later
    if (m < n)
    {
        // Transpose A -> At (n x m)
        double **At = alloc_matrix(n, m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                At[i][j] = A[j][i];

        double **U_t, **Vt_t;
        double *S_temp;
        svd_golub_reinsch(At, n, m, &U_t, &S_temp, &Vt_t);
        // U (m x m) = Vt_t^T (m x m) -- note size adjustment: U_t is n x n, Vt_t is m x m in recursion
        double **U_final = alloc_matrix(m, m);
        double **Vt_final = alloc_matrix(n, n);
        // Vt_t has size m x m (since full_matrices True was used). U_t has size n x n.
        // But recursion produced U_t (n x n), Vt_t (m x m)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
            {
                // Vt_t is m x m, we want its transpose
                Vt_final[i][j] = Vt_t[j][i];
            }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                // U_final is m x m; U_t is n x n; fill top-left n x n with U_t^T
                // But to be consistent, we create Vt_final as above and U_final from U_t transpose.
                // Simpler: U_final (m x m) fill by pulling from Vt_t's transpose for first m rows/cols:
                // To match Python logic: original U = Vt_t^T (m×m), original Vt = U_t^T (n×n)
                // We already set Vt_final (n x n) below; fill U_final from Vt_t's transpose (size m)
                U_final[i][j] = Vt_t[j][i];
            }
        // fill Vt_final (n x n) from U_t^T, but sizes differ: we'll fill top-left n x n from U_t^T
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Vt_final[i][j] = U_t[j][i];

        *outU = U_final;
        *outS = S_temp;
        *outVt = Vt_final;

        // cleanup recursion temporaries
        free_matrix(At);
        free_matrix(U_t);
        free_matrix(Vt_t);
        return;
    }

    // 1) bidiagonalization
    double **U_bi, **B, **Vt_bi;
    householder_bidiagonalization(A, m, n, &U_bi, &B, &Vt_bi);

    // 2) build T = B^T * B (n x n)
    double **T = alloc_matrix(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                s += B[k][i] * B[k][j];
            T[i][j] = s;
        }
    }

    // 3) Jacobi eigendecomposition on T
    double *eigvals;
    double **eigvecs; // eigvecs is n x n with columns being eigenvectors
    jacobi_eigen_decomposition(T, n, &eigvals, &eigvecs, 1e-12, 1000);

    // 4) collect eigenpairs and sort by abs(eigval) desc
    typedef struct
    {
        double val;
        double *vec;
    } EigPair;
    EigPair *pairs = (EigPair *)malloc(n * sizeof(EigPair));
    for (int j = 0; j < n; ++j)
    {
        // copy column j of eigvecs into new vector to own memory
        double *col = alloc_vector(n);
        for (int i = 0; i < n; ++i)
            col[i] = eigvecs[i][j];
        pairs[j].val = eigvals[j];
        pairs[j].vec = col;
    }
    // sort (simple selection sort for clarity)
    for (int i = 0; i < n - 1; ++i)
    {
        int idx = i;
        double best = fabs(pairs[i].val);
        for (int j = i + 1; j < n; ++j)
        {
            if (fabs(pairs[j].val) > best)
            {
                best = fabs(pairs[j].val);
                idx = j;
            }
        }
        if (idx != i)
        {
            EigPair t = pairs[i];
            pairs[i] = pairs[idx];
            pairs[idx] = t;
        }
    }
    for (int i = 0; i < n; ++i)
        if (pairs[i].val < 0.0)
            pairs[i].val = 0.0;

    // 5) singular values
    double *S_vals = alloc_vector(n);
    for (int i = 0; i < n; ++i)
        S_vals[i] = sqrt(pairs[i].val);

    // 6) V_B matrix (n x n): columns are pairs[j].vec
    double **V_B_mat = alloc_matrix(n, n);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
            V_B_mat[i][j] = pairs[j].vec[i];
    }

    // 7) Compute U_B_cols: u_i = (B * v_i) / sigma_i
    double **U_B_cols = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; ++i)
    {
        if (S_vals[i] == 0.0)
        {
            U_B_cols[i] = alloc_vector(m); // zeros
        }
        else
        {
            double sigma = S_vals[i];
            double *col = alloc_vector(m);
            for (int r = 0; r < m; ++r)
            {
                double s = 0.0;
                for (int c = 0; c < n; ++c)
                    s += B[r][c] * V_B_mat[c][i];
                col[r] = s / sigma;
            }
            U_B_cols[i] = col;
        }
    }

    // 8) Orthonormalize U_B_cols via modified Gram-Schmidt -> U_B (list of orthonormal column vectors)
    double **U_B;
    int U_B_len = modified_gram_schmidt(U_B_cols, n, m, &U_B);

    // 9) Complete U_B to full basis (m columns) using standard basis vectors, orthonormalizing each
    double **U_B_full_cols = (double **)malloc(m * sizeof(double *));
    int U_B_full_len = 0;
    for (int i = 0; i < U_B_len; ++i)
        U_B_full_cols[U_B_full_len++] = U_B[i];
    for (int idx = 0; idx < m && U_B_full_len < m; ++idx)
    {
        double *e = alloc_vector(m);
        e[idx] = 1.0;
        // orthogonalize against existing columns
        for (int j = 0; j < U_B_full_len; ++j)
        {
            double proj = 0.0;
            for (int r = 0; r < m; ++r)
                proj += U_B_full_cols[j][r] * e[r];
            for (int r = 0; r < m; ++r)
                e[r] -= proj * U_B_full_cols[j][r];
        }
        double norm = 0.0;
        for (int r = 0; r < m; ++r)
            norm += e[r] * e[r];
        norm = sqrt(norm);
        if (norm < 1e-12)
        {
            free(e);
            continue;
        }
        for (int r = 0; r < m; ++r)
            e[r] /= norm;
        U_B_full_cols[U_B_full_len++] = e;
    }
    // If still less than m (degenerate), fill with zeros (shouldn't happen normally)
    while (U_B_full_len < m)
        U_B_full_cols[U_B_full_len++] = alloc_vector(m);

    // 10) Compute U_final = U_bi * U_B_full  (U_B_full stored as array of column vectors)
    double **U_final = alloc_matrix(m, m);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                s += U_bi[i][k] * U_B_full_cols[j][k];
            U_final[i][j] = s;
        }
    }

    // 11) Compute Vt_final = V_B^T * Vt_bi  (V_B_mat columns are right singular vectors)
    double **Vt_final = alloc_matrix(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += V_B_mat[k][i] * Vt_bi[k][j];
            Vt_final[i][j] = s;
        }
    }

    // 12) Sign convention: make largest absolute element in each U column non-negative by flipping sign of col and corresponding Vt row
    for (int j = 0; j < n; ++j)
    {
        int imax = 0;
        double best = fabs(U_final[0][j]);
        for (int i = 1; i < m; ++i)
        {
            double v = fabs(U_final[i][j]);
            if (v > best)
            {
                best = v;
                imax = i;
            }
        }
        if (U_final[imax][j] < 0.0)
        {
            for (int i = 0; i < m; ++i)
                U_final[i][j] = -U_final[i][j];
            for (int k = 0; k < n; ++k)
                Vt_final[j][k] = -Vt_final[j][k];
        }
    }

    // outputs
    *outU = U_final;
    *outS = S_vals;
    *outVt = Vt_final;

    /* ----------------- cleanup intermediate resources (only free memory we own) ------------- */

    // free T and eigvecs
    free_matrix(T);
    for (int j = 0; j < n; ++j)
        free(pairs[j].vec);
    free(pairs);
    free(eigvals);
    free_matrix(eigvecs);

    // free V_B_mat and B and U_bi, Vt_bi
    free_matrix(V_B_mat);
    for (int i = 0; i < n; ++i)
        free(U_B_cols[i]);
    free(U_B_cols);
    free(U_B); // note: U_B entries moved into U_B_full_cols, so pointers are in U_B_full_cols
    // free U_B_full_cols array structure (but not the individual column buffers used in U_final because U_final references different storage)
    // U_final has its own storage, so we free U_B_full_cols' column buffers now:
    for (int i = 0; i < U_B_full_len; ++i)
        free(U_B_full_cols[i]);
    free(U_B_full_cols);

    free_matrix(B);
    free_matrix(U_bi);
    free_matrix(Vt_bi);
}

/* ----------------- small helpers: print matrix, reconstruct and compute error -------------- */

void print_matrix(const char *name, double **M, int rows, int cols)
{
    // printf("%s =\n", name);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            printf(" %12.8f", M[i][j]);
        printf("\n");
    }
    // printf("\n");
}

void print_vector(const char *name, double *v, int n)
{
    printf("%s = [", name);
    for (int i = 0; i < n; ++i)
    {
        if (i)
            printf(", ");
        printf("%12.6f", v[i]);
    }
    printf("]\n\n");
}

double max_abs_diff_matrix(double **A, double **B, int m, int n)
{
    double maxd = 0.0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            double d = fabs(A[i][j] - B[i][j]);
            if (d > maxd)
                maxd = d;
        }
    return maxd;
}

double **matmul_alloc(double **A, int m, int p, double **B, int n)
{
    double **C = alloc_matrix(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < p; ++k)
                s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    return C;
}

int main(void)
{
    srand_custom(42);
    int m = 3, n = 3;

    double **U, **Vt;
    double *S;
    double **A = alloc_matrix(m, n);
    double **Sigma = alloc_matrix(m, n);
    freopen("error.log", "w", stderr);
    for (int j = 0; j < 10000; j++)
    {
        // printf("idx:%d\n", j);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                A[i][j] = rand_uniform();

        print_matrix("A", A, m, n);

        svd_golub_reinsch(A, m, n, &U, &S, &Vt);

        // print_matrix("U", U, m, m);
        // print_vector("S", S, n);
        // print_matrix("Vt", Vt, n, n);

        // reconstruct A' = U * diag(S) * Vt
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                Sigma[i][j] = 0.0;
        for (int i = 0; i < n; ++i)
            Sigma[i][i] = S[i];

        double **UV = matmul_alloc(U, m, m, Vt, n); // (m x n)

        print_matrix("UV\n", UV, m, n);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
                fprintf(stderr, " %12.8f", UV[i][j]);

            printf("\n");
            fprintf(stderr, "\n");
        }
        free_matrix(UV);
    }

    // cleanup
    free_matrix(A);
    free_matrix(U);
    free_matrix(Vt);
    free(S);
    free_matrix(Sigma);

    return 0;
}
