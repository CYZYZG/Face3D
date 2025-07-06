#ifndef PNP_SOLVER_2_H
#define PNP_SOLVER_2_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>

#define MATRIX_AT(m, i, j, cols) m[(i) * (cols) + (j)]
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX_ITER 50
#define TOLERANCE 1e-12

typedef struct {
    double x, y, z;
} Vector3d;

typedef struct {
    double x, y;
} Vector2d;

typedef struct {
    double data[9]; // 3x3 matrix in row-major order
} Matrix3d;

int solvePnPbyDLT4(const Matrix3d* K,
    const Vector3d* pts3d,
    const Vector2d* pts2d,
    int num_points,
    Matrix3d* R,
    Vector3d* t);

#endif // PNP_SOLVER_H