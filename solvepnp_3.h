#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 三维点
typedef struct {
    double data[3];
} Vector3d;

// 二维点
typedef struct {
    double data[2];
} Vector2d;

// 3x3 矩阵
typedef struct {
    double data[3][3];
} Matrix3d;

int solvePnPbyDLT3(const double K[3][3],
    const Vector3d pts3d[6],
    const Vector2d pts2d[6],
    Matrix3d* R,
    Vector3d* t);

void print_matrix(int rows, int cols, const double* mat);
void print_vector3(const Vector3d* v);
void print_matrix3(const Matrix3d* m);