/*
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "solvepnp.h"  // 包含solvePnPbyDLT的声明

using namespace Eigen;
using namespace std;

int main() {
    // 1. 定义6个人脸关键点的3D世界坐标（毫米）- 使用Eigen::Vector3d
    vector<Vector3d> objectPoints;
    objectPoints.push_back(Vector3d(-100, 50, 0));    // 左眼
    objectPoints.push_back(Vector3d(100, 50, 0));     // 右眼
    objectPoints.push_back(Vector3d(0, 0, 0));        // 鼻尖
    objectPoints.push_back(Vector3d(-70, -100, 0));   // 左嘴角
    objectPoints.push_back(Vector3d(70, -100, 0));    // 右嘴角
    objectPoints.push_back(Vector3d(0, -120, 0));     // 下巴中心

    // 2. 定义对应的2D图像坐标（像素）- 使用Eigen::Vector2d
    vector<Vector2d> imagePoints;
    imagePoints.push_back(Vector2d(270, 215));        // 左眼投影
    imagePoints.push_back(Vector2d(370, 215));        // 右眼投影
    imagePoints.push_back(Vector2d(320, 240));        // 鼻尖投影
    imagePoints.push_back(Vector2d(285, 290));        // 左嘴角投影
    imagePoints.push_back(Vector2d(355, 290));        // 右嘴角投影
    imagePoints.push_back(Vector2d(320, 300));        // 下巴中心投影

    // 3. 定义相机内参矩阵 - 使用Eigen::Matrix3d
    Matrix3d K;
    K << 500, 0, 320,   // fx, s, cx
         0, 500, 240,   // 0, fy, cy
         0, 0, 1;       // 内参最后一行

    // 4. 准备输出变量
    Matrix3d R;   // 旋转矩阵
    Vector3d t;   // 平移向量

    // 5. 调用函数
    solvePnPbyDLT(K ,objectPoints, imagePoints, R, t);

    // 6. 打印结果
    cout << "Rotation Matrix:\n" << R << endl;
    cout << "Translation Vector:\n" << t << endl;
    
    // 可选：转换为旋转向量
    AngleAxisd rotationVector(R);
    cout << "\nRotation Vector (angle-axis):\n"
         << "Angle = " << rotationVector.angle() * 180 / M_PI << " deg\n"
         << "Axis = " << rotationVector.axis().transpose() << endl;

    return 0;
}
*/



/*
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include "solvepnp_2.h"
// ------------------ 测试函数 ------------------
int main() {
    // 示例相机内参
    Matrix3d K = { {
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0, 1.0
    } };

    // 创建6个3D点
    Vector3d pts3d[6] = {
        {-100, 50, 0},
        {100, 50, 0},
        {0, 0, 0},
        {-70, -100, 0},
        {70, -100, 0},
        {0, -120, 0}
    };

    // 创建对应的2D点（这里使用近似值）
    Vector2d pts2d[6] = {
        {270, 215},
        {370, 215},
        {320, 240},
        {285, 290},
        {355, 290},
        {320, 300}
    };

    Matrix3d R;
    Vector3d t;

    if (solvePnPbyDLT2(&K, pts3d, pts2d, 6, &R, &t)) {
        printf("PnP solution found!\n");
        printf("Rotation Matrix:\n");
        for (int i = 0; i < 3; i++) {
            printf("%f %f %f\n", R.data[i * 3], R.data[i * 3 + 1], R.data[i * 3 + 2]);
        }
        printf("Translation: %f %f %f\n", t.x, t.y, t.z);
    }
    else {
        printf("PnP solution failed!\n");
    }

    return 0;
}
*/

/*
#include "solvepnp_3.h"

int main() {
    // 相机内参
    double K[3][3] = {
        {500, 0, 320},
        {0, 500, 240},
        {0, 0, 1}
    };

    // 3D 点（世界坐标系）
    Vector3d pts3d[6] = {
        {-100, 50, 0},
        {100, 50, 0},
        {0, 0, 0},
        {-70, -100, 0},
        {70, -100, 0},
        {0, -120, 0}
    };

    // 2D 点（图像坐标系），假设相机在原点，无旋转和平移时的投影
    Vector2d pts2d[6] = {
        {270, 215},
        {370, 215},
        {320, 240},
        {285, 290},
        {355, 290},
        {320, 300}
    };

    Matrix3d R;
    Vector3d t;

    if (solvePnPbyDLT(K, pts3d, pts2d, &R, &t)) {
        printf("Estimated Rotation:\n");
        print_matrix3(&R);

        printf("Estimated Translation:\n");
        print_vector3(&t);
    }
    else {
        printf("Failed to compute PnP.\n");
    }

    return 0;
}*/

#include "solvepnp_4.h"
// ------------------ 测试函数 ------------------
int main() {
    // 示例相机内参
    Matrix3d K = { {
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0, 1.0
    } };

    // 创建6个3D点
    Vector3d pts3d[6] = {
        {-100, 50, 0},
        {100, 50, 0},
        {0, 0, 0},
        {-70, -100, 0},
        {70, -100, 0},
        {0, -120, 0}
    };

    // 创建对应的2D点（这里使用近似值）
    Vector2d pts2d[6] = {
        {270, 215},
        {370, 215},
        {320, 240},
        {285, 290},
        {355, 290},
        {320, 300}
    };

    Matrix3d R;
    Vector3d t;

    if (solvePnPbyDLT4(&K, pts3d, pts2d, 6, &R, &t)) {
        printf("PnP solution found!\n");
        printf("Rotation Matrix:\n");
        for (int i = 0; i < 3; i++) {
            printf("%f %f %f\n", R.data[i * 3], R.data[i * 3 + 1], R.data[i * 3 + 2]);
        }
        printf("Translation: %f %f %f\n", t.x, t.y, t.z);
    }
    else {
        printf("PnP solution failed!\n");
    }

    return 0;
}