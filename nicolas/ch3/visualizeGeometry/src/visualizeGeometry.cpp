/* System Libraries */
#include <iostream>
#include <iomanip>
#include <cmath>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../../common/libUtils_basic.h"
#include "../../../common/libUtils_eigen.h"

using namespace std;
using namespace Eigen;

/* ===================================== */
/*  RotationMatrix Overloading Operators */
/* ===================================== */
struct RotationMatrix{
    Matrix3d matrix = Matrix3d::Identity();
};

// Overload operator <<
ostream &operator<<(ostream &out, const RotationMatrix &r){
    Matrix3d matrix = r.matrix;

    out.setf(ios::fixed);

    out << '=';
    out << "[" << setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
        << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
        << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";

    return out;
}

// Overload operator >>
istream &operator>>(istream &in, RotationMatrix &r){
    return in;
}

/* ======================================== */
/*  TranslationVector Overloading Operators */
/* ======================================== */
struct TranslationVector{
    Vector3d trans = Vector3d(0, 0, 0);
};

// Overload operator <<
ostream &operator<<(ostream &out, const TranslationVector &t){
    out << "=[" << t.trans(0) << ',' << t.trans(1) << ',' << t.trans(2) << "]";

    return out;
}

// Overload operator >>
istream &operator>>(istream &in, TranslationVector &t){
    return in;
}

/* ===================================== */
/*  QuaternionDraw Overloading Operators */
/* ===================================== */
struct QuaternionDraw {
    Quaterniond q;
};

// Overload operator <<
ostream &operator<<(ostream &out, const QuaternionDraw &quat) {
    auto c = quat.q.coeffs();
    out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
    return out;
}

// Overload operator >>
istream &operator>>(istream &in, const QuaternionDraw &quat) {
    return in;
}

/* ================================================================= */
/*  This program demonstrates how to create a Visualization Program  */
/*  for the Rotation Matrix, Eugle Angle, quaternion.                */
/* ================================================================= */
int main(int argc, char** argv){
    pangolin::CreateWindowAndBind("Visualize Geometry", 1000, 600);  // Initialize OpenGL window

    glEnable(GL_DEPTH_TEST);                            // The Depth Test is a per-sample processing operation performed after the Fragment Shader (and sometimes before). https://www.khronos.org/opengl/wiki/Depth_Test

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),
        pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
    );  // Object representing attached OpenGl Matrices/transforms

    const int UI_WIDTH = 500;

    pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

    /* UI */
    pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
    pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
    pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
    pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    /* Loop */
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();
        Matrix<double, 4, 4> m = matrix;

        // Rotation Matrix
        RotationMatrix R;
        for (int i=0; i<3; i++)
            for (int j=0; j<3; j++)
                R.matrix(i, j) = m(j, i);
        rotation_matrix = R;

        // Translation Vector
        TranslationVector t;
        t.trans = Vector3d(m(0, 3), m(1, 3), m(2, 3));
        t.trans = -R.matrix * t.trans;

        // Euler angles
        TranslationVector euler;
        euler.trans = R.matrix.eulerAngles(2, 1, 0);  // [yaw, pitch, roll]

        // Quaternion
        QuaternionDraw quat;
        quat.q = Quaterniond(R.matrix);

        glColor3f(1.0, 1.0, 1.0);  // White
        pangolin::glDrawColouredCube();

        // Draw the original axis
        glLineWidth(3);
        glBegin(GL_LINES);

        glColor3f(0.8f, 0.f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(10, 0, 0);

        glColor3f(0.f, 0.8f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 10, 0);

        glColor3f(0.2f, 0.2f, 1.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 10);

        glEnd();

        pangolin::FinishFrame();
    }

    return 0;
}