/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <vector>
// #include <algorithm>
#include <unistd.h>  // usleep()
#include <fstream>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
typedef vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> TrajectoryType;

TrajectoryType ReadTrajectory(const string &path);
void DrawTrajectory(const TrajectoryType &poses);

/* Global Variables */
// string trajectory_path = "../../examples/trajectory.txt";
string trajectory_path = "/home/nicolas/github/nicolasrosa-forks/slam/slambook2/nicolas/ch3/examples/trajectory.txt";

/* ====== */
/*  Main  */
/* ====== */
int main(int argc, char** argv){
    // Read trajectory file
    TrajectoryType poses = ReadTrajectory(trajectory_path);

    // Draw trajectory in pangolin
    DrawTrajectory(poses);

    return 0;
}

/* =========== */
/*  Functions  */
/* =========== */
TrajectoryType ReadTrajectory(const string &path){
    ifstream fin(path);
    TrajectoryType trajectory;

    if(!fin){
        cout << "Cannot find trajectory file at '" << path << "'." << endl;
        return trajectory;
    }else{
        cout << "Read '" << path << "' was successful." << endl;
    }

    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz));

        printMatrix<Matrix4d>("Twr: ", Twr.matrix());

        trajectory.push_back(Twr);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}

void DrawTrajectory(const TrajectoryType &poses){
    // Create Pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);                            // The Depth Test is a per-sample processing operation performed after the Fragment Shader (and sometimes before). https://www.khronos.org/opengl/wiki/Depth_Test
    glEnable(GL_BLEND);                                 // If enabled, blend the computed fragment color values with the values in the color buffers. https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glEnable.xml
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Pixels can be drawn using a function that blends the incoming (source) RGBA values with the RGBA values that are already in the frame buffer (the destination values). https://www.khronos.org/registry/OpenGL-Refpages/es2.0/xhtml/glBlendFunc.xml

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );  // Object representing attached OpenGl Matrices/transforms

    pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

    /* Loop */
    while (pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);

        /* Draw three axes of each pose */
        for (size_t i=0; i < poses.size(); i++) {
            // Get origin (robot position in world coordinates) and axes unit-length vectors
            Vector3d Ow = poses[i].translation();            // Origin, Ow = Twr.Or = twr
            Vector3d Xw = poses[i]*(0.1*Vector3d(1, 0, 0));  // X-Axis
            Vector3d Yw = poses[i]*(0.1*Vector3d(0, 1, 0));  // Y-Axis
            Vector3d Zw = poses[i]*(0.1*Vector3d(0, 0, 1));  // Z-Axis

            // Draw coordinate axes vertexes
            glBegin(GL_LINES);

            // X-Axis (Red)
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d (Ow[0], Ow[1], Ow[2]);
            glVertex3d (Xw[0], Ow[1], Ow[2]);

            // Y-Axis (Green)
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d (Ow[0], Ow[1], Ow[2]);
            glVertex3d (Yw[0], Yw[1], Yw[2]);

            // Z-Axis (Blue)
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d (Ow[0], Ow[1], Ow[2]);
            glVertex3d (Zw[0], Zw[1], Zw[2]);

            glEnd();
        }

        /* Draw the connections between Poses */
        for(size_t i=0; i < poses.size()-1; i++){
            // Get two consecutive poses
            auto p1 = poses[i], p2 = poses[i+1];  // Placeholder type specifiers. https://en.cppreference.com/w/cpp/language/auto

            // Draw links vertexes
            glBegin(GL_LINES);
            glColor3f(0.0, 0.0, 0.0);
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }

        pangolin::FinishFrame();
        usleep(5000);  // sleep 5 ms
    }
}
