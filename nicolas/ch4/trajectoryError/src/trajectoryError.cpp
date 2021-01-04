/* System Libraries */
#include <iostream>
#include <fstream>
#include <unistd.h>


/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

/* Sophys Libraries */
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

/* Pangolin Library */
#include <pangolin/pangolin.h>

/* Custom Libraries */
#include "../../include/libUtils.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
string estimated_file = "../../trajectoryError/src/estimated.txt";
string groundtruth_file = "../../trajectoryError/src/groundtruth.txt";

/* Function Scopes */
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

TrajectoryType ReadTrajectory(const string &path);
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &est);

/* ====== */
/*  Main  */
/* ====== */
/*  This Program demonstrates the calculation of the Absolute Trajectory Error (ATE) */
int main(int argc, char **argv){
    cout << argc << endl; 



    print("helloTrajectoryError!");

    // 1. Read the two trajectories (Sequences of Poses)
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
    TrajectoryType estimated = ReadTrajectory(estimated_file);

    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    int N = estimated.size();

    // 2. Calculate the Absolute Trajectory Error (ATE)
    double ate_all_sum = 0;
    double ate_trans_sum = 0;
    
    for(size_t i=0; i < N; i++){
        auto pose_est = estimated[i], pose_gt = groundtruth[i];

        string pose_gt_str = ("pose_gt[" + to_string(i) + "]: ");
        string pose_est_str = ("pose_est[" + to_string(i) + "]: ");

        printMatrix<Matrix4d>(pose_gt_str.c_str(), pose_gt.matrix());  
        printMatrix<Matrix4d>(pose_est_str.c_str(), pose_est.matrix());

        double ate_all_error = (pose_gt.inverse()*pose_est).log().norm();
        double ate_trans_error = (pose_gt.inverse()*pose_est).translation().norm();
        
        ate_all_sum += ate_all_error * ate_all_error;
        ate_trans_sum += ate_trans_error * ate_trans_error;
    }

    double ate_all = sqrt(ate_all_sum/double(N));
    double ate_trans = sqrt(ate_trans_sum/double(N));

    cout << "Absolute Trajectory Error (ATE_all)/Root-Mean-Squared Error (RMSE): " << ate_all << endl;
    cout << "Average Translational Error (ATE_trans): " << ate_trans << endl << endl;

    // 3. Calculate the Relative Pose Error (RPE)
    double rpe_all_sum = 0;
    double rpe_trans_sum = 0;

    int dt = 1;

    for(size_t i=0; i < N-dt; i++){
        auto pose_est = estimated[i], pose_gt = groundtruth[i];
        auto pose_est_dt = estimated[i+dt], pose_gt_dt = groundtruth[i+dt];
        
        double rpe_all_error = ((pose_gt.inverse()*pose_gt_dt).inverse()*(pose_est.inverse()*pose_est_dt)).log().norm();
        double rpe_trans_error = ((pose_gt.inverse()*pose_gt_dt).inverse()*(pose_est.inverse()*pose_est_dt)).translation().norm();
        
        rpe_all_sum += rpe_all_error * rpe_all_error;
        rpe_trans_sum += rpe_trans_error * rpe_trans_error;
    }

    double rpe_all = sqrt(rpe_all_sum/(double(N)-dt));
    double rpe_trans = sqrt(rpe_trans_sum/(double(N)-dt));

    cout << "Relative Pose Error (RPE_all): " << rpe_all << endl;
    cout << "Relative Pose Error Error (RPE_trans): " << rpe_trans << endl << endl;

    // 4. Display the trajectories in a 3D Window.
    DrawTrajectory(groundtruth, estimated);

    cout << "Done." << endl;

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
        cout << "Read '" << path << "' was sucessful." << endl;
    }

    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Transformation Matrix (T), Pose
        Sophus::SE3d pose(Eigen::Quaterniond(qx, qy, qz, qw), Eigen::Vector3d(tx, ty, tz)); // T, SE(3) from q,t.
        trajectory.push_back(pose);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}


void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &est) {
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
  .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
  .SetHandler(new pangolin::Handler3D(s_cam));

  /* Loop */
  while (pangolin::ShouldQuit() == false){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);

    /* Draw the connections between Poses */
    for (size_t i=0; i < gt.size()-1; i++) {
      // Get two consecutive poses
      auto p1 = gt[i], p2 = gt[i + 1];

      glBegin(GL_LINES);
      glColor3f(0.0f, 0.0f, 1.0f);  // Blue for ground truth trajectory
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    /* Draw the connections between Poses */
    for(size_t i=0; i < est.size()-1; i++){
      // Get two consecutive poses
      auto p1 = est[i], p2 = est[i + 1];
      
      // Draw links vertexes
      glBegin(GL_LINES);
      glColor3f(1.0f, 0.0f, 0.0f);  // Red for estimated trajectory
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    
    pangolin::FinishFrame();
    usleep(5000);  // sleep 5 ms
  }
}
