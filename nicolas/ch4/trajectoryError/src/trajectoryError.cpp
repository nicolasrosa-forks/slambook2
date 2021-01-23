/* Libraries */
#include "../include/trajectoryError.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
string traj_est_filepath = "../../trajectoryError/src/estimated.txt";
string traj_gt_filepath = "../../trajectoryError/src/groundtruth.txt";

TimeStamp time_est, time_gt;

/* ================================================================================================ */
/*                                     Metrics Description                                          */
/* ------------------------------------------------------------------------------------------------ */
/*  This Program demonstrates the calculation of the Absolute Trajectory Error (ATE)                */
/*  a) Absolute Trajectory Error (ATE), which measures the difference between the translation       */
/*     part of two trajectories by first aligning them into a common reference frame, and           */
/*                                                                                                  */
/*  b) Relative Pose Error (RPE), which measures the difference between relative transformations    */
/*     at time instances i and i+k, for different values of k. This method is independent of the    */
/*     reference frame but when the scale of the map is not known (for example monocular mapping),  */ 
/*     a scale alignment needs to be done before comparing trajectories using RPE.                  */
/*                                                                                                  */
/*  Ref: Salas, Marta, et al. "Trajectory alignment and evaluation in SLAM: Horns method vs         */
/*       alignment on the manifold." Robotics: Science and Systems Workshop: The problem of         */
/*       mobile sensors. 2015.                                                                      */
/* ================================================================================================ */

/* This Program demonstrates the calculation of the Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) */
int main(int argc, char **argv){
    print("[trajectoryError] Hello!");

    // 1. Read the two trajectories (Sequences of Poses)
    TrajectoryType estimated = ReadTrajectory(time_est, traj_est_filepath);
    TrajectoryType groundtruth = ReadTrajectory(time_gt, traj_gt_filepath);
    
    assert(!estimated.empty() && !groundtruth.empty());
    assert(estimated.size() == groundtruth.size());

    int N = groundtruth.size();

    // 2. Calculate the Absolute Trajectory Error (ATE)
    // FIXME: Do I still need to align the trajectories? Do I need to use that HORN algorithm?
    double ate_all_sum = 0;
    double ate_trans_sum = 0;
    
    for(size_t i=0; i < N; i++){
        auto pose_est = estimated[i], pose_gt = groundtruth[i];

        string pose_est_str = "pose_est[" + to_string(i) + "]: ";
        string pose_gt_str = "pose_gt[" + to_string(i) + "]: ";
        
        printMatrix<Matrix4d>(pose_est_str.c_str(), pose_est.matrix());
        printMatrix<Matrix4d>(pose_gt_str.c_str(), pose_gt.matrix());  
        
        double ate_all_error = (pose_gt.inverse()*pose_est).log().norm();
        double ate_trans_error = (pose_gt.inverse()*pose_est).translation().norm();
        
        ate_all_sum += ate_all_error * ate_all_error;
        ate_trans_sum += ate_trans_error * ate_trans_error;
    }

    double ate_all = sqrt(ate_all_sum/double(N));
    double ate_trans = sqrt(ate_trans_sum/double(N));

    cout << "Absolute Trajectory Error (ATE_all)/Root-Mean-Squared Error (RMSE): " << ate_all << endl;
    cout << "Average Translational Error (ATE_trans): " << ate_trans << endl << endl;  //TODO: Is this result correct?

    // 3. Calculate the Relative Pose Error (RPE)
    double rpe_all_sum = 0;
    double rpe_trans_sum = 0;

    double dt = 1;  //Remember: dt isn't a increment of i-index, but time.
    size_t k=1;
    int j=0;
    
    for(size_t i=0; i < N; i++){  // In practice, this for run 'j=N-dt' times.
        auto pose_est = estimated[i], pose_gt = groundtruth[i];
        
        cout << "i: " << i << endl;
        cout << "t_diff:" << time_gt[i+k] - time_gt[i] << endl;
    
        while(i+k < N && (time_gt[i+k] - time_gt[i] < dt)){
          k++;
        }

        if (i+k < N){
          auto pose_est_dt = estimated[i+k], pose_gt_dt = groundtruth[i+k];  
          
          double rpe_all_error = ((pose_gt.inverse()*pose_gt_dt).inverse()*(pose_est.inverse()*pose_est_dt)).log().norm();
          double rpe_trans_error = ((pose_gt.inverse()*pose_gt_dt).inverse()*(pose_est.inverse()*pose_est_dt)).translation().norm();
        
          rpe_all_sum += rpe_all_error * rpe_all_error;
          rpe_trans_sum += rpe_trans_error * rpe_trans_error;

          cout << "error value updated!" << endl;

          k=1;
          j++;
          
          cout << "k: " << k << endl << endl;
        }else{
          break;
        }   
    }

    cout << "\nNumber of error updates (j): " << j << endl << endl;

    double rpe_all = sqrt(rpe_all_sum/(double(j)));
    double rpe_trans = sqrt(rpe_trans_sum/(double(j)));

    cout << "Relative Pose Error (RPE_all): " << rpe_all << endl;  //TODO: Is this result correct? Implementation was supervised by Nuno
    cout << "Relative Pose Error Error (RPE_trans): " << rpe_trans << endl << endl;  //TODO: Is this result correct? Implementation was supervised by Nuno

    // 4. Display the trajectories in a 3D Window.
    cout << "Press 'ESC' to exit the program" << endl;
    DrawTrajectory(groundtruth, estimated);

    cout << "Done." << endl;
}

/* =========== */
/*  Functions  */
/* =========== */
TrajectoryType ReadTrajectory(TimeStamp &timestamps, const string &path){
    ifstream fin(path);  // The 'pose.txt' contains the Twc transformations!
    TrajectoryType trajectory;

    cout << "[trajectoryError] Reading '" << path << "'... ";
    if(!fin){
        cerr << "File not found!" << endl;
        return trajectory;
    }else{
        cout << "Successful" << endl;
    }

    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Transformation Matrix (T), Pose
        Sophus::SE3d pose(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));  // T, SE(3) from q,t.
        timestamps.push_back(time);
        trajectory.push_back(pose);
    }
    cout << "Read total of " << trajectory.size() << " pose entries." << endl << endl;

    return trajectory;
}

void DrawTrajectory(const TrajectoryType &est, const TrajectoryType &gt) {
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
