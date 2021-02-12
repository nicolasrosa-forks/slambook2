/* Libraries */
#include "../include/gaussNewton.h"

using namespace std;
using namespace Eigen;

double RMSE(const Vector3d est, const Vector3d gt){
    double sum = 0.0;
    int N = 3;

    for(int i=0;i<N;i++){
        sum += pow(est[i]-gt[i], 2.0);
    }

    return sqrt(sum/(double)N);
}

/* ====== */
/*  Main  */
/* ====== */
int main(int argc, char **argv) {
    cout << "[gaussNetwon] Hello!" << endl;

    /* Variables */
    double ar = 1.0, br = 2.0, cr = 1.0;        // Real parameters values
    double ae = 2.0, be = -1.0, ce = 5.0;       // Estimated parameters values
    int N = 100;                                // Number of Data points

    cv::RNG rng;                                // OpenCV Random Number generator
    double w_sigma = 1.0;                       // Noise sigma value
    double inv_sigma = 1.0 / w_sigma;

    /* ----- Data Generation ----- */
    vector<double> x_data, y_data;              // Data Vectors
  
    for(int i=0; i < N; i++){
        double x = i/100.0;
        double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma);
    
        x_data.push_back(x);
        y_data.push_back(y);
    }

    printVec("x_data: ", x_data);
    printVec("y_data: ", y_data);

    /* ----- Gauss-Newton Method ----- */
    int iterations = 100;               // Number of iterations
    double cost = 0.0, lastCost = 0.0;  // The cost of current iteration and the cost of the previous one

    print("Summary: ");
    Timer t1 = chrono::steady_clock::now();
    // Iteration Loop
    for(int iter=0; iter < iterations; iter++){
        cout << iter << endl;

        Matrix3d H = Matrix3d::Zero();     // Hessian, H(x) = J^T.inv(W).J in Gauss-Newton //FIXME: This equation is not in the book!
        Vector3d g = Vector3d::Zero();     // Bias, g(x) = -J(x).f(x)
        cost = 0.0;                        // Reset

        // Data Loop
        for(int i=0; i<N; i++){
            double xi = x_data[i], yi = y_data[i];  // the i-th data point
            
            /* ----- Compute Error ----- */
            // Residual = Y_real - Y_estimated = _measurement - h(_x, _estimate)
            double yi_e = exp(ae*xi*xi + be*xi + ce);
            double error = yi - yi_e;                // e(x) = y_r - y_e = y_r - exp(a.x^2+b.x+c)
            
            /* ----- Jacobians ----- */
            Vector3d J;        // Jacobian matrix of the error
            J[0] = -xi*xi*yi_e;  // de(x)/da
            J[1] = -xi*yi_e;     // de(x)/db
            J[2] = -yi_e;        // de(x)/dc

            H +=  inv_sigma * inv_sigma * J * J.transpose();  //FIXME: ?
            g += -inv_sigma * inv_sigma * J * error;  // -J(x)*f(x), f(x)=e(x)

            // Least Squares Cost
            cost += error * error;  // The actual error function being minimized in the loop is e(x)^2.
        }

        // Solve the Linear System Ax=b, H(x)*∆x = g(x)
        Vector3d dx = H.ldlt().solve(g);

        if(isnan(dx[0])){
            cout << "Result is nan!" << endl;
            break;
        }

        // Early Stop
        if(iter > 0 && cost >= lastCost){
            cout << "cost: " << cost << " >= lastCost: " << lastCost << ", break." << endl;
            break;
        }

        // Update
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
    Timer t2 = chrono::steady_clock::now();

    printTimeElapsed("Solver time: ", t1, t2);

    // Computes the RMSE
    Vector3d abc_r = {ar, br, cr};
    Vector3d abc_e = {ae, be, ce};
    
    double rmse = RMSE(abc_e, abc_r);

    /* ----- Results ----- */ 
    cout << "RMSE: " << rmse << endl;

    cout << "\n---" << endl;
    cout << "Real:\t   a,b,c = ";
    cout << ar << ", " << br << ", " << cr;
    cout << endl;

    cout << "Estimated: a,b,c = ";
    cout << ae << ", " << be << ", " << ce;
    cout << "\n---" <<endl;

    cout << "\nDone." << endl;

    return 0;
}
