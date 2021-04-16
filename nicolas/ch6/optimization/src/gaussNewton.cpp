/* Custom Libraries */
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
    cout << "[gaussNetwon] Hello!" << endl << endl;

    /* Variables */
    double ar = 1.0, br = 2.0, cr = 1.0;        // Real parameters values
    double ae = 2.0, be = -1.0, ce = 5.0;       // Estimated parameters values (x0, Initial Guess)
    int N = 100;                                // Number of Data points

    cv::RNG rng;                                // OpenCV Random Number generator
    double w_sigma = 1.0;                       // Noise sigma value, w ~ N(0,σ^2)
    double inv_sigma = 1.0 / w_sigma;

    /* ----- Data Generation ----- */
    vector<double> x_data, y_data;              // Data Vectors

    for(int i=0; i < N; i++){
        double x = i/100.0;
        double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma);  // y = exp(a.x^2+b.x+c) + w

        x_data.push_back(x);
        y_data.push_back(y);
    }

    printVec("x_data: ", x_data);
    printVec("y_data: ", y_data);

    /* ----- Gauss-Newton Method ----- */
    int iterations = 100;               // Number of iterations
    double cost = 0.0, lastCost = 0.0;  // The cost of current iteration and the cost of the previous one

    /* Iteration Loop */
    cout << "Summary: " << endl;
    Timer t1 = chrono::steady_clock::now();
    for(int iter=0; iter < iterations; iter++){
        Matrix3d H = Matrix3d::Zero();     // Hessian, H(x) = J(x)'*Ω*J(x) in Gauss-Newton
        Vector3d b = Vector3d::Zero();     // Bias, g(x) = -J(x).f(x)
        
        cost = 0.0;                        // Reset

        /* Data Loop, Compute Cost */
        for(int i=0; i<N; i++){
            double xi = x_data[i], yi_r = y_data[i];  // the i-th data point

            /* ----- Compute Error ----- */
            // Residual = Y_real - Y_estimated = _measurement - h(_x, _estimate)
            double yi_e = exp(ae*xi*xi + be*xi + ce);
            double ei = yi_r - yi_e;                // e(x) = y_r - y_e = y_r - exp(a.x^2+b.x+c)

            /* ----- Jacobians ----- */
            Vector3d J;          // Jacobian matrix of the error
            J[0] = -xi*xi*yi_e;  // de(x)/da
            J[1] = -xi*yi_e;     // de(x)/db
            J[2] = -yi_e;        // de(x)/dc

            /* ----- Hessian and Bias ----- */
            // The Slambook2 doesn't have the Gauss-Newton equation considering the information matrix (inverse of covariance)
            // The following equations are from Wangxin's Blog, http://wangxinliu.com/slam/optimization/research&study/g2o-3/
            H +=  inv_sigma * inv_sigma * J * J.transpose();  // Hessian, H(x) = J(x)'*Ω*J(x)
            b += -inv_sigma * inv_sigma * J * ei;             // Bias, g(x) = -b(x) = -Ω*J(x)*f(x), f(x)=e(x)

            // Least-Squares Cost (Objective Function)
            // This is the actual error function being minimized by solving the proposed linear system: min_x(sum_i ||ei(x)||^2).
            cost += ei * ei;  // Summation of the squared residuals.
        }

        /* ----- Solve ----- */
        // Solve the Linear System A*x=b, H(x)*∆x = g(x)
        Vector3d dx = H.ldlt().solve(b);  // ∆x

        // Check Solution
        if(isnan(dx[0])){
            cout << "Update is nan!" << endl;
            break;
        }

        /* Stopping Criteria */
        if (dx.norm() < 1e-6 || (iter > 0 && cost >= lastCost)){
            // If the cost increased, the update was not good.
            cout << "cost: " << cost << " >= lastCost: " << lastCost << ", break." << endl;
            break;
        }

        /* Update */
        // x_k+1 = x_k + ∆x_k
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        cout << "it: " << iter << ",\tcost: " << cost << ",\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    // Computes the RMSE
    Vector3d abc_r = {ar, br, cr};
    Vector3d abc_e = {ae, be, ce};

    double rmse = RMSE(abc_e, abc_r);

    /* ----- Results ----- */
    cout << "\n---" << endl;
    cout << "Real:\t   a,b,c = ";
    cout << ar << ", " << br << ", " << cr;
    cout << endl;

    cout << "Estimated: a,b,c = ";
    cout << ae << ", " << be << ", " << ce;
    cout << "\n---" <<endl;

    cout << "RMSE: " << rmse << endl;

    cout << "\nDone." << endl;

    return 0;
}
