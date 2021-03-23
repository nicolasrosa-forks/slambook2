/* Custom Libraries */
#include "../include/ceresCurveFitting.h"

using namespace std;

/* Functor */
/* This is a functor, which represents the objective function to be minimized.
    1. The first line just stores the values within the attributes (private variables) of the struct;
    2. The second part defines what should be done when the operator () is called (which is invoked internally in the ceres::AutoDiffCostFunction? Function);
    3. They correspond to the private variables of the struct, being filled with the CURVE_FITTING_COST constructor (double x, double y).

*/
struct CURVE_FITTING_COST{
    // 1. Constructor
    CURVE_FITTING_COST(double x, double y): _x(x), _y(y) {}

    // 2. Calculation of residuals
    template<typename T>
    bool operator()(const T *const abc_e, // model parameters, there are 3 dimensions.
    T *residual) const {
        // Residual = Y_real - Y_estimated
        residual[0] = T(_y) - ceres::exp(abc_e[0]*T(_x)*T(_x) + abc_e[1]*T(_x) + abc_e[2]);  // e(x) = y_r - y_e = y_r - exp(a.x^2+b.x+c)


        return true;  /* "After Ceres sums the squares of them (residuals), it is used as the value of the objective function." */
    }

    // 3. Private attributes
    const double _x, _y;    // x,y data
};

double RMSE(const double est[], const double gt[]){
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
    cout << "[ceresCurveFitting] Hello!" << endl<< endl;

    /* Variables */
    double ar = 1.0, br =  2.0, cr = 1.0;       // Real parameters values
    double ae = 2.0, be = -1.0, ce = 5.0;       // Estimated parameters values

    double abc_r[3] = {ar, br, cr};             // Vector with the real values
    double abc_e[3] = {ae, be, ce};             // Vector with the values to be estimated

    int N = 100;                                // Number of Data points

    cv::RNG rng;                                // OpenCV Random Number generator
    double w_sigma = 1.0;                       // Noise sigma value, w ~ N(0,σ^2)

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

    /* ----- Least Squares Estimation ----- */
    // Construct a least squares problem
    ceres::Problem problem;

    // Residual Block Definition
    for(int i=0; i<N; i++){
        // Adds each error term to the objective function
        problem.AddResidualBlock(
            /* 1. Use automatic derivation
                Template parameters: error type, output dimension, input dimension.
                  Dimensions should be consistent with the passed struct.
                Function parameters:
                  When the code run "new CURVE_FITTING_COST(x_data[i], y_data[i])",
                  it's just creating the functor "CURVE_FITTING_COST" by using its the declared constructor.

                The operator () of "CURVE_FITTING_COST" will be called internally by the "ceres::AutoDiffCostFunction"

                "In fact, Ceres will pass the Jacobian matrix as a type parameter to this function to realize the
                function of automatic derivation (auto-diff, which is one of the best feature of Ceres)."
            */
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),

            /* 2. The reason of "nullptr" is due to we don't have a loss function, just the least squares equation. */
            nullptr,  // Core function, not used here, empty

            /* 3. Parameter Block */
            abc_e);  // Parameters to be estimated
    }

    // Configure the solver
    ceres::Solver::Options options;                             // There are many configuration items to fill in here
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // How to solve the incremental equation
    options.minimizer_progress_to_stdout = true;                // Output to std::cout

    // Optimization information
    ceres::Solver::Summary summary;

    /* ----- Solve ----- */
    print("Summary: ");
    Timer t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    // Computes the RMSE
    double rmse = RMSE(abc_e, abc_r);

    /* ----- Results ----- */
    print(summary.BriefReport());

    cout << "\n---" << endl;
    cout << "Real:\t   a,b,c = ";
    for(auto item:abc_r) cout << item << ", ";
    cout << endl;

    cout << "Estimated: a,b,c = ";
    for(auto item:abc_e) cout << item << ", ";
    cout << "\n---" <<endl;

    cout << "RMSE: " << rmse << endl;

    cout << "\nDone." << endl;

    return 0;
}