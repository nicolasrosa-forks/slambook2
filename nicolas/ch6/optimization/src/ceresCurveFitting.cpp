/* Libraries */
#include "../include/ceresCurveFitting.h"

using namespace std;

/* Functor */
/* This is a functor, which represents the objective function to be minimized.
   
   1. A primeira linha só está guardando os valores dentro dos campos da struct mesmo;
   2. A segunda parte define o que deve ser feito quando o operador () é chamado (Invocado internamente na função ceres::AutoDiffCostFunction?);
   3. Correspondem às variáveis privadas da struct, sendo preenchidas com o construtor CURVE_FITTING_COST(double x, double y).
   
*/
struct CURVE_FITTING_COST{
    // Constructor
    CURVE_FITTING_COST(double x, double y): _x(x), _y(y) {}

    // Calculation of residuals
    template<typename T>
    bool operator()(const T *const abc_e, // model parameters, there are 3 dimensions.
    T *residual) const {
        // Residual = Y_real - Y_estimated
        residual[0] = T(_y) - ceres::exp(abc_e[0]*T(_x)*T(_x) + abc_e[1]*T(_x) + abc_e[2]);  // err = y-y^ = y-exp(a.x^2+b.c+c)
        
        return true;
    }

    // Private attributes
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
    double w_sigma = 1.0;                       // Noise sigma value

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

    /* ----- Least Squares Estimation ----- */
    // Construct a least squares problem
    ceres::Problem problem;        

    // Calculation of residuals
    for(int i=0; i<N; i++){
        // Add error term to the problem
        problem.AddResidualBlock(
            /* 1. Use automatic derivation 
                Template parameters: error type, output dimension, input dimension. 
                  Dimensions should be consistent with the passed struct.
                Function parameters:
                  Quando o código faz "new CURVE_FITTING_COST(x_data[i], y_data[i])",
                  ele está apenas criando o functor "CURVE_FITTING_COST" utilizando o construtor dele.

                O operador () da "CURVE_FITTING_COST" será chamado internamente pelo AutoDiffCostFunction
            */
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])),
            
            /* 2. O "nullptr" é porque não temos função de perda, só mínimos quadrados mesmo. */
            nullptr,  // Core function, not used here, empty

            /* 3. "abc_e" são os parâmetros a serem estimados */
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

    printTimeElapsed("Solver time: ", t1, t2);

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