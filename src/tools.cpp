#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if(estimations.size() != ground_truth.size()){
        std::cout << "Estimation and ground truth size don't match."<<std::endl;
    }
    else{
        for(auto i = 0; i < estimations.size(); ++i){
            VectorXd residual = estimations[i] - ground_truth[i];
            residual = residual.array()  * residual.array();
            rmse += residual;
        }

        rmse = rmse/estimations.size();

        rmse = rmse.array().sqrt();
    }

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
#ifdef DEBUG_LOG
    std::cout << "Jacobian: Enter"<< std::endl;
#endif
    MatrixXd Hj(3, 4);
    //State variables.
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float sqr_sum = px*px + py*py;
    if(sqr_sum < 0.00001){
        px += 0.001;
        py += 0.001;
        sqr_sum = px*px + py*py;
    }
    float sum_sqrt = sqrt(sqr_sum);
    float sum_sqrt_cube = sqr_sum * sum_sqrt;
    float prd_1 = vx*py - vy*px; 
    float prd_2 = vy*px - vx*py; 

    Hj << px/sum_sqrt, py/sum_sqrt, 0, 0,
          -py/sqr_sum, px/sqr_sum, 0, 0,
         (py*prd_1)/sum_sqrt_cube, (px*prd_2)/sum_sqrt_cube, px/sum_sqrt, py/sum_sqrt;

#ifdef DEBUG_LOG
    std::cout << "Jacobian: Exit"<< std::endl;
#endif
    return Hj;
}
