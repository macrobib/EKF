#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
#ifdef DEBUG_LOG
    std::cout <<"Predict: Enter"<< std::endl;
#endif
    x_ = F_ * x_;
    std::cout << "Predicted x:"<< x_ << std::endl;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
#ifdef DEBUG_LOG
    std::cout <<"Predict: Exit."<< std::endl;
#endif
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
#ifdef DEBUG_LOG
    std::cout << "Update EKF Start."<< std::endl;
#endif
    VectorXd hx(3);
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);

    double rho = sqrt(px*px + py*py);
    if(rho < 0.00001){
        px += 0.001;
        py += 0.001;
        rho = sqrt(px*px + py*py);
    }
    double theta = atan2(py, px);
    std::cout<< "Theta: " << theta<< std::endl;
    double rho_rate = (px*vx + py*vy)/rho;

    hx << rho, theta, rho_rate;
    VectorXd y = z - hx;
 	while ( y(1) > M_PI || y(1) < -M_PI ) {
    if ( y(1) > M_PI ) {
      y(1) -= M_PI;
    } else {
      y(1) += M_PI;
    }
  }
    Update(y);
#ifdef DEBUG_LOG
    std::cout << "Update EKF End."<< std::endl;
#endif
}

void KalmanFilter::UpdateKF(const VectorXd &z){
#ifdef DEBUG_LOG
    std::cout << "Update KF Start."<< std::endl;
#endif
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    Update(y);
#ifdef DEBUG_LOG
    std::cout << "Update KF End."<< std::endl;
#endif
}

void KalmanFilter::Update(const VectorXd &y) {
  /**
    * update the state by using Kalman Filter equations
  */
#ifdef DEBUG_LOG
    std::cout<< "Common update: Start" << std::endl;
#endif
    MatrixXd Ht = H_.transpose();
    MatrixXd S = (H_ * P_ * Ht) + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // new estimate.
    x_ = x_  + (K*y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K*H_) * P_;
#ifdef DEBUG_LOG
    std::cout<< "Common update: End" << std::endl;
#endif
}
