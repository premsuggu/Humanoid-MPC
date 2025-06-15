#include "mpc_utils.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp> 
#include <iostream>

MPCUtils::MPCUtils(int N, int nq, int nv, int n_ee, double dt,
                   const std::vector<std::string>& ee_names,
                   RobotModel* robot_model)
    : N_(N), nq_(nq), nv_(nv), n_ee_(n_ee), dt_(dt),
      ee_names_(ee_names), robot_model_(robot_model)
{
    IntegrationFunction();
    createDynamicsFunctions();  
}


void MPCUtils::IntegrationFunction() {
    using namespace pinocchio;
    
    // Create symbolic variables
    ::casadi::SX cs_q = ::casadi::SX::sym("q", nq_);
    ::casadi::SX cs_v = ::casadi::SX::sym("v", nv_);  // Just velocity, not v*dt
    ::casadi::SX cs_dt = ::casadi::SX::sym("dt", 1);  // Time step as parameter
    
    // Convert to Pinocchio autodiff types
    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef typename ADModel::ConfigVectorType ConfigVectorAD;
    typedef typename ADModel::TangentVectorType TangentVectorAD;
    
    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    
    ConfigVectorAD q_ad(nq_);
    TangentVectorAD v_ad(nv_);
    
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = cs_q(i);
    }
    for (int i = 0; i < nv_; ++i) {
        v_ad[i] = cs_v(i);
    }
    
    // Perform integration: q_next = integrate(q, v * dt)
    ConfigVectorAD q_next_ad(nq_);
    TangentVectorAD v_dt_ad = v_ad * cs_dt(0);  // Scale velocity by dt inside
    integrate(ad_model, q_ad, v_dt_ad, q_next_ad);
    
    // Convert result back to CasADi SX
    ::casadi::SX cs_q_next = ::casadi::SX::zeros(nq_, 1);
    for (int i = 0; i < nq_; ++i) {
        cs_q_next(i) = q_next_ad[i];
    }
    
    // Create CasADi function with dt as parameter
    casadi_integrate_ = ::casadi::Function("integrate", 
                                          {cs_q, cs_v, cs_dt}, 
                                          {cs_q_next});
}

void MPCUtils::createSymbolicVariables() {
    q_sym_.resize(N_ + 1);  
    v_sym_.resize(N_ + 1);  
    a_sym_.resize(N_);      
    f_sym_.resize(N_);      

    //PARAMETERS
    q_ref_sym_.clear(); v_ref_sym_.clear(); a_ref_sym_.clear();
    zmp_ref_sym_.clear(); com_ref_sym_.clear();
    ee_pos_ref_sym_.clear(); ee_ori_ref_sym_.clear();

    for (int k = 0; k <= N_; ++k) {
        q_ref_sym_.push_back(::casadi::SX::sym("q_ref_" + std::to_string(k), nq_));
        v_ref_sym_.push_back(::casadi::SX::sym("v_ref_" + std::to_string(k), nv_));
        com_ref_sym_.push_back(::casadi::SX::sym("com_ref_" + std::to_string(k), 3));
    }
    for (int k = 0; k < N_; ++k) {
        a_ref_sym_.push_back(::casadi::SX::sym("a_ref_" + std::to_string(k), nv_));
        zmp_ref_sym_.push_back(::casadi::SX::sym("zmp_ref_" + std::to_string(k), 2));
        std::vector<::casadi::SX> ee_pos_k, ee_ori_k;
        for (int ee = 0; ee < n_ee_; ++ee) {
            ee_pos_k.push_back(::casadi::SX::sym("ee_pos_ref_" + std::to_string(k) + "_" + std::to_string(ee), 3));
            ee_ori_k.push_back(::casadi::SX::sym("ee_ori_ref_" + std::to_string(k) + "_" + std::to_string(ee), 4));
        }
        ee_pos_ref_sym_.push_back(ee_pos_k);
        ee_ori_ref_sym_.push_back(ee_ori_k);
    }

    // Accelerations & contact forces (optimization variables)
    int max_contacts = n_ee_;

    for (int k = 0; k < N_; ++k) {
        a_sym_[k] = ::casadi::SX::sym("a_" + std::to_string(k), nv_);
        f_sym_[k] = ::casadi::SX::sym("f_" + std::to_string(k), 6 * max_contacts);
    }

    // Initial conditions
    q_sym_[0] = ::casadi::SX::sym("q0", nq_);
    v_sym_[0] = ::casadi::SX::sym("v0", nv_);
    
    for (int k = 0; k < N_; ++k) {
        v_sym_[k+1] = v_sym_[k] + dt_ * a_sym_[k];
        
        // CORRECT: Pass velocity and dt separately
        q_sym_[k+1] = casadi_integrate_(std::vector<::casadi::SX>{
            q_sym_[k], 
            v_sym_[k+1], 
            ::casadi::SX(dt_)
        })[0];
    }
}


void MPCUtils::setContactSchedule(const ContactSchedule& schedule) {
    contact_schedule_ = schedule;
}

void MPCUtils::setBounds(const ::casadi::DM& lbx, const ::casadi::DM& ubx,
                         const ::casadi::DM& lbg, const ::casadi::DM& ubg) {
    lbx_ = lbx;
    ubx_ = ubx;
    lbg_ = lbg;
    ubg_ = ubg;
    std::cout << "Bounds set: " << lbx.size1() << " variables, " << lbg.size1() << " constraints" << std::endl;
}


// Helper functions for dynamics computation
void MPCUtils::createDynamicsFunctions() {
    using namespace pinocchio;
    
    std::cout << "Creating dynamics functions..." << std::endl;
    
    // Create symbolic variables
    ::casadi::SX cs_q = ::casadi::SX::sym("q", nq_);
    ::casadi::SX cs_v = ::casadi::SX::sym("v", nv_);
    
    // Convert to Pinocchio autodiff types
    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef typename ADModel::Data ADData;
    typedef typename ADModel::ConfigVectorType ConfigVectorAD;
    typedef typename ADModel::TangentVectorType TangentVectorAD;
    
    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);
    
    // Convert CasADi SX to Pinocchio AD types
    ConfigVectorAD q_ad(nq_);
    TangentVectorAD v_ad(nv_);
    
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = cs_q(i);
    }
    for (int i = 0; i < nv_; ++i) {
        v_ad[i] = cs_v(i);
    }
    
    //-----------------------------------------------------------------------
    // 1. CREATE MASS MATRIX FUNCTION
    crba(ad_model, ad_data, q_ad);
    ::casadi::SX cs_M = ::casadi::SX::zeros(nv_, nv_);
    for (int i = 0; i < nv_; ++i) {
        for (int j = 0; j < nv_; ++j) {
            cs_M(i, j) = ad_data.M(i, j);
        }
    }
    mass_matrix_func_ = ::casadi::Function("mass_matrix", {cs_q}, {cs_M});
    
    //-----------------------------------------------------------------------
    // 2. CREATE CORIOLIS FUNCTION
    rnea(ad_model, ad_data, q_ad, v_ad, TangentVectorAD::Zero(nv_));
    ::casadi::SX cs_C_v = ::casadi::SX::zeros(nv_, 1);
    for (int i = 0; i < nv_; ++i) {
        cs_C_v(i) = ad_data.tau[i];
    }
    coriolis_func_ = ::casadi::Function("coriolis", {cs_q, cs_v}, {cs_C_v});
    
    //-----------------------------------------------------------------------
    // 3. CREATE GRAVITY FUNCTION
    rnea(ad_model, ad_data, q_ad, TangentVectorAD::Zero(nv_), TangentVectorAD::Zero(nv_));
    ::casadi::SX cs_g = ::casadi::SX::zeros(nv_, 1);
    for (int i = 0; i < nv_; ++i) {
        cs_g(i) = ad_data.tau[i];
    }
    gravity_func_ = ::casadi::Function("gravity", {cs_q}, {cs_g});

    //-----------------------------------------------------------------------
    // 4. Contact Jacobian Functions
    for (const std::string& ee_name : ee_names_) {
        FrameIndex frame_id = robot_model_->getFrameId(ee_name);
        
        // First compute joint Jacobians, then get frame Jacobian
        computeJointJacobians(ad_model, ad_data, q_ad);
        updateFramePlacements(ad_model, ad_data);
        
        // Create output matrix for Jacobian
        Eigen::Matrix<ADScalar, 6, Eigen::Dynamic> J_frame(6, nv_);
        J_frame.setZero();
        
        // Use getFrameJacobian after computeJointJacobians
        getFrameJacobian(ad_model, ad_data, frame_id, LOCAL_WORLD_ALIGNED, J_frame);
        
        // Convert to CasADi SX
        ::casadi::SX cs_J = ::casadi::SX::zeros(6, nv_);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < nv_; ++j) {
                cs_J(i, j) = J_frame(i, j);
            }
        }
        
        // Create function for this end-effector
        contact_jacobian_funcs_[ee_name] = ::casadi::Function(
            "jacobian_" + ee_name, {cs_q}, {cs_J});
    }
    
    //-------------------------------END-------------------------------------
    std::cout << "Dynamics functions created successfully!" << std::endl;
}

::casadi::SX MPCUtils::computeMassMatrix(const ::casadi::SX& q) const {
    // Use pre-computed CasADi function
    return mass_matrix_func_({q})[0];
}

::casadi::SX MPCUtils::computeCoriolisMatrix(const ::casadi::SX& q, const ::casadi::SX& v) const {
    // Use pre-computed CasADi function
    std::vector<::casadi::SX> inputs = {q, v};
    return coriolis_func_(inputs)[0];
}

::casadi::SX MPCUtils::computeGravityVector(const ::casadi::SX& q) const {
    // Use pre-computed CasADi function
    return gravity_func_({q})[0];
}

::casadi::SX MPCUtils::computeContactJacobian(int k, const std::string& ee_name) const {
    // Use pre-computed CasADi function
    return contact_jacobian_funcs_.at(ee_name)({q_sym_[k]})[0];
}


// End-effector position computation (based on integrated states)
::casadi::SX MPCUtils::ee_pos_sym(int k, const std::string& ee_name) const {
    using namespace pinocchio;
    
    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;
    
    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);
    
    typedef ADModel::ConfigVectorType ConfigVectorAD;
    ConfigVectorAD q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = q_sym_[k](i);
    }
    
    // Compute forward kinematics
    forwardKinematics(ad_model, ad_data, q_ad);
    updateFramePlacements(ad_model, ad_data);
    
    // Get frame position
    FrameIndex frame_id = robot_model_->getFrameId(ee_name);
    auto pos = ad_data.oMf[frame_id].translation();
    
    ::casadi::SX ee_pos = ::casadi::SX::zeros(3, 1);
    for (int i = 0; i < 3; ++i) {
        ee_pos(i) = pos[i];
    }
    
    return ee_pos;
}


::casadi::SX MPCUtils::ee_ori_sym(int k, const std::string& ee_name) const {
    using namespace pinocchio;

    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;

    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);

    typedef ADModel::ConfigVectorType ConfigVectorAD;
    ConfigVectorAD q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = q_sym_[k](i);
    }

    forwardKinematics(ad_model, ad_data, q_ad);
    updateFramePlacements(ad_model, ad_data);

    FrameIndex frame_id = robot_model_->getFrameId(ee_name);
    auto rot = ad_data.oMf[frame_id].rotation();

    // Pinocchio provides quaternion conversion for Eigen matrices, but not directly for CasADi SX.
    // So, use the manual conversion as you had:
    ::casadi::SX trace = rot(0,0) + rot(1,1) + rot(2,2);
    ::casadi::SX quat = ::casadi::SX::zeros(4, 1);

    ::casadi::SX w = ::casadi::SX::sqrt(1.0 + trace) / 2.0;
    ::casadi::SX x = (rot(2,1) - rot(1,2)) / (4.0 * w);
    ::casadi::SX y = (rot(0,2) - rot(2,0)) / (4.0 * w);
    ::casadi::SX z = (rot(1,0) - rot(0,1)) / (4.0 * w);

    quat(0) = w; quat(1) = x; quat(2) = y; quat(3) = z;
    return quat; // [w, x, y, z]
}


// Center of Mass computation
::casadi::SX MPCUtils::com_position_sym(int k) const {
    using namespace pinocchio;
    
    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;
    
    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);
    
    typedef ADModel::ConfigVectorType ConfigVectorAD;
    ConfigVectorAD q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = q_sym_[k](i);
    }
    
    // Compute center of mass
    centerOfMass(ad_model, ad_data, q_ad);
    
    ::casadi::SX com = ::casadi::SX::zeros(3, 1);
    for (int i = 0; i < 3; ++i) {
        com(i) = ad_data.com[0](i);
    }
    
    return com;
}


// ZMP computation from accelerations
::casadi::SX MPCUtils::zmp_sym(int k) const {
    using namespace pinocchio;
    
    typedef ::casadi::SX ADScalar;
    typedef ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;
    
    const Model& model = robot_model_->getModel();
    ADModel ad_model = model.cast<ADScalar>();
    ADData ad_data(ad_model);
    
    typedef ADModel::ConfigVectorType ConfigVectorAD;
    ConfigVectorAD q_ad(nq_);
    for (int i = 0; i < nq_; ++i) {
        q_ad[i] = q_sym_[k](i);
    }
    
    // Compute forward kinematics first (required for CoM Jacobian)
    forwardKinematics(ad_model, ad_data, q_ad);
    updateFramePlacements(ad_model, ad_data);
    
    // Compute CoM Jacobian using Pinocchio's jacobianCenterOfMass
    jacobianCenterOfMass(ad_model, ad_data, q_ad);
    
    // Extract CoM position and Jacobian
    ::casadi::SX com_pos = ::casadi::SX::zeros(3, 1);
    for (int i = 0; i < 3; ++i) {
        com_pos(i) = ad_data.com[0](i);
    }
    
    // Extract CoM Jacobian (3x nv matrix)
    ::casadi::SX J_com = ::casadi::SX::zeros(3, nv_);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < nv_; ++j) {
            J_com(i, j) = ad_data.Jcom(i, j);
        }
    }
    
    ::casadi::SX com_ddot = ::casadi::SX::mtimes(J_com, a_sym_[k]);
    
    ::casadi::SX zmp = ::casadi::SX::zeros(2, 1);
    ::casadi::SX g = 9.81;
    
    // ZMP formula: ZMP = CoM_xy - (CoM_z / (CoM_ddot_z + g)) * CoM_ddot_xy
    ::casadi::SX denom = com_ddot(2) + g;
    
    // To avoid division by zero with a small epsilon
    ::casadi::SX eps = 1e-4;
    ::casadi::SX safe_denom = denom + eps;
    
    zmp(0) = com_pos(0) - (com_pos(2) / safe_denom) * com_ddot(0);
    zmp(1) = com_pos(1) - (com_pos(2) / safe_denom) * com_ddot(1);
    
    return zmp;
}


// Torque computation in terms of acceleration (symbolic)
::casadi::SX MPCUtils::torque_sym(int k) const {
    // τ = M(q) * a + C(q,v) * v + g(q) - J^T * f_contact
    ::casadi::SX M = computeMassMatrix(q_sym_[k]);
    ::casadi::SX C_v = computeCoriolisMatrix(q_sym_[k], v_sym_[k]);
    ::casadi::SX g = computeGravityVector(q_sym_[k]);
    
    ::casadi::SX contact_contribution = ::casadi::SX::zeros(nv_, 1);

    for (int ee = 0; ee < n_ee_; ++ee) {
        ::casadi::SX J_contact = computeContactJacobian(k, ee_names_[ee]);
        
        ::casadi::SX f_contact = f_sym_[k](::casadi::Slice(ee*6, (ee+1)*6));
        contact_contribution += ::casadi::SX::mtimes(J_contact.T(), f_contact);
    }
    
    ::casadi::SX tau = ::casadi::SX::mtimes(M, a_sym_[k]) + C_v + g - contact_contribution;
    return tau;
}


// Build cost function
::casadi::SX MPCUtils::buildCost(
    const ::casadi::SX& Qq,
    const ::casadi::SX& Qv,
    const ::casadi::SX& Ra,
    const ::casadi::SX& Qf,
    const ::casadi::SX& Wee,
    const ::casadi::SX& Wzmp,
    const ::casadi::SX& Wcom,
    const ::casadi::SX& W_tau
) const {
    ::casadi::SX cost = 0;    
    std::cout << "Building cost function..." << std::endl;
    // Running costs
    for (int k = 0; k < N_; ++k) {
        // 1. Position tracking cost
        ::casadi::SX q_error = q_sym_[k] - ::casadi::SX(q_ref_sym_[k]);
        cost += ::casadi::SX::mtimes({q_error.T(), Qq, q_error});
        
        // 2. Velocity tracking cost
        ::casadi::SX v_error = v_sym_[k] - ::casadi::SX(v_ref_sym_[k]);
        cost += ::casadi::SX::mtimes({v_error.T(), Qv, v_error});
        
        // 3. Acceleration regularization (main optimization variables)
        ::casadi::SX a_error = a_sym_[k] - ::casadi::SX(a_ref_sym_[k]);
        cost += ::casadi::SX::mtimes({a_error.T(), Ra, a_error});
        
        // 4. End-effector tracking costs
        for (int ee = 0; ee < n_ee_; ++ee) {
            // Position tracking
            ::casadi::SX ee_pos_error = ee_pos_sym(k, ee_names_[ee]) - ::casadi::SX(ee_pos_ref_sym_[k][ee]);
            cost += ::casadi::SX::mtimes({ee_pos_error.T(), Wee, ee_pos_error});
            
            // Orientation tracking 
            ::casadi::SX ee_ori_error = ee_ori_sym(k, ee_names_[ee]) - ::casadi::SX(ee_ori_ref_sym_[k][ee]);
            cost += 0.1 * ::casadi::SX::mtimes({ee_ori_error.T(), ee_ori_error});
        }
        
        // 5. ZMP tracking cost
        ::casadi::SX zmp_error = zmp_sym(k) - ::casadi::SX(zmp_ref_sym_[k]);
        cost += ::casadi::SX::mtimes({zmp_error.T(), Wzmp, zmp_error});

        // 6. CoM tracking cost 
        ::casadi::SX com_error = com_position_sym(k) - ::casadi::SX(com_ref_sym_[k]);
        cost += ::casadi::SX::mtimes({com_error.T(), Wcom, com_error});


        // Torque Regularization
        ::casadi::SX tau = torque_sym(k);
        cost += ::casadi::SX::mtimes({tau.T(),W_tau ,tau});
    }
    
    // Terminal cost
    ::casadi::SX q_final_error = q_sym_[N_] - ::casadi::SX(q_ref_sym_[N_]);
    ::casadi::SX v_final_error = v_sym_[N_] - ::casadi::SX(v_ref_sym_[N_]);
    cost += ::casadi::SX::mtimes({q_final_error.T(), Qf, q_final_error});
    cost += ::casadi::SX::mtimes({v_final_error.T(), 5*Qv, v_final_error});
    
    std::cout << "Cost function built successfully" << std::endl;
    return cost;
}

// Build constraints
::casadi::SX MPCUtils::buildConstraints(
    const ::casadi::SX& zmp_x_min,
    const ::casadi::SX& zmp_x_max,
    const ::casadi::SX& zmp_y_min,
    const ::casadi::SX& zmp_y_max,
    const std::vector<double>& v_min,
    const std::vector<double>& v_max,
    const std::vector<double>& tau_max,
    const std::vector<double>& tau_min,
    double min_clearance,
    double mu,
    const std::vector<std::pair<std::string, std::string>>& collision_avoidance_pairs
) const {
    std::vector<::casadi::SX> constraints;
    std::cout << "Building constraints..." << std::endl;
    
    
    for (int k = 0; k < N_; ++k) {        
        // 1. Joint position limits: q_min <= q_k <= q_max
        std::vector<double> q_min = robot_model_->getLowerLimits();
        std::vector<double> q_max = robot_model_->getUpperLimits();
        
        for (int i = 0; i < nq_; ++i) {
            constraints.push_back(q_sym_[k](i) - q_min[i]);  // q >= q_min          //>=0
            constraints.push_back(q_max[i] - q_sym_[k](i));  // q <= q_max          //>=0
        }

        // 2. Velocity limits: v_min <= v_k <= v_max
        for (int i = 0; i < nv_; ++i) {
            constraints.push_back(v_max[i] - v_sym_[k](i));  // >=0
            constraints.push_back(v_sym_[k](i) - v_min[i]);  // >=0
        }

        // 3. Torque limits: tau_min <= tau_k <= tau_max
        ::casadi::SX tau = torque_sym(k);
        for (int i = 6; i < nv_; ++i) {
            constraints.push_back(tau(i) - tau_min[i]);  // tau >= tau_min
            constraints.push_back(tau_max[i] - tau(i));  // tau <= tau_max
        }

        // 4. ZMP constraints (INEQUALITY)
        ::casadi::SX zmp = zmp_sym(k);
        constraints.push_back(zmp(0) - zmp_x_min);  // zmp_x >= zmp_x_min
        constraints.push_back(zmp_x_max - zmp(0));  // zmp_x <= zmp_x_max
        constraints.push_back(zmp(1) - zmp_y_min);  // zmp_y >= zmp_y_min
        constraints.push_back(zmp_y_max - zmp(1));  // zmp_y <= zmp_y_max
        
        // 5. Swing foot clearance constraints
        for (int ee = 0; ee < 2; ++ee) {  // Only feet (first 2 end-effectors)
            if (k < contact_schedule_.size() && !contact_schedule_[k][ee]) {            // Swing foot
                ::casadi::SX foot_pos = ee_pos_sym(k, ee_names_[ee]);
                constraints.push_back(foot_pos(2) - min_clearance);  // foot_z >= min_clearance
            }
        }
        
        // // 6. Contact height constraints for stance feet
        // for (int ee = 0; ee < 2; ++ee) {                                     // Only feet
        //     if (k < contact_schedule_.size() && contact_schedule_[k][ee]) {  // Stance foot
        //         ::casadi::SX foot_pos = ee_pos_sym(k, ee_names_[ee]);
        //         constraints.push_back(foot_pos(2) - (-0.01));  // foot_z >= -1cm (ground contact)
        //         constraints.push_back(0.01 - foot_pos(2));     // foot_z <= 1cm (stay on ground)
        //     }
        // }

        // 7. Friction cone constraints for stance feet
        for (int ee = 0; ee < n_ee_; ++ee) {
            if (k < contact_schedule_.size() && contact_schedule_[k][ee]) {
                ::casadi::SX f_contact = f_sym_[k](::casadi::Slice(ee*6, (ee+1)*6));
                
                ::casadi::SX fx = f_contact(0);
                ::casadi::SX fy = f_contact(1);
                ::casadi::SX fz = f_contact(2);
                
                // Friction cone constraints
                constraints.push_back(fx - mu * fz);      // fx <= mu*fz
                constraints.push_back(-fx - mu * fz);     // -fx <= mu*fz
                constraints.push_back(fy - mu * fz);      // fy <= mu*fz
                constraints.push_back(-fy - mu * fz);     // -fy <= mu*fz
                // constraints.push_back(fz - 10.0);        // fz >= 10N (minimum normal force)
            }
        }


        // 8. No-slip constraints for stance feet: J_contact * v = 0
        for (int ee = 0; ee < 2; ++ee) {  // Only feet
            if (k < contact_schedule_.size() && contact_schedule_[k][ee]) {                 // Stance foot
                ::casadi::SX J_contact = computeContactJacobian(k, ee_names_[ee]);
                ::casadi::SX contact_velocity = ::casadi::SX::mtimes(J_contact, v_sym_[k]);
                
                // No slip in x and y directions (allow small tolerance)
                double slip_tolerance = 0.01;           // 1cm/s
                constraints.push_back(contact_velocity(0) - (-slip_tolerance));  // vx >= -tol
                constraints.push_back(slip_tolerance - contact_velocity(0));     // vx <= tol
                constraints.push_back(contact_velocity(1) - (-slip_tolerance));  // vy >= -tol
                constraints.push_back(slip_tolerance - contact_velocity(1));     // vy <= tol
            }
        }

        // 9. Collision avoidance constraints
        for (const auto& pair : collision_avoidance_pairs) {
            const std::string& ee1 = pair.first;
            const std::string& ee2 = pair.second;
            
            ::casadi::SX pos1 = ee_pos_sym(k, ee1);
            ::casadi::SX pos2 = ee_pos_sym(k, ee2);
            
            ::casadi::SX dist_sq = (pos1(0) - pos2(0)) * (pos1(0) - pos2(0)) +
                                   (pos1(1) - pos2(1)) * (pos1(1) - pos2(1)) +
                                   (pos1(2) - pos2(2)) * (pos1(2) - pos2(2));
            
            constraints.push_back(dist_sq - min_clearance * min_clearance);  // dist^2 - min_clearance^2 >= 0
        }

        // 10. CoM height constraint (prevents robot from falling)
        // ::casadi::SX com_pos = com_position_sym(k);
        // constraints.push_back(com_pos(2) - 0.8);   // CoM height >= 80cm
        // constraints.push_back(1.2 - com_pos(2));   // CoM height <= 120cm
        
        // // 11. Base orientation limits (prevent robot from tipping over)
        // // Assuming quaternion representation for base orientation
        // if (nq_ > 6) {  // If using quaternion for base
        //     ::casadi::SX roll = q_sym_[k](3);   // Roll angle
        //     ::casadi::SX pitch = q_sym_[k](4);  // Pitch angle
            
        //     double max_tilt = 0.3;  // ~17 degrees max tilt
        //     constraints.push_back(roll - (-max_tilt));   // roll >= -max_tilt
        //     constraints.push_back(max_tilt - roll);      // roll <= max_tilt
        //     constraints.push_back(pitch - (-max_tilt));  // pitch >= -max_tilt
        //     constraints.push_back(max_tilt - pitch);     // pitch <= max_tilt
        // }
    }

    // 11. Equality constraints
    for (int k = 0; k < N_; ++k){
    
        for (int ee = 0; ee < n_ee_; ++ee) {
            if (k < contact_schedule_.size() && !contact_schedule_[k][ee]) {
                // Force non-contact end-effectors to have zero force
                ::casadi::SX f_contact = f_sym_[k](::casadi::Slice(ee*6, (ee+1)*6));
                
                // All 6 components must be zero for non-contact
                for (int i = 0; i < 6; ++i) {
                    constraints.push_back(f_contact(i));      // f = 0 (equality)
                }
            }
        }
    }

    std::cout << "Built " << constraints.size() << " constraints" << std::endl;
    
    if (constraints.empty()) {
        return ::casadi::SX::zeros(1, 1);
    }
    
    return ::casadi::SX::vertcat(constraints);
}


void MPCUtils::setupSolver(const ::casadi::SX& cost, const ::casadi::SX& constraints, 
                           const ReferenceTrajectories& refs) {
    std::cout << "Setting up solver..." << std::endl;
    
    std::vector<::casadi::SX> opt_vars;
    for (int k = 0; k < N_; ++k) {
        opt_vars.push_back(a_sym_[k]);
    }
    for (int k = 0; k < N_; ++k) {
        opt_vars.push_back(f_sym_[k]);
    }
    ::casadi::SX x_opt = ::casadi::SX::vertcat(opt_vars);

    // Parameters = initial conditions + references
    std::vector<::casadi::SX> params = {q_sym_[0], v_sym_[0]};
    std::vector<::casadi::SX> param_vec;
    param_vec.push_back(q_sym_[0]);
    param_vec.push_back(v_sym_[0]);
    for (int k = 0; k <= N_; ++k) param_vec.push_back(q_ref_sym_[k]);
    for (int k = 0; k <= N_; ++k) param_vec.push_back(v_ref_sym_[k]);
    for (int k = 0; k < N_; ++k) param_vec.push_back(a_ref_sym_[k]);
    for (int k = 0; k < N_; ++k) param_vec.push_back(zmp_ref_sym_[k]);
    for (int k = 0; k <= N_; ++k) param_vec.push_back(com_ref_sym_[k]);

    for (int k = 0; k < N_; ++k) {
        for (int ee = 0; ee < n_ee_; ++ee) {
            param_vec.push_back(ee_pos_ref_sym_[k][ee]);
        }
    }
    for (int k = 0; k < N_; ++k) {
        for (int ee = 0; ee < n_ee_; ++ee) {
            param_vec.push_back(ee_ori_ref_sym_[k][ee]);
        }
    }

    // 5. Now vertcat
    ::casadi::SX p = ::casadi::SX::vertcat(param_vec);
    // Create NLP problem
    ::casadi::SXDict nlp;
    nlp["x"] = x_opt;      // symbolic accelerations
    nlp["p"] = p;          // Parameters (initial states (q0, v0) + reference trajectories)
    nlp["f"] = cost;       // Cost function
    nlp["g"] = constraints; // Constraints
    
    // Solver options
    ::casadi::Dict opts;
    opts["print_time"] = false;
    opts["verbose"] = false;
    opts["ipopt.print_level"] = 0;           // Silent output (0-12, where 0 is silent)
    opts["ipopt.max_iter"] = 500;            // Maximum iterations
    opts["ipopt.tol"] = 1e-6;                // Convergence tolerance
    opts["ipopt.acceptable_tol"] = 1e-4;     // Acceptable tolerance
    opts["ipopt.linear_solver"] = "ma27";    // Linear Solver (Default = ma27)
    
    // Create solver
    solver_ = ::casadi::nlpsol("mpc_solver", "ipopt", nlp, opts);
    
    std::cout << "Solver created successfully!" << std::endl;
    std::cout << "Optimization variables: " << x_opt.size1() << std::endl;
    std::cout << "Parameters: " << p.size1() << std::endl;
    std::cout << "Constraints: " << constraints.size1() << std::endl;
}

::casadi::DM MPCUtils::solve(
    const ::casadi::DM& q0,
    const ::casadi::DM& v0,
    const ReferenceTrajectories& refs
) {
    std::cout << "Solving nlp Problem..." << std::endl;
    // Prepare parameter vector
    std::vector<::casadi::DM> param_values;
    param_values.push_back(q0);
    param_values.push_back(v0);

    for (int k = 0; k <= N_; ++k) param_values.push_back(refs.q_ref[k]);
    for (int k = 0; k <= N_; ++k) param_values.push_back(refs.v_ref[k]);
    for (int k = 0; k < N_; ++k) param_values.push_back(refs.a_ref[k]);
    for (int k = 0; k < N_; ++k) param_values.push_back(refs.zmp_ref[k]);
    for (int k = 0; k <= N_; ++k) param_values.push_back(refs.com_ref[k]);

    for (int k = 0; k < N_; ++k)
        for (int ee = 0; ee < n_ee_; ++ee)
            param_values.push_back(refs.ee_pos_ref[k][ee]);
    for (int k = 0; k < N_; ++k)
        for (int ee = 0; ee < n_ee_; ++ee)
            param_values.push_back(refs.ee_ori_ref[k][ee]);

    ::casadi::DM p_val = ::casadi::DM::vertcat(param_values);
    
    int total_accel_vars = N_ * nv_;
    int total_force_vars = N_ * (6 * n_ee_);
    // Initial guess for BOTH accelerations and contact forces  - **need to enforce warm start
    ::casadi::DM x0 = ::casadi::DM::zeros(total_accel_vars + total_force_vars, 1);
    // Solve
    ::casadi::DMDict arg;
    arg["x0"] = x0;
    arg["p"] = p_val;
    arg["lbx"] = lbx_;
    arg["ubx"] = ubx_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    
    try {
        ::casadi::DMDict res = solver_(arg);
        ::casadi::DM solution = res.at("x");
        return solution;
        
    } catch (const std::exception& e) {
        std::cerr << "solve failed: " << e.what() << std::endl;
        return ::casadi::DM::zeros(N_ * nv_, 1);
    }
}
