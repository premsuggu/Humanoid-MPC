#ifndef MPC_UTILS_HPP
#define MPC_UTILS_HPP

#include <casadi/casadi.hpp>
#include <vector>
#include <string>
#include <map>
#include "robot_model.hpp"


class MPCUtils {
public:
    struct ReferenceTrajectories {
        std::vector<::casadi::DM> q_ref;                        // Position reference [N+1]
        std::vector<::casadi::DM> v_ref;                        // Velocity reference [N+1]
        std::vector<::casadi::DM> a_ref;                        // Acceleration reference [N]
        std::vector<std::vector<::casadi::DM>> ee_pos_ref;      // [N+1][n_ee]
        std::vector<std::vector<::casadi::DM>> ee_ori_ref;      // [N+1][n_ee]
        std::vector<::casadi::DM> zmp_ref;                      // ZMP reference [N]
        std::vector<::casadi::DM> com_ref;                      // CoM reference [N+1]
    };

    using ContactSchedule = std::vector<std::vector<bool>>;

    MPCUtils(int N, int nq, int nv, int n_ee, double dt,
             const std::vector<std::string>& ee_names,
             RobotModel* robot_model);

    void createSymbolicVariables();
    void setContactSchedule(const ContactSchedule& schedule);
    void setBounds(const ::casadi::DM& lbx, const ::casadi::DM& ubx,
                   const ::casadi::DM& lbg, const ::casadi::DM& ubg);

    ::casadi::SX buildCost(
        const ReferenceTrajectories& refs,
        const ::casadi::SX& Qq,          // Position cost
        const ::casadi::SX& Qv,          // Velocity cost  
        const ::casadi::SX& Ra,          // Acceleration cost
        const ::casadi::SX& Qf,          // Terminal cost
        const ::casadi::SX& Wee,         // End-effector cost
        const ::casadi::SX& Wzmp,        // ZMP cost
        const ::casadi::SX& W_tau
    ) const;

    ::casadi::SX buildConstraints(
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
    ) const;


    void setupSolver(const ::casadi::SX& cost, const ::casadi::SX& constraints, 
                     const ReferenceTrajectories& refs);

    ::casadi::DM solve(
        const ::casadi::DM& q0,
        const ::casadi::DM& v0,
        const ReferenceTrajectories& refs
    );


    // Symbolic helper functions (based on acceleration optimization)
    ::casadi::SX ee_pos_sym(int k, const std::string& ee_name) const;
    ::casadi::SX ee_ori_sym(int k, const std::string& ee_name) const;
    ::casadi::SX com_position_sym(int k) const;
    ::casadi::SX zmp_sym(int k) const;
    ::casadi::SX torque_sym(int k) const;
    

    // Accessors
    const std::vector<::casadi::SX>& q_sym() const { return q_sym_; }
    const std::vector<::casadi::SX>& v_sym() const { return v_sym_; }
    const std::vector<::casadi::SX>& a_sym() const { return a_sym_; }  // Main optimization variable
    int N() const { return N_; }
    int nq() const { return nq_; }
    int nv() const { return nv_; }
    int n_ee() const { return n_ee_; }
    double dt() const { return dt_; }

private:
    int N_, nq_, nv_, n_ee_;
    double dt_;
    std::vector<std::string> ee_names_;
    RobotModel* robot_model_;

    // Symbolic variables
    std::vector<::casadi::SX> q_sym_;    // States (dependent variables)
    std::vector<::casadi::SX> v_sym_;    // States (dependent variables)  
    std::vector<::casadi::SX> a_sym_;    // MAIN OPTIMIZATION VARIABLES
    std::vector<::casadi::SX> f_sym_;    // MAIN OPTIMIZATION VARIABLES
    
    ContactSchedule contact_schedule_;
    ::casadi::DM lbx_, ubx_, lbg_, ubg_;
    ::casadi::Function solver_;

    // helper functions (casadi::Function)
    ::casadi::Function casadi_integrate_;
    ::casadi::Function mass_matrix_func_;
    ::casadi::Function coriolis_func_;
    ::casadi::Function gravity_func_;
    std::map<std::string, ::casadi::Function> contact_jacobian_funcs_;
    void createDynamicsFunctions();

    
    // Helper functions for dynamics computation
    void IntegrationFunction();

    ::casadi::SX computeMassMatrix(const ::casadi::SX& q) const;
    ::casadi::SX computeCoriolisMatrix(const ::casadi::SX& q, const ::casadi::SX& v) const;
    ::casadi::SX computeGravityVector(const ::casadi::SX& q) const;
    ::casadi::SX computeContactJacobian(int k, const std::string& ee_name) const;

};

#endif // MPC_UTILS_HPP
