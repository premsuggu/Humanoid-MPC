#include <iostream>
#include <vector>
#include <cmath>
#include "robot_model.hpp"
#include "mpc_utils.hpp"

// Generate reference trajectories
MPCUtils::ReferenceTrajectories generateAccelerationBasedTrajectory(int N, double dt, int nq, int nv, int n_ee) {
    MPCUtils::ReferenceTrajectories refs;
    
    // Initialize trajectory vectors
    refs.q_ref.resize(N + 1);
    refs.v_ref.resize(N + 1);
    refs.a_ref.resize(N);
    refs.zmp_ref.resize(N);
    refs.com_ref.resize(N + 1);
    refs.ee_pos_ref.resize(N + 1, std::vector<casadi::DM>(n_ee));
    refs.ee_ori_ref.resize(N + 1, std::vector<casadi::DM>(n_ee));
    
    // Walking parameters
    double step_length = 0.3;
    double step_height = 0.05;
    double step_time = 0.8;
    double com_height = 0.98;
    
    for (int k = 0; k <= N; ++k) {
        double t = k * dt;
        
        // Position reference (standing pose with slight forward motion)
        refs.q_ref[k] = casadi::DM::zeros(nq, 1);
        refs.q_ref[k](2) = com_height;  // pelvis height
        refs.q_ref[k](0) = t * 0.05;    // slow forward motion
        
        // Velocity reference
        refs.v_ref[k] = casadi::DM::zeros(nv, 1);
        refs.v_ref[k](0) = 0.05;  // forward velocity
        
        // CoM reference
        refs.com_ref[k] = casadi::DM::zeros(3, 1);
        refs.com_ref[k](0) = t * 0.05;
        refs.com_ref[k](1) = 0.0;
        refs.com_ref[k](2) = com_height;
        
        // Acceleration reference (smooth)
        if (k < N) {
            refs.a_ref[k] = casadi::DM::zeros(nv, 1);
            
            // ZMP reference (centered)
            refs.zmp_ref[k] = casadi::DM::zeros(2, 1);
            refs.zmp_ref[k](0) = t * 0.05;  // follow CoM
            refs.zmp_ref[k](1) = 0.0;
        }
        
        // End-effector references
        for (int ee = 0; ee < n_ee; ++ee) {
            if (ee < 2) { // Feet
                double phase = fmod(t / step_time, 2.0);
                bool is_swing = (ee == 0 && phase > 1.0) || (ee == 1 && phase <= 1.0);
                
                refs.ee_pos_ref[k][ee] = casadi::DM::zeros(3, 1);
                refs.ee_pos_ref[k][ee](0) = (ee == 0 ? -0.1 : 0.1) + t * 0.05;
                refs.ee_pos_ref[k][ee](1) = (ee == 0 ? 0.15 : -0.15);
                refs.ee_pos_ref[k][ee](2) = is_swing ? step_height : 0.0;
                
            } else { // Hands
                refs.ee_pos_ref[k][ee] = casadi::DM::zeros(3, 1);
                refs.ee_pos_ref[k][ee](0) = 0.0;
                refs.ee_pos_ref[k][ee](1) = (ee == 2 ? 0.3 : -0.3);
                refs.ee_pos_ref[k][ee](2) = 1.2;
            }
            
            // Orientation reference (identity quaternion)
            refs.ee_ori_ref[k][ee] = casadi::DM({1.0, 0.0, 0.0, 0.0});
        }
    }
    
    return refs;
}

// Generate contact schedule
MPCUtils::ContactSchedule generateContactSchedule(int N, double dt) {
    MPCUtils::ContactSchedule schedule(N, std::vector<bool>(4, false));
    
    double step_time = 0.8;
    
    for (int k = 0; k < N; ++k) {
        double t = k * dt;
        double phase = fmod(t / step_time, 2.0);
        
        schedule[k][0] = (phase <= 1.0); // left foot
        schedule[k][1] = (phase > 1.0);  // right foot
        schedule[k][2] = false;          // left hand
        schedule[k][3] = false;          // right hand
    }
    
    return schedule;
}

int main() {
    try {
        std::cout << "=== H1 Humanoid Acceleration-Based MPC ===" << std::endl;
        
        // Robot setup
        std::string urdf_path = "/home/prem/wb_mpc/robots/h1_description/urdf/h1_with_hand.urdf";
        std::vector<std::string> ee_names = {
            "left_ankle_link",
            "right_ankle_link",
            "left_hand_link",
            "right_hand_link"
        };
        
        // Load robot model
        std::cout << "Loading robot model..." << std::endl;
        RobotModel robot(urdf_path, ee_names);
        const pinocchio::Model& model = robot.getModel();

        std::cout << "\nAvailable frames:" << std::endl;
        for(size_t i = 0; i < model.frames.size(); ++i) {
            std::cout << i << ": " << model.frames[i].name << std::endl;
        }

        
        int nq = robot.getNumJoints();
        int nv = model.nv;
        int n_ee = ee_names.size();
        
        std::cout << "Robot loaded successfully!" << std::endl;
        std::cout << "  DOF: " << nq << std::endl;
        std::cout << "  Velocity DOF: " << nv << std::endl;
        std::cout << "  End-effectors: " << n_ee << std::endl;
        
        // MPC parameters
        int N = 10;
        double dt = 0.05;
        int simulation_steps = 50;
        
        // Create MPC instance
        std::cout << "Initializing acceleration-based MPC..." << std::endl;
        MPCUtils mpc(N, nq, nv, n_ee, dt, ee_names, &robot);
        
        // Create symbolic variables and dynamics
        mpc.createSymbolicVariables();
        std::cout << "MPC initialized!" << std::endl;
        
        // Generate trajectories and contact schedule
        std::cout << "Generating reference trajectories..." << std::endl;
        MPCUtils::ReferenceTrajectories refs = generateAccelerationBasedTrajectory(N, dt, nq, nv, n_ee);
        
        MPCUtils::ContactSchedule contact_schedule = generateContactSchedule(N, dt);
        mpc.setContactSchedule(contact_schedule);
        
        // Weight matrices
        casadi::SX Qq = casadi::SX::eye(nq) * 1.0;      // Position cost
        casadi::SX Qv = casadi::SX::eye(nv) * 0.1;      // Velocity cost
        casadi::SX Ra = casadi::SX::eye(nv) * 0.01;     // Acceleration cost (main)
        casadi::SX Qf = casadi::SX::eye(nq) * 10.0;     // Terminal cost
        casadi::SX Wee = casadi::SX::eye(3) * 1.0;      // End-effector cost
        casadi::SX Wzmp = casadi::SX::eye(2) * 5.0;     // ZMP cost
        casadi::SX W_tau = casadi::SX::eye(nv) * 0.001; // Torque Regularization cost
        
        // Build optimization problem
        std::cout << "Building optimization problem..." << std::endl;
        casadi::SX cost = mpc.buildCost(refs, Qq, Qv, Ra, Qf, Wee, Wzmp, W_tau);
        
        std::vector<double> v_min(nv, -10.0);           // Velocity limits
        std::vector<double> v_max(nv, 10.0);
        std::vector<double> tau_min(nv, -100.0);        // Torque limits  
        std::vector<double> tau_max(nv, 100.0);
        std::vector<std::pair<std::string, std::string>> collision_pairs; // Empty for now

        casadi::SX constraints = mpc.buildConstraints(
            casadi::SX(-0.05), casadi::SX(0.05),  // ZMP x bounds
            casadi::SX(-0.05), casadi::SX(0.05),  // ZMP y bounds
            v_min, v_max,                         // Velocity limits
            tau_max, tau_min,                     // Torque limits
            0.05,                                 // min clearance
            0.7,                                  // friction coefficient
            collision_pairs                       // collision avoidance pairs
        );
        
        // Bounds for optimization variables
        int total_acc_vars = N * nv;  
        int total_force_vars = N * (6 * n_ee);  
        int total_vars = total_acc_vars + total_force_vars;

        casadi::DM lbx = casadi::DM::zeros(total_vars, 1);
        casadi::DM ubx = casadi::DM::zeros(total_vars, 1);

        // Set acceleration bounds
        for (int i = 0; i < total_acc_vars; ++i) {
            lbx(i) = -50.0;  
            ubx(i) = 50.0;
        }

        // Set contact force bounds (for ALL end-effectors)
        for (int i = total_acc_vars; i < total_vars; ++i) {
            lbx(i) = -1000.0;  
            ubx(i) = 1000.0;
        }


        casadi::DM lbg = casadi::DM::zeros(constraints.size1(), 1);
        casadi::DM ubg = casadi::DM::zeros(constraints.size1(), 1);
        
        for (int i = 0; i < constraints.size1(); ++i) {
            lbg(i) = 0.0;      // Inequality constraints >= 0
            ubg(i) = ::casadi::inf;      // Upper bound (infinity)
        }

        int total_eq_constraints = 0;
        for (int k = 0; k < N; ++k){
            for (int ee = 0; ee < n_ee; ++ee) {
                if (k < contact_schedule.size() && !contact_schedule[k][ee]) {
                    // Non-contact end-effectors must have zero force
                    total_eq_constraints += 6;  // 6 components of force
                }
            }
        }
        int eq_constraint_start_idx = constraints.size1() - total_eq_constraints;

        for (int i = eq_constraint_start_idx; i < constraints.size1(); ++i) {
            lbg(i) =0;
            ubg(i) = 0; 
        }
        
        mpc.setBounds(lbx, ubx, lbg, ubg);
        
        // Setup solver
        std::cout << "Setting up solver..." << std::endl;
        mpc.setupSolver(cost, constraints, refs);
        std::cout << "Solver ready!" << std::endl;
        
        // Initial state
        casadi::DM q0 = casadi::DM::ones(nq, 1);
        q0(2) = 0.98; 
        casadi::DM v0 = casadi::DM::zeros(nv, 1);
        
        // MPC simulation loop
        std::cout << "\n=== Starting Acceleration-Based MPC Simulation ===" << std::endl;
        
        for (int step = 0; step < simulation_steps; ++step) {
            double current_time = step * dt;
            
            std::cout << "Step " << step << " (t=" << current_time << "s)" << std::endl;
            
            try {
                // Solve MPC
                casadi::DM solution = mpc.solve(q0, v0, refs);
                
                // Extract first acceleration
                casadi::DM a_optimal = solution(casadi::Slice(0, nv));

                // contact force of first time step
                casadi::DM f_optimal = solution(casadi::Slice(N * nv, N * nv + 6 * n_ee));
                
                // Integrate to get next state 
                casadi::DM v_next = v0 + dt * a_optimal;

                Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(
                    static_cast<const double*>(q0.ptr()), nq);
                Eigen::VectorXd v_dt_eigen = Eigen::Map<const Eigen::VectorXd>(
                    static_cast<const double*>((v_next * dt).ptr()), nv);
                
                Eigen::VectorXd q_next_eigen(nq);
                pinocchio::integrate(robot.getModel(), q_eigen, v_dt_eigen, q_next_eigen);
                
                casadi::DM q_next = casadi::DM::zeros(nq, 1);
                for (int i = 0; i < nq; ++i) {
                    q_next(i) = q_next_eigen[i];
                }
                
                // Update state
                q0 = q_next;
                v0 = v_next;
                
                // Print progress
                if (step % 5 == 0) {
                    std::cout << "  CoM height: " << double(q0(2)) << " m" << std::endl;
                    std::cout << "  Forward position: " << double(q0(0)) << " m" << std::endl;
                    std::cout << "  Forward velocity: " << double(v0(0)) << " m/s" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "MPC solve failed at step " << step << ": " << e.what() << std::endl;
                break;
            }
        }
        
        std::cout << "\n=== Acceleration-Based MPC Simulation Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
