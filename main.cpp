#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "robot_model.hpp"
#include "mpc_utils.hpp"

std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

void saveTrajectoryToCSV(const std::vector<std::vector<double>>& trajectory, 
                        const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : trajectory) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    std::vector<std::vector<double>> q_trajectory;
    try {        
        // Robot setup
        std::string urdf_path = "/home/prem/wb_mpc/robots/h1_description/urdf/h1.urdf";
        std::vector<std::string> ee_names = {
            "left_ankle_link",
            "right_ankle_link",
        };
        
        // Load robot model
        std::cout << "Loading robot model..." << std::endl;
        RobotModel robot(urdf_path, ee_names);
        const pinocchio::Model& model = robot.getModel();

        
        int nq = robot.getNumJoints();
        int nv = model.nv;
        int n_ee = ee_names.size();
        
        std::cout << "Robot loaded successfully!" << std::endl;
        std::cout << "  DOF: " << model.nq << std::endl;
        std::cout << "  Velocity DOF: " << model.nv << std::endl;
        std::cout << "  End-effectors: " << n_ee << std::endl;
        
        // MPC parameters
        int N = 20;
        double dt = 0.05;
        int simulation_steps = 119;
        
        // Create MPC instance
        MPCUtils mpc(N, nq, nv, n_ee, dt, ee_names, &robot);
        
        // Create symbolic variables and dynamics
        mpc.createSymbolicVariables();
        std::cout << "MPC initialized!" << std::endl;
    // -------------------------------------Load references----------------------------------------
        std::string walk_dir = "/home/prem/wb_mpc/walk/";

        // Load matrices from CSV
        auto q_ref_mat = readCSV(walk_dir + "q_ref.csv");
        auto v_ref_mat = readCSV(walk_dir + "v_ref.csv");
        auto a_ref_mat = readCSV(walk_dir + "a_ref.csv");
        auto com_ref_mat = readCSV(walk_dir + "com_ref.csv");
        auto zmp_ref_mat = readCSV(walk_dir + "zmp_ref.csv");
        auto ee_pos_ref_mat = readCSV(walk_dir + "ee_pos_ref.csv");
        auto ee_ori_ref_mat = readCSV(walk_dir + "ee_ori_ref.csv");
        auto contact_schedule_mat = readCSV(walk_dir + "contact_schedule.csv");

        // Fill ReferenceTrajectories struct
        MPCUtils::ReferenceTrajectories refs;
        for (const auto& row : q_ref_mat){
            refs.q_ref.push_back(casadi::DM(row));
        }            
        for (const auto& row : v_ref_mat){
            refs.v_ref.push_back(casadi::DM(row));
        }
        for (const auto& row : a_ref_mat){
            refs.a_ref.push_back(casadi::DM(row));
        }
        for (const auto& row : com_ref_mat){
            refs.com_ref.push_back(casadi::DM(row));
        }
        for (const auto& row : zmp_ref_mat){
            refs.zmp_ref.push_back(casadi::DM(row));
        }

        // End-effector positions: each row is [ee1_x, ee1_y, ee1_z, ee2_x, ee2_y, ee2_z, ...]
        for (const auto& row : ee_pos_ref_mat) {
            std::vector<casadi::DM> ee_row;
            for (int ee = 0; ee < n_ee; ++ee)
                ee_row.push_back(casadi::DM(std::vector<double>(row.begin() + 3*ee, row.begin() + 3*(ee+1))));
            refs.ee_pos_ref.push_back(ee_row);
        }

        // End-effector orientations: each row is [ee1_w, ee1_x, ee1_y, ee1_z, ee2_w, ...]
        for (const auto& row : ee_ori_ref_mat) {
            std::vector<casadi::DM> ee_row;
            for (int ee = 0; ee < n_ee; ++ee)
                ee_row.push_back(casadi::DM(std::vector<double>(row.begin() + 4*ee, row.begin() + 4*(ee+1))));
            refs.ee_ori_ref.push_back(ee_row);
        }

        // Contact schedule: each row is a time step, columns are 0/1 for each foot
        MPCUtils::ContactSchedule contact_schedule;
        for (const auto& row : contact_schedule_mat) {
            std::vector<bool> cs_row;
            for (double val : row)
                cs_row.push_back(val > 0.5);
            contact_schedule.push_back(cs_row);
        }
    // -----------------------------------------------------------------------------
        // Weight matrices
        casadi::SX Qq = casadi::SX::eye(nq) * 5.0;      // Position cost
        casadi::SX Qv = casadi::SX::eye(nv) * 0.1;      // Velocity cost
        casadi::SX Ra = casadi::SX::eye(nv) * 0.01;     // Acceleration cost (main)
        casadi::SX Qf = casadi::SX::eye(nq) * 20.0;     // Terminal cost
        casadi::SX Wee = casadi::SX::eye(3) * 1.0;      // End-effector position & orientation (quaternion) cost
        casadi::SX Wzmp = casadi::SX::eye(2) * 5.0;     // ZMP cost
        casadi::SX W_tau = casadi::SX::eye(nv) * 0.001; // Torque Regularization cost
        casadi::SX Wcom = casadi::SX::eye(3) * 5.0;     // CoM tracking cost

        // Build optimization problem
        std::cout << "Building optimization problem..." << std::endl;
        casadi::SX cost = mpc.buildCost(Qq, Qv, Ra, Qf, Wee, Wzmp, Wcom, W_tau);
        
        std::vector<double> v_min(nv, -2.0);           // Velocity limits
        std::vector<double> v_max(nv, 2.0);
        std::vector<double> tau_min(nv, -150.0);        // Torque limits  
        std::vector<double> tau_max(nv, 150.0);
        std::vector<std::pair<std::string, std::string>> collision_pairs; // Empty for now

        casadi::SX constraints = mpc.buildConstraints(
            casadi::SX(-0.12), casadi::SX(0.12),  // ZMP x bounds
            casadi::SX(-0.12), casadi::SX(0.12),  // ZMP y bounds
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
            lbx(i) = -10.0;  
            ubx(i) = 10.0;
        }

        // Set contact force bounds
        for (int i = total_acc_vars; i < total_vars; ++i) {
            lbx(i) = -1000.0;  
            ubx(i) = 1000.0;
        }


        casadi::DM lbg = casadi::DM::zeros(constraints.size1(), 1);
        casadi::DM ubg = casadi::DM::zeros(constraints.size1(), 1);
        
        for (int i = 0; i < constraints.size1(); ++i) {
            lbg(i) = 0.0;                      // Inequality constraints >= 0
            ubg(i) = ::casadi::inf;            // Upper bound (infinity)
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
            lbg(i) = 0;
            ubg(i) = 0; 
        }
        
        mpc.setBounds(lbx, ubx, lbg, ubg);
        
        // Setup solver
        std::cout << "Setting up solver..." << std::endl;
        mpc.setupSolver(cost, constraints, refs);
        std::cout << "Solver ready!" << std::endl;
        
        // pinocchio's Neutral config
        Eigen::VectorXd q_neutral = pinocchio::neutral(robot.getModel());

        // Initial state
        casadi::DM q0 = casadi::DM::zeros(nq, 1);
        for (int i = 0; i < nq; ++i) {
            q0(i) = q_neutral[i];
        }
        q0(2) = 0.98;
        casadi::DM v0 = casadi::DM::zeros(nv, 1);
        // MPC simulation loop        
        for (int step = 0; step < simulation_steps; ++step) {
            double current_time = step * dt;
            std::cout << "Step " << step << " (t =" << current_time << "s)" << std::endl;
            MPCUtils::ReferenceTrajectories ref_window;
            // For N+1 arrays (state refs)
            for (int k = 0; k <= N; ++k) {
                int idx = std::min(step + k, int(refs.q_ref.size()) - 1);
                ref_window.q_ref.push_back(refs.q_ref[idx]);
                ref_window.v_ref.push_back(refs.v_ref[idx]);
                ref_window.com_ref.push_back(refs.com_ref[idx]);
            }

            // For N arrays (control refs)
            for (int k = 0; k < N; ++k) {
                int idx = std::min(step + k, int(refs.a_ref.size()) - 1);
                ref_window.a_ref.push_back(refs.a_ref[idx]);
                ref_window.zmp_ref.push_back(refs.zmp_ref[idx]);
                ref_window.ee_pos_ref.push_back(refs.ee_pos_ref[idx]);
                ref_window.ee_ori_ref.push_back(refs.ee_ori_ref[idx]);
            }

            // Contact schedule
            MPCUtils::ContactSchedule contact_window;
            for (int k = 0; k < N; ++k) {
                int idx = std::min(step + k, int(contact_schedule.size()) - 1);
                contact_window.push_back(contact_schedule[idx]);
            }

            mpc.setContactSchedule(contact_window);
            try {
                // Solve MPC
                casadi::DM solution = mpc.solve(q0, v0, ref_window);
                
                // Extract first acceleration
                casadi::DM a_optimal = solution(casadi::Slice(0, nv));

                // contact force of first time step
                casadi::DM f_optimal = solution(casadi::Slice(N * nv, N * nv + 6 * n_ee));
                
                // Integrate to get next state 
                casadi::DM v_next = v0 + dt * a_optimal;

                Eigen::VectorXd q_eigen = Eigen::Map<const Eigen::VectorXd>(
                    static_cast<const double*>(q0.ptr()), nq);
                Eigen::VectorXd v_next_eigen = Eigen::Map<const Eigen::VectorXd>(
                    static_cast<const double*>(v_next.ptr()), nv);  // Just velocity

                Eigen::VectorXd q_next_eigen(nq);
                pinocchio::integrate(robot.getModel(), q_eigen, v_next_eigen * dt, q_next_eigen);
                
                casadi::DM q_next = casadi::DM::zeros(nq, 1);
                for (int i = 0; i < nq; ++i) {
                    q_next(i) = q_next_eigen[i];
                }

                std::vector<double> q_step;
                for (int i = 0; i < nq; ++i) {
                    q_step.push_back(double(q_next(i)));
                }
                q_trajectory.push_back(q_step);
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
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Saving trajectories..." << std::endl;
    saveTrajectoryToCSV(q_trajectory, "q_optimal.csv");

    return 0;
}
