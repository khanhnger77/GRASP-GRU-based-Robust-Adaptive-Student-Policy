if(RL_TOOLS_ENABLE_HDF5)
    add_executable(payload_policy_pre_training_sample_dynamics_parameters pre_training/sample_dynamics_parameters.cpp)
    target_link_libraries(payload_policy_pre_training_sample_dynamics_parameters PRIVATE RLtools::RLtools)

    add_executable(payload_policy_pre_training pre_training/main.cpp)
    target_link_libraries(payload_policy_pre_training PRIVATE RLtools::RLtools)
endif()
