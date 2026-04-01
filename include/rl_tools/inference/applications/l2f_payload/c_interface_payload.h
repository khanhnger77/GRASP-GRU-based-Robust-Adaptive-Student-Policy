#ifndef RL_TOOLS_INFERENCE_APPLICATIONS_L2F_PAYLOAD_C_INTERFACE_H
#define RL_TOOLS_INFERENCE_APPLICATIONS_L2F_PAYLOAD_C_INTERFACE_H

#include <stdint.h>
#include "../../executor/c_interface.h"

#define RL_TOOLS_INTERFACE_APPLICATIONS_L2F_PAYLOAD_ACTION_DIM 4

#ifdef __cplusplus
extern "C" {
#endif
    // Observation struct for the payload GRU student policy.
    // All quantities are in the FLU (Forward-Left-Up) world frame unless noted.
    // position       : drone_pos_flu - trajectory_setpoint_flu  (world frame error)
    // orientation    : quaternion [w, x, y, z] of drone in FLU world frame
    // linear_velocity: drone_vel_flu - trajectory_vel_flu       (world frame error)
    // angular_velocity: body angular velocity in FLU body frame
    // previous_action: last motor command in [-1, 1] (relative to throttle range)
    typedef struct {
        float position[3];
        float orientation[4];
        float linear_velocity[3];
        float angular_velocity[3];
        float previous_action[RL_TOOLS_INTERFACE_APPLICATIONS_L2F_PAYLOAD_ACTION_DIM];
    } RLtoolsInferenceApplicationsL2FPayloadObservation;

    typedef struct {
        float action[RL_TOOLS_INTERFACE_APPLICATIONS_L2F_PAYLOAD_ACTION_DIM];
    } RLtoolsInferenceApplicationsL2FPayloadAction;

    void rl_tools_inference_applications_l2f_payload_init();
    void rl_tools_inference_applications_l2f_payload_reset();
    float rl_tools_inference_applications_l2f_payload_test(RLtoolsInferenceApplicationsL2FPayloadAction* action);
    RLtoolsInferenceExecutorStatus rl_tools_inference_applications_l2f_payload_control(
        uint64_t nanoseconds,
        RLtoolsInferenceApplicationsL2FPayloadObservation* observation,
        RLtoolsInferenceApplicationsL2FPayloadAction* action);
    const char* rl_tools_inference_applications_l2f_payload_checkpoint_name();

#ifdef __cplusplus
}
#endif

#endif
