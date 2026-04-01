#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_INFERENCE_APPLICATIONS_L2F_PAYLOAD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_INFERENCE_APPLICATIONS_L2F_PAYLOAD_H

#include "../../executor/executor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::inference::applications{
    namespace l2f_payload{
        // Specification for the payload GRU student policy inference executor.
        // INPUT_DIM = 3(pos_err) + 9(R) + 3(vel_err) + 3(omega) + OUTPUT_DIM*ACTION_HISTORY
        //           = 18 + OUTPUT_DIM*ACTION_HISTORY_LENGTH
        template <typename T_T, typename T_TI, typename T_TIMESTAMP,
                  T_TI T_ACTION_HISTORY_LENGTH, T_TI T_OUTPUT_DIM,
                  typename T_POLICY,
                  T_TIMESTAMP T_CONTROL_INTERVAL_INTERMEDIATE_NS,
                  T_TIMESTAMP T_CONTROL_INTERVAL_NATIVE_NS,
                  bool T_FORCE_SYNC_INTERMEDIATE = false,
                  T_TI T_FORCE_SYNC_NATIVE = 0,
                  bool T_DYNAMIC_ALLOCATION = true>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using TIMESTAMP = T_TIMESTAMP;
            using POLICY = T_POLICY;
            static constexpr T_TI ACTION_HISTORY_LENGTH = T_ACTION_HISTORY_LENGTH;
            static constexpr T_TI OUTPUT_DIM = T_OUTPUT_DIM;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
            using EXECUTOR_SPEC = executor::Specification<T, TI, TIMESTAMP, POLICY,
                T_CONTROL_INTERVAL_INTERMEDIATE_NS, T_CONTROL_INTERVAL_NATIVE_NS,
                T_FORCE_SYNC_INTERMEDIATE, T_FORCE_SYNC_NATIVE, T_DYNAMIC_ALLOCATION>;
        };

        template <typename SPEC>
        struct Observation{
            using T = typename SPEC::T;
            T position[3];
            T orientation[4]; // Quaternion [w, x, y, z]
            T linear_velocity[3];
            T angular_velocity[3];
            T previous_action[SPEC::OUTPUT_DIM];
        };

        template <typename SPEC>
        struct Action{
            float action[SPEC::OUTPUT_DIM];
        };
    }

    template <typename SPEC>
    struct L2FPayload{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using TIMESTAMP = typename SPEC::TIMESTAMP;

        T action_history[SPEC::ACTION_HISTORY_LENGTH][SPEC::OUTPUT_DIM];

        // INPUT_DIM = pos(3) + R(9) + vel(3) + omega(3) + action_history
        static constexpr TI INPUT_DIM = 18 + SPEC::OUTPUT_DIM * SPEC::ACTION_HISTORY_LENGTH;

        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, INPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, SPEC::OUTPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> output;
        Executor<typename SPEC::EXECUTOR_SPEC> executor;
        TI steps_since_original_control_step;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
