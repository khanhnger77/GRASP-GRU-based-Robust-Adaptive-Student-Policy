#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

#include <rl_tools/utils/generic/typing.h>

namespace builder{
    using namespace rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY{

        using BASE_ENV = rl_tools::rl::environments::Multirotor<Specification<T, TI>>;

        static constexpr auto MODEL = parameters::dynamics::REGISTRY::x500_real;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
        struct DOMAIN_RANDOMIZATION_OPTIONS{
            static constexpr bool ENABLED = false;
            static constexpr bool THRUST_TO_WEIGHT = ENABLED;
            static constexpr bool MASS = ENABLED;
            static constexpr bool TORQUE_TO_INERTIA = ENABLED;
            static constexpr bool MASS_SIZE_DEVIATION = ENABLED;
            static constexpr bool ROTOR_TORQUE_CONSTANT = ENABLED;
            static constexpr bool DISTURBANCE_FORCE = ENABLED;
            static constexpr bool ROTOR_TIME_CONSTANT = ENABLED;
        };

        struct TRAJECTORY_OPTIONS{
            static constexpr bool LANGEVIN = true;
        };

        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
        using PARAMETERS_BASE_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, TRAJECTORY_OPTIONS,
            ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS,
            ParametersDisturbances<ParametersSpecification<T, TI,
            ParametersBase<PARAMETERS_SPEC>>>>>>>;
        using PARAMETERS_TYPE = ParametersPayload<ParametersPayloadSpecification<T, TI, PARAMETERS_BASE_TYPE>>;

        static constexpr TI SIMULATION_FREQUENCY = 100;

        static constexpr auto BASE_PARAMS = BASE_ENV::SPEC::PARAMETER_VALUES;

        static constexpr auto BASE_MDP = [](){
            auto mdp = BASE_PARAMS.mdp;
            mdp.init = parameters::init::init_60_deg<PARAMETERS_SPEC>;
            return mdp;
        }();

        // Drop happens at 2.5 seconds into the episode (250 steps at 100 Hz)
        static constexpr TI DROP_STEP = 250;

        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {   // ParametersTrajectory (NEXT_COMPONENT of ParametersPayload)
                {   // ParametersDomainRandomization
                    {   // ParametersDisturbances
                        {   // ParametersBase
                            BASE_PARAMS.dynamics,
                            BASE_PARAMS.integration,
                            BASE_MDP
                        },
                        BASE_PARAMS.disturbances
                    },
                    BASE_PARAMS.domain_randomization
                },
                BASE_PARAMS.trajectory
            },
            {   // payload
                (T)0.5,          // rope_length [m]
                (T)0.5,          // payload_mass [kg] — ~26% of drone, causes visible jolt at drop
                DROP_STEP        // drop_step
            }
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI EPISODE_STEP_LIMIT = 10 * SIMULATION_FREQUENCY; // 10s: 2.5s with payload + 7.5s hover
            static constexpr TI CLOSED_FORM = false;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0;
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                StateLastAction<StateSpecification<T, TI,
                StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE_MOTOR_DELAY = StatePayload<StateSpecification<T, TI,
                StateTrajectory<StateSpecification<T, TI,
                StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM,
                StateRandomForce<StateSpecification<T, TI,
                STATE_BASE>>>>>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StatePayload<StateSpecification<T, TI,
                StateRandomForce<StateSpecification<T, TI,
                STATE_BASE>>>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::TrajectoryTrackingPosition<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::TrajectoryTrackingLinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 1,
                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>>
            >>>>>>>>>;
            // Critic (privileged) sees payload state: theta, phi, dtheta, dphi, attached
            using OBSERVATION_TYPE_PRIVILEGED = observation::TrajectoryTrackingPosition<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::TrajectoryTrackingLinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 1,
                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI,
                    observation::Payload<observation::PayloadSpecification<T, TI>
                    >>>>>>>>>>>>>;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr TI N_DYNAMICS_VALUES = 1;
            static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<parameters::dynamics::REGISTRY::x500_real, PARAMETERS_SPEC>
            };
            static constexpr T STATE_LIMIT_POSITION = 100000;
            static constexpr T STATE_LIMIT_VELOCITY = 100000;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
