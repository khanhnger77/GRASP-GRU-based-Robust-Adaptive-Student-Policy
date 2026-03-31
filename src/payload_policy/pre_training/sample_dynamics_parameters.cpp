#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <fstream>
#include <filesystem>
namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = double;

using namespace rlt::rl::environments::l2f;

// --- Build the parameter chain ---
using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
struct DOMAIN_RANDOMIZATION_OPTIONS{
    static constexpr bool ENABLED = true;
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

using BASE_ENV_TYPE = rlt::rl::environments::Multirotor<Specification<T, TI>>;
static constexpr auto BASE_PARAMS = BASE_ENV_TYPE::SPEC::PARAMETER_VALUES;
static constexpr PARAMETERS_TYPE NOMINAL_PARAMETERS = {
    {   // ParametersTrajectory
        {   // ParametersDomainRandomization
            {   // ParametersDisturbances
                {   // ParametersBase
                    BASE_PARAMS.dynamics,
                    BASE_PARAMS.integration,
                    BASE_PARAMS.mdp
                },
                BASE_PARAMS.disturbances
            },
            BASE_PARAMS.domain_randomization
        },
        BASE_PARAMS.trajectory
    },
    {(T)0.5, (T)0.5, (TI)250}  // rope_length, payload_mass, drop_step
};

struct STATIC_PARAMETERS{
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 1000;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI ANGULAR_VELOCITY_DELAY = 0;
    using STATE_TYPE = StateBase<StateSpecification<T, TI>>;
    using OBSERVATION_TYPE = observation::LastComponent<TI>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    static constexpr PARAMETERS_TYPE PARAMETER_VALUES = NOMINAL_PARAMETERS;
    static constexpr TI N_DYNAMICS_VALUES = 1;
    static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
        parameters::dynamics::registry<parameters::dynamics::REGISTRY::x500_real, PARAMETERS_SPEC>
    };
    static constexpr T STATE_LIMIT_POSITION = 100000;
    static constexpr T STATE_LIMIT_VELOCITY = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

using ENVIRONMENT = rlt::rl::environments::Multirotor<Specification<T, TI, STATIC_PARAMETERS>>;

int main(int argc, char** argv){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 0;
    TI N = 10;
    rlt::init(device, rng, seed);
    ENVIRONMENT env;
    ENVIRONMENT::Parameters params;
    rlt::init(device, env);

    auto overwrite = [](auto& parameters) -> void {
        parameters.mdp.init = parameters::init::init_60_deg<PARAMETERS_SPEC>;
        parameters.mdp.reward = {
            false,   // non-negative
            01.00,   // scale
            01.50,   // constant
            -100.00, // termination penalty
            01.00,   // position
            00.00,   // position_clip
            00.10,   // orientation
            00.00,   // linear_velocity
            00.00,   // angular_velocity
            00.00,   // linear_acceleration
            00.00,   // angular_acceleration
            00.00,   // action
            01.00,   // d_action
            00.00,   // position_error_integral
        };
        parameters.domain_randomization = {
            2.6, 3.9,    // thrust_to_weight min/max  (x500 ~3.25)
            17.0, 32.0,  // torque_to_inertia min/max (x500 ~23)
            1.60, 2.40,  // mass min/max              (x500 2.0kg ±20%)
            0.06,        // mass_size_deviation
            0.040, 0.068, // motor_time_constant_rising  min/max (0.054 ±~25%)
            0.040, 0.068, // motor_time_constant_falling min/max
            0.081, 0.135, // rotor_torque_constant min/max (0.108 ±~25%)
            0.12,         // orientation_offset_angle_max (~7 deg)
            0.16          // disturbance_force_max
        };
        parameters.payload.rope_length  = 0.5;
        parameters.payload.payload_mass = 1.0;
        parameters.payload.drop_step    = 250; // 2.5s at 100Hz
    };

    // Set x500_real as base dynamics in env so sample_initial_parameters starts from it
    env.parameters.dynamics = parameters::dynamics::x500::real<T, TI>;
    overwrite(env.parameters);

    std::filesystem::path output_path = "./src/payload_policy/dynamics_parameters/";
    if (!std::filesystem::exists(output_path)){
        std::filesystem::create_directories(output_path);
    }
    for (TI set_i = 0; set_i < N; ++set_i){
        overwrite(params);
        rlt::sample_initial_parameters(device, env, params, rng); // randomizes dynamics around x500_real
        overwrite(params); // force fixed payload/reward/init config
        std::ofstream output(output_path / (std::to_string(set_i) + ".json"));
        auto params_copy = params;
        params_copy.domain_randomization = parameters::domain_randomization_disabled<T>;
        output << rlt::json(device, env, params_copy);
        output.close();
        std::cout << "Saved set " << set_i << ": mass=" << params.dynamics.mass << std::endl;
    }

    rlt::free(device, rng);
    return 0;
}
