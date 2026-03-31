struct OPTIONS_POST_TRAINING : OPTIONS_PRE_TRAINING{
    // Pre-training already has MOTOR_DELAY=true and action history via StateRotorsHistory.
    // No additional changes needed for payload post-training.
};

static_assert(sizeof(TI) == 8);
// training hyperparameters
constexpr TI NUM_EPISODES         = 128;
constexpr TI NUM_EPISODES_EVAL    = 100;
constexpr TI N_EPOCH              = 1000;
constexpr TI SEQUENCE_LENGTH      = 500;
constexpr TI BATCH_SIZE           = 64;
constexpr T  SOLVED_RETURN        = 1000; // pre-training achieves ~1300
constexpr TI HIDDEN_DIM           = 16;
constexpr TI NUM_TEACHERS         = 10;
constexpr TI NUM_ACTIVE_TEACHERS  = NUM_TEACHERS;
constexpr TI EPOCH_TEACHER_FORCING = 10;
constexpr bool DYNAMIC_ALLOCATION = true;
constexpr bool SHUFFLE            = true;
constexpr bool TEACHER_DETERMINISTIC = true;
constexpr bool ON_POLICY          = true;
constexpr TI TEACHER_STUDENT_MIX  = 0;
// position in payload env is trajectory-relative → no absolute-position offset correction needed
constexpr bool STEADY_STATE_POSITION_CORRECTION = false;
constexpr TI STEADY_STATE_POSITION_OFFSET_ESTIMATION_START = 0; // unused

// student environment (same type as pre-training: already has motor delay + action history)
using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY<DEVICE, T, TI, OPTIONS_POST_TRAINING>::ENVIRONMENT;
constexpr TI STEADY_STATE_POSITION_OFFSET_ESTIMATION_END = ENVIRONMENT::EPISODE_STEP_LIMIT;

#ifdef RL_TOOLS_POST_TRAINING
// Teacher environment: same type as student environment for payload.
// The pre-training SAC actor observed OBSERVATION_TYPE (no privileged payload state),
// which is identical to what the GRU student observes → no observation-type override needed.
struct ENVIRONMENT_TEACHER_STATIC_PARAMETERS : ENVIRONMENT::SPEC::STATIC_PARAMETERS{
    static constexpr auto PARAMETER_VALUES = [](){
        auto params = ENVIRONMENT::SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
        // ensure teacher operates without observation noise
        params.mdp.observation_noise.position         = 0;
        params.mdp.observation_noise.orientation      = 0;
        params.mdp.observation_noise.linear_velocity  = 0;
        params.mdp.observation_noise.angular_velocity = 0;
        params.mdp.observation_noise.imu_acceleration = 0;
        return params;
    }();
};
using ENVIRONMENT_TEACHER_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_TEACHER_STATIC_PARAMETERS>;
using ENVIRONMENT_TEACHER = rl_tools::rl::environments::Multirotor<ENVIRONMENT_TEACHER_SPEC>;
#endif

// GRU student network: Dense(relu) -> GRU -> Dense(identity)
template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;

using INPUT_LAYER_CONFIG  = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM,
    rlt::nn::activation_functions::ActivationFunction::RELU,
    rlt::nn::layers::dense::DefaultInitializer<T, TI>,
    rlt::nn::parameters::groups::Input>;
using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;

using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM,
    rlt::nn::parameters::groups::Normal>;
using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;

using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, ENVIRONMENT::ACTION_DIM,
    rlt::nn::activation_functions::ActivationFunction::IDENTITY,
    rlt::nn::layers::dense::DefaultInitializer<T, TI>,
    rlt::nn::parameters::groups::Output>;
using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;

using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;

using CAPABILITY  = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
using ACTOR       = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;

struct ADAM_PARAMETERS : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA              = 0.0001;
    static constexpr T WEIGHT_DECAY       = 0;
    static constexpr T WEIGHT_DECAY_INPUT  = 0;
    static constexpr T WEIGHT_DECAY_OUTPUT = 0;
};
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;

using OUTPUT_SHAPE = ACTOR::OUTPUT_SHAPE;

using RESULT      = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES,      ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using RESULT_EVAL = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using DATA        = rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<RESULT::SPEC>>;
using DATA_EVAL   = rlt::rl::utils::evaluation::NoData<rlt::rl::utils::evaluation::DataSpecification<RESULT_EVAL::SPEC>>;
