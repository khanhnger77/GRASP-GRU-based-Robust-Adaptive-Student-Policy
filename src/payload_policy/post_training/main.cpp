#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include "environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "../pre_training/config.h"
#include "../pre_training/options.h"
namespace rlt = rl_tools;

#include "helper.h"

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

#define RL_TOOLS_POST_TRAINING
#include "config.h"

// Derive teacher (pre-trained SAC actor) type from pre-training loop config
constexpr TI DATASET_SIZE = (ON_POLICY ? 1 : N_EPOCH) * (1 + TEACHER_STUDENT_MIX) * NUM_TEACHERS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT;

using FACTORY          = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG      = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
using ACTOR_ORIG       = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
using ACTOR_TEACHER    = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

struct CHECKPOINT_PARAMETERS{
    static constexpr TI TEST_INPUT_BATCH_SIZE = 2;
};

// note: make sure that the rng is advanced in the exact same way as in pre-training
// so that the dynamics parameters used to generate data match the ones seen by each teacher

int main(int argc, char** argv){
    // declarations
    DEVICE device;
    RNG rng;
    ACTOR_TEACHER actor_teacher[NUM_TEACHERS];
    ENVIRONMENT_TEACHER::Parameters teacher_parameters[NUM_TEACHERS];
    typename ACTOR_TEACHER::Buffer<> actor_teacher_buffer;
    ACTOR actor, best_actor;
    ACTOR::Buffer<> actor_buffer;
    OPTIMIZER actor_optimizer;
    std::cout << "Input shape: " << std::endl;
    rlt::print(device, ACTOR::INPUT_SHAPE{});
    std::cout << "Dataset size: " << DATASET_SIZE << std::endl;
    rlt::Tensor<rlt::tensor::Specification<TeacherMeta<T>, TI, rlt::tensor::Shape<TI, NUM_TEACHERS>>>                       teacher_metas;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>>                                   dataset_episode_start_indices;
    rlt::Tensor<rlt::tensor::Specification<T,  TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::Observation::DIM>>>    dataset_input;
    rlt::Tensor<rlt::tensor::Specification<T,  TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::ACTION_DIM>>>          dataset_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>>                                 dataset_truncated;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>>                                 dataset_reset;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>>                                   epoch_indices;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> batch_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>>       batch_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>>>               batch_reset;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>>                                                            d_output;

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        rlt::malloc(device, actor_teacher[teacher_i]);
    }
    rlt::malloc(device, actor_teacher_buffer);
    rlt::malloc(device, actor);
    rlt::malloc(device, best_actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, teacher_metas);
    rlt::malloc(device, dataset_episode_start_indices);
    rlt::malloc(device, dataset_input);
    rlt::malloc(device, dataset_output_target);
    rlt::malloc(device, dataset_truncated);
    rlt::malloc(device, dataset_reset);
    rlt::malloc(device, epoch_indices);
    rlt::malloc(device, batch_input);
    rlt::malloc(device, batch_output_target);
    rlt::malloc(device, batch_reset);
    rlt::malloc(device, d_output);

    // init
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;
    TI current_episode = 0;
    TI current_index   = 0;
    T  best_return     = 0;
    bool best_return_set = false;

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    std::filesystem::path run_path = "logs/" + timestamp_string;
    rlt::init(device, device.logger, run_path.string());
#else
    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    std::filesystem::path run_path = "logs/" + timestamp_string;
#endif
    rlt::init(device, rng, seed);
    rlt::init_weights(device, actor, rng);

    // --- load pre-trained teacher policies ---
    const std::string pre_training_experiment  = "2026-03-30_04-32-13";
    const std::string pre_training_name        = "payload-policy-pre-training";
    const std::string dynamics_parameters_path = "./src/payload_policy/dynamics_parameters/";

    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        rlt::utils::extrack::Path cpp_copy;
        cpp_copy.experiment                = pre_training_experiment;
        cpp_copy.name                      = pre_training_name;
        cpp_copy.attributes["dynamics-id"] = std::to_string(teacher_i);
        cpp_copy.step                      = "000000001000000";
        rlt::find_latest_run(device, "experiments", cpp_copy);

        auto actor_file = HighFive::File(cpp_copy.checkpoint_path.string(), HighFive::File::ReadOnly);
        rlt::load(device, actor_teacher[teacher_i], actor_file.getGroup("actor"));

        // load matching dynamics parameters
        std::ifstream dyn_file(dynamics_parameters_path + std::to_string(teacher_i) + ".json");
        if (!dyn_file){
            std::cerr << "Failed to open dynamics parameter file for teacher " << teacher_i << std::endl;
            return 1;
        }
        std::string dyn_json((std::istreambuf_iterator<char>(dyn_file)), std::istreambuf_iterator<char>());
        dyn_file.close();

        ENVIRONMENT_TEACHER env;
        rlt::from_json(device, env, dyn_json, teacher_parameters[teacher_i]);

        // verify teacher performance
        rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT_TEACHER, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>> result;
        rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<decltype(result)::SPEC>> data;
        RNG rng_copy = rng;
        rlt::malloc(device, data);
        sample_trajectories<ENVIRONMENT_TEACHER>(device, actor_teacher[teacher_i], teacher_parameters[teacher_i], result, data, rng_copy);

        // payload uses trajectory-relative position → no steady-state position correction
        auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
        teacher_meta.steady_state_position_offset[0] = 0;
        teacher_meta.steady_state_position_offset[1] = 0;
        teacher_meta.steady_state_position_offset[2] = 0;
        rlt::set(device, teacher_metas, teacher_meta, teacher_i);

        rlt::free(device, data);
        std::cout << "Teacher policy (" << cpp_copy.checkpoint_path.string() << ")"
                  << " mean return: " << result.returns_mean
                  << " episode length: " << result.episode_length_mean
                  << " share terminated: " << result.share_terminated << std::endl;
        if (result.returns_mean < SOLVED_RETURN){
            std::cerr << "Mean return (" << result.returns_mean << ") too low for teacher " << teacher_i << std::endl;
            return 1;
        }
    }

    for (TI i=0; i < DATASET_SIZE; i++){
        rlt::set(device, epoch_indices, i, i);
    }

    rlt::reset_optimizer_state(device, actor_optimizer, actor);

    // --- DAgger training loop ---
    for (TI epoch_i=0; epoch_i < N_EPOCH; epoch_i++){
        current_episode = ON_POLICY ? 0 : current_episode;
        current_index   = ON_POLICY ? 0 : current_index;

        RESULT average_result;
        average_result.returns_mean        = 0;
        average_result.returns_std         = 0;
        average_result.episode_length_mean = 0;
        average_result.episode_length_std  = 0;
        average_result.share_terminated    = 0;
        TI NUM_AVG = 0;

        if (epoch_i < EPOCH_TEACHER_FORCING || TEACHER_STUDENT_MIX > 0){
            // behavioral cloning: teacher rollouts
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
                constexpr TI TEACHER_EPOCHS = (TEACHER_STUDENT_MIX > 0 ? TEACHER_STUDENT_MIX : 1);
                for (TI teacher_epoch_i=0; teacher_epoch_i < TEACHER_EPOCHS; teacher_epoch_i++){
                    static_assert(DATASET_SIZE >= NUM_TEACHERS * TEACHER_EPOCHS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT);
                    auto result = gather_epoch<ENVIRONMENT_TEACHER,
                        ENVIRONMENT_TEACHER::Observation, ENVIRONMENT::Observation,
                        NUM_EPISODES, TEACHER_DETERMINISTIC>(
                            device, actor_teacher[teacher_i], teacher_meta,
                            teacher_parameters[teacher_i], actor_teacher[teacher_i],
                            dataset_episode_start_indices, dataset_input, dataset_output_target,
                            dataset_truncated, dataset_reset, current_episode, current_index, rng);
                    if (epoch_i < EPOCH_TEACHER_FORCING){
                        NUM_AVG++;
                        average_result.returns_mean        += result.returns_mean;
                        average_result.returns_std         += result.returns_mean * result.returns_mean;
                        average_result.episode_length_mean += result.episode_length_mean;
                        average_result.episode_length_std  += result.episode_length_mean * result.episode_length_mean;
                        average_result.share_terminated    += result.share_terminated;
                    }
                }
            }
        }

        if (epoch_i >= EPOCH_TEACHER_FORCING){
            // DAgger: student rolls out, teacher labels
            using RESULT_STUDENT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
            RESULT_STUDENT results[NUM_TEACHERS];
            std::vector<std::tuple<TI, T>> active_teachers;
            rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<RESULT_STUDENT::SPEC>> datas[NUM_TEACHERS];
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                rlt::malloc(device, datas[teacher_i]);
                auto& result = results[teacher_i];
                auto& data   = datas[teacher_i];
                sample_trajectories<ENVIRONMENT>(device, actor, teacher_parameters[teacher_i], result, data, rng);
                NUM_AVG++;
                average_result.returns_mean        += result.returns_mean;
                average_result.returns_std         += result.returns_mean * result.returns_mean;
                average_result.episode_length_mean += result.episode_length_mean;
                average_result.episode_length_std  += result.episode_length_mean * result.episode_length_mean;
                average_result.share_terminated    += result.share_terminated;
                active_teachers.emplace_back(teacher_i, result.returns_mean);
            }
            auto argsort = [](const auto& v, auto comp){
                std::vector<std::size_t> idx(v.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](auto i, auto j){ return comp(v[i], v[j]); });
                return idx;
            };
            auto indices = argsort(active_teachers, [](const auto& a, const auto& b){ return std::get<1>(a) < std::get<1>(b); });
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                if (indices[teacher_i] < NUM_ACTIVE_TEACHERS){
                    auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
                    add_to_dataset<ENVIRONMENT,
                        ENVIRONMENT_TEACHER::Observation, ENVIRONMENT::Observation,
                        TEACHER_DETERMINISTIC>(
                            device, datas[teacher_i], actor_teacher[teacher_i], teacher_meta,
                            dataset_episode_start_indices, dataset_input, dataset_output_target,
                            dataset_truncated, dataset_reset, current_episode, current_index, rng);
                }
                rlt::free(device, datas[teacher_i]);
            }
        }

        average_result.returns_mean        /= NUM_AVG;
        average_result.returns_std         /= NUM_AVG;
        average_result.returns_std          = std::sqrt(average_result.returns_std - average_result.returns_mean * average_result.returns_mean);
        average_result.episode_length_mean /= NUM_AVG;
        average_result.episode_length_std  /= NUM_AVG;
        average_result.episode_length_std   = std::sqrt(average_result.episode_length_std - average_result.episode_length_mean * average_result.episode_length_mean);
        average_result.share_terminated    /= NUM_AVG;

        rlt::add_scalar(device, device.logger, "evaluation/return/mean",         average_result.returns_mean);
        rlt::add_scalar(device, device.logger, "evaluation/return/std",          average_result.returns_std);
        rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", average_result.episode_length_mean);
        rlt::add_scalar(device, device.logger, "evaluation/episode_length/std",  average_result.episode_length_std);
        rlt::add_scalar(device, device.logger, "evaluation/share_terminated",    average_result.share_terminated);
        rlt::log(device, device.logger,
            (epoch_i >= EPOCH_TEACHER_FORCING ? "Student" : "Teacher"),
            " Mean return: ", average_result.returns_mean,
            " Mean episode length: ", average_result.episode_length_mean,
            " Share terminated: ", average_result.share_terminated * 100, "%");

        if (epoch_i >= EPOCH_TEACHER_FORCING && (!best_return_set || average_result.returns_mean > best_return)){
            best_return     = average_result.returns_mean;
            best_return_set = true;
            rlt::copy(device, device, actor, best_actor);
            std::cout << "Best return: " << best_return << std::endl;
        }

        TI N         = current_index;
        TI N_EPISODE = current_episode;

        for (TI i=0; i < N_EPISODE; i++){
            rlt::set(device, epoch_indices, i, i);
        }
        if constexpr(SHUFFLE){
            for (TI sample_i=0; sample_i < N_EPISODE; sample_i++){
                TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N_EPISODE - 1, rng);
                TI target_value = rlt::get(device, epoch_indices, target_index);
                rlt::set(device, epoch_indices, rlt::get(device, epoch_indices, sample_i), target_index);
                rlt::set(device, epoch_indices, target_value, sample_i);
            }
        }

        constexpr TI BATCH_SIZE_CONST = INPUT_SHAPE::GET<1>;
        T  epoch_loss       = 0;
        TI epoch_loss_count = 0;
        TI epoch_episode_index = 0;
        std::cout << "Epoch: " << epoch_i << " has " << N << " samples and " << N_EPISODE << " episodes" << std::endl;
        TI N_BATCHES_TARGET = std::max((TI)1, N_EPISODE / BATCH_SIZE_CONST);
        TI batch_i       = 0;
        TI epoch_episode = rlt::get(device, epoch_indices, epoch_episode_index);
        TI current_sample = rlt::get(device, dataset_episode_start_indices, epoch_episode);

        while (batch_i < N_BATCHES_TARGET){
            for (TI sample_i=0; sample_i < BATCH_SIZE_CONST; sample_i++){
                bool reset = false;
                for (TI step_i=0; step_i < SEQUENCE_LENGTH; step_i++){
                    auto input             = rlt::view(device, dataset_input,       current_sample);
                    auto input_target_step = rlt::view(device, batch_input,         step_i);
                    auto input_target      = rlt::view(device, input_target_step,   sample_i);
                    rlt::copy(device, device, input, input_target);

                    auto output_target      = rlt::view(device, dataset_output_target, current_sample);
                    auto output_target_step = rlt::view(device, batch_output_target,   step_i);
                    auto output_target_tgt  = rlt::view(device, output_target_step,    sample_i);
                    rlt::copy(device, device, output_target, output_target_tgt);

                    rlt::set(device, batch_reset, reset, step_i, sample_i, 0);

                    TI next_sample = (current_sample + 1) % N;
                    reset = next_sample == 0 || rlt::get(device, dataset_truncated, current_sample);
                    if (!reset){
                        current_sample = next_sample;
                    } else {
                        if (++epoch_episode_index >= N_EPISODE){
                            epoch_episode_index = 0;
                        }
                        current_sample = rlt::get(device, dataset_episode_start_indices,
                                                  rlt::get(device, epoch_indices, epoch_episode_index));
                    }
                }
            }

            rlt::Mode<rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(batch_reset)>>> mode;
            mode.reset_container = batch_reset;
            rlt::forward(device, actor, batch_input, actor_buffer, rng, mode);

            auto output_matrix_view        = rlt::matrix_view(device, rlt::output(device, actor));
            auto output_target_matrix_view = rlt::matrix_view(device, batch_output_target);
            auto d_output_matrix_view      = rlt::matrix_view(device, d_output);

            rlt::nn::loss_functions::mse::gradient(device, output_matrix_view, output_target_matrix_view, d_output_matrix_view);
            T loss = rlt::nn::loss_functions::mse::evaluate(device, output_matrix_view, output_target_matrix_view);

            rlt::set_step(device, device.logger, epoch_i * (N / BATCH_SIZE_CONST) + batch_i);
            rlt::add_scalar(device, device.logger, "loss", loss);
            epoch_loss += loss;
            epoch_loss_count++;

            rlt::zero_gradient(device, actor);
            rlt::backward(device, actor, batch_input, d_output, actor_buffer, mode);
            rlt::step(device, actor_optimizer, actor);
            batch_i++;
        }

        std::cout << "Epoch: " << epoch_i
                  << " #Batches: " << batch_i
                  << " Loss: " << epoch_loss / epoch_loss_count << std::endl;

        // save per-epoch checkpoint
        auto target_path = run_path / "checkpoints" / std::to_string(epoch_i);
        if (!std::filesystem::exists(target_path)){
            std::filesystem::create_directories(target_path);
        }
        rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT, CHECKPOINT_PARAMETERS>(device, target_path.string(), actor, rng);

        // evaluate student on a fresh environment sample
        {
            using EVAL_ACTOR_BS   = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
            using EVAL_ACTOR_TYPE = typename EVAL_ACTOR_BS::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
            rlt::rl::environments::DummyUI ui;
            EVAL_ACTOR_TYPE eval_actor;
            typename EVAL_ACTOR_TYPE::template Buffer<DYNAMIC_ALLOCATION> eval_buffer;
            rlt::malloc(device, eval_actor);
            rlt::malloc(device, eval_buffer);
            rlt::copy(device, device, actor, eval_actor);

            ENVIRONMENT env_eval;
            ENVIRONMENT::Parameters env_eval_parameters;
            rlt::init(device, env_eval);
            rlt::sample_initial_parameters(device, env_eval, env_eval_parameters, rng);
            rlt::Mode<rlt::mode::Default<>> eval_mode;
            RESULT_EVAL result_eval;
            DATA_EVAL   data_eval;
            rlt::evaluate(device, env_eval, ui, eval_actor, result_eval, data_eval, rng, eval_mode);

            rlt::add_scalar(device, device.logger, "student/return/mean",         result_eval.returns_mean);
            rlt::add_scalar(device, device.logger, "student/return/std",          result_eval.returns_std);
            rlt::add_scalar(device, device.logger, "student/episode_length/mean", result_eval.episode_length_mean);
            rlt::add_scalar(device, device.logger, "student/share_terminated",    result_eval.share_terminated);
            rlt::log(device, device.logger,
                "Student eval: Mean return: ", result_eval.returns_mean,
                " ep_len: ", result_eval.episode_length_mean,
                " terminated: ", result_eval.share_terminated * 100, "%");

            rlt::free(device, eval_actor);
            rlt::free(device, eval_buffer);
        }
    }

    // save best checkpoint
    rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT, CHECKPOINT_PARAMETERS>(device, run_path.string(), best_actor, rng);

    // --- free ---
    rlt::free(device, rng);
    rlt::free(device, actor_optimizer);
    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        rlt::free(device, actor_teacher[teacher_i]);
    }
    rlt::free(device, actor_teacher_buffer);
    rlt::free(device, actor);
    rlt::free(device, best_actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, teacher_metas);
    rlt::free(device, dataset_episode_start_indices);
    rlt::free(device, dataset_input);
    rlt::free(device, dataset_output_target);
    rlt::free(device, dataset_truncated);
    rlt::free(device, dataset_reset);
    rlt::free(device, epoch_indices);
    rlt::free(device, batch_input);
    rlt::free(device, batch_output_target);
    rlt::free(device, batch_reset);
    rlt::free(device, d_output);
    return 0;
}
