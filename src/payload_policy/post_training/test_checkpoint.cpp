#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

// ── Replace with the student checkpoint to test ──
#include "../../../logs/PLACEHOLDER_TIMESTAMP/checkpoints/PLACEHOLDER_EPOCH/checkpoint.h"
// ─────────────────────────────────────────────────

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG    = DEVICE::SPEC::RANDOM::ENGINE<>;
using T      = float;
using TI     = typename DEVICE::index_t;

#include "environment.h"
#include "../pre_training/options.h"
#include "config.h"

int main(){
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, (TI)100);

    // Verify against the example embedded in the checkpoint header
    using ACTOR_TYPE = rl_tools::checkpoint::actor::TYPE
        ::template CHANGE_BATCH_SIZE<TI, 1>
        ::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
    ACTOR_TYPE::Buffer<false>         buffer;
    ACTOR_TYPE::template State<false> state;

    rl_tools::checkpoint::actor::TYPE::Buffer<> test_buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::example::output::SHAPE>> output;
    rlt::malloc(device, test_buffer);
    rlt::malloc(device, output);

    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rl_tools::checkpoint::actor::module,
                  rl_tools::checkpoint::example::input::container,
                  output, test_buffer, rng, mode);

    T abs_diff = rlt::abs_diff(device, rl_tools::checkpoint::example::output::container, output)
               / rl_tools::checkpoint::example::output::SPEC::SIZE;

    auto last_step_output   = rlt::view(device, output,
        rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);
    auto last_step_expected = rlt::view(device, rl_tools::checkpoint::example::output::container,
        rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);

    std::cout << "last_step_output:\n";   rlt::print(device, last_step_output);
    std::cout << "last_step_expected:\n"; rlt::print(device, last_step_expected);
    std::cout << "abs_diff to checkpoint example: " << abs_diff << std::endl;

    rlt::free(device, test_buffer);
    rlt::free(device, output);
    return abs_diff < 1e-5 ? 0 : 1;
}
