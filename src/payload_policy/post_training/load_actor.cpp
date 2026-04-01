// Compile & run (from repo root):
//   g++ -std=c++17 src/payload_policy/post_training/load_actor.cpp -I include && ./a.out
//
// This verifies the exported GRU student checkpoint is inference-compatible with the
// DefaultARM device (same code path as the embedded PX4 module).
// Replace the checkpoint include below with the path produced by payload_policy_post_training.

#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION // allow ARM + CPU in same TU
#include <rl_tools/operations/arm.h>
#include <rl_tools/operations/cpu.h> // only for rlt::print

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

// ── Replace with the checkpoint.h produced by payload_policy_post_training ──
#include "../../../logs/PLACEHOLDER_TIMESTAMP/checkpoints/PLACEHOLDER_EPOCH/checkpoint.h"
// ─────────────────────────────────────────────────────────────────────────────

namespace rlt = rl_tools;
namespace checkpoint_ns = rl_tools::checkpoint::actor;

using DEVICE       = rlt::devices::DefaultARM;
using DEVICE_PRINT = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t;
using T  = checkpoint_ns::TYPE::T;

constexpr TI INPUT_DIM  = rlt::get_last(checkpoint_ns::TYPE::INPUT_SHAPE{});
constexpr TI OUTPUT_DIM = rlt::get_last(checkpoint_ns::TYPE::OUTPUT_SHAPE{});

int main(){
    // Single-step GRU inference (matches the embedded module usage)
    using MODEL = checkpoint_ns::TYPE
        ::template CHANGE_BATCH_SIZE<TI, 1>
        ::template CHANGE_SEQUENCE_LENGTH<TI, 1>;

    MODEL::Buffer<false>         buffer; // stack allocation
    MODEL::template State<false> state;  // GRU hidden state
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::INPUT_SHAPE,  false>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::OUTPUT_SHAPE, false>> output;

    DEVICE       device;
    DEVICE_PRINT device_print;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::init(device, rng, (TI)0);

    // Zero the GRU hidden state (cold start)
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::reset(device, checkpoint_ns::module, state, rng);

    // Fill input with random values and run one inference step
    rlt::randn(device, input, rng);
    rlt::evaluate_step(device, checkpoint_ns::module, input, state, output, buffer, rng, mode);

    rlt::log(device_print, device_print.logger, "GRU step output (INPUT_DIM=",
             INPUT_DIM, ", OUTPUT_DIM=", OUTPUT_DIM, "): ");
    rlt::print(device_print, output);

    return 0;
}
