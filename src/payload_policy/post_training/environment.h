#include "../pre_training/environment.h"

namespace builder{
    // For payload post-training, the environment is the same as pre-training
    // (motor delay and action history are already present in pre-training)
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY_POST_TRAINING : ENVIRONMENT_FACTORY<DEVICE, T, TI, OPTIONS>{};
}
