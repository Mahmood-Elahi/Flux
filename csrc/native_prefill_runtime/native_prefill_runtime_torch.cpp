#include "native_prefill_runtime.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.class_<flux::NativeSmolLM2Prefill>("NativeSmolLM2Prefill")
        .def(torch::init<
            at::Tensor,
            at::Tensor,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            std::int64_t,
            std::vector<double>,
            std::vector<double>>())
        .def("prefill", &flux::NativeSmolLM2Prefill::prefill)
        .def("replay", &flux::NativeSmolLM2Prefill::replay)
        .def("logits", &flux::NativeSmolLM2Prefill::logits)
        .def("final_hidden", &flux::NativeSmolLM2Prefill::final_hidden)
        .def("key_cache", &flux::NativeSmolLM2Prefill::key_cache)
        .def("value_cache", &flux::NativeSmolLM2Prefill::value_cache)
        .def("device_position", &flux::NativeSmolLM2Prefill::device_position)
        .def("device_cache_length", &flux::NativeSmolLM2Prefill::device_cache_length)
        .def("addresses", &flux::NativeSmolLM2Prefill::addresses)
        .def("position", &flux::NativeSmolLM2Prefill::position)
        .def("cache_length", &flux::NativeSmolLM2Prefill::cache_length)
        .def("prompt_length", &flux::NativeSmolLM2Prefill::prompt_length)
        .def("capacity", &flux::NativeSmolLM2Prefill::capacity)
        .def("replay_count", &flux::NativeSmolLM2Prefill::replay_count)
        .def("workspace_bytes", &flux::NativeSmolLM2Prefill::workspace_bytes)
        .def("cache_bytes", &flux::NativeSmolLM2Prefill::cache_bytes)
        .def("stable_buffer_bytes", &flux::NativeSmolLM2Prefill::stable_buffer_bytes);
}
