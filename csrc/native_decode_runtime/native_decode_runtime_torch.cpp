#include "native_decode_runtime.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(flux, library) {
    library.class_<flux::NativeSmolLM2LayerDecode>(
        "NativeSmolLM2LayerDecode")
        .def(torch::init<
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            at::Tensor,
            std::int64_t,
            std::int64_t,
            double,
            double>())
        .def("replay", &flux::NativeSmolLM2LayerDecode::replay)
        .def("reset", &flux::NativeSmolLM2LayerDecode::reset)
        .def("output", &flux::NativeSmolLM2LayerDecode::output)
        .def("key_cache", &flux::NativeSmolLM2LayerDecode::key_cache)
        .def("value_cache", &flux::NativeSmolLM2LayerDecode::value_cache)
        .def("device_position", &flux::NativeSmolLM2LayerDecode::device_position)
        .def("workspace", &flux::NativeSmolLM2LayerDecode::workspace)
        .def("addresses", &flux::NativeSmolLM2LayerDecode::addresses)
        .def("position", &flux::NativeSmolLM2LayerDecode::position)
        .def("cache_length", &flux::NativeSmolLM2LayerDecode::cache_length)
        .def("capacity", &flux::NativeSmolLM2LayerDecode::capacity)
        .def("replay_count", &flux::NativeSmolLM2LayerDecode::replay_count)
        .def("workspace_bytes", &flux::NativeSmolLM2LayerDecode::workspace_bytes);

    library.class_<flux::NativeSmolLM2Decode>("NativeSmolLM2Decode")
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
            std::vector<at::Tensor>,
            std::vector<at::Tensor>,
            std::int64_t,
            std::int64_t,
            std::vector<double>,
            std::vector<double>>())
        .def("replay", &flux::NativeSmolLM2Decode::replay)
        .def("generate_greedy", &flux::NativeSmolLM2Decode::generate_greedy)
        .def("reset", &flux::NativeSmolLM2Decode::reset)
        .def("logits", &flux::NativeSmolLM2Decode::logits)
        .def("current_token", &flux::NativeSmolLM2Decode::current_token)
        .def("generated_tokens", &flux::NativeSmolLM2Decode::generated_tokens)
        .def("device_generation_step", &flux::NativeSmolLM2Decode::device_generation_step)
        .def("key_cache", &flux::NativeSmolLM2Decode::key_cache)
        .def("value_cache", &flux::NativeSmolLM2Decode::value_cache)
        .def("device_position", &flux::NativeSmolLM2Decode::device_position)
        .def("device_cache_length", &flux::NativeSmolLM2Decode::device_cache_length)
        .def("workspace", &flux::NativeSmolLM2Decode::workspace)
        .def("addresses", &flux::NativeSmolLM2Decode::addresses)
        .def("position", &flux::NativeSmolLM2Decode::position)
        .def("cache_length", &flux::NativeSmolLM2Decode::cache_length)
        .def("capacity", &flux::NativeSmolLM2Decode::capacity)
        .def("replay_count", &flux::NativeSmolLM2Decode::replay_count)
        .def("generation_step", &flux::NativeSmolLM2Decode::generation_step)
        .def("workspace_bytes", &flux::NativeSmolLM2Decode::workspace_bytes)
        .def("stable_buffer_bytes", &flux::NativeSmolLM2Decode::stable_buffer_bytes)
        .def("vocabulary_size", &flux::NativeSmolLM2Decode::vocabulary_size)
        .def("layer_count", &flux::NativeSmolLM2Decode::layer_count);
}
