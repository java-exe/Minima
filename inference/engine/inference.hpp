// C++ Inference Engine (Person 2, Woche 10-11). Siehe plan.md, Phase 3.
// Laedt model.onnx + vocab.bin und generiert Code ohne Python zur Laufzeit.
#pragma once

#include <string>
// #include <onnxruntime_cxx_api.h>   // Ort::Session etc.
#include "bpe.hpp"

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path,
                    const std::string& vocab_path);

    std::string generate(const std::string& prompt,
                         int   max_new_tokens = 200,
                         float temperature    = 0.8f,
                         int   top_k          = 50);

private:
    // Ort::Session session_;   // onnxruntime Session
    BPE tokenizer_;
    // Top-K Sampling intern
};
