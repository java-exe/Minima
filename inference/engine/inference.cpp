// C++ Inference Engine Implementierung (Person 2, Woche 10-11).
// Siehe plan.md, Phase 3.
#include "inference.hpp"

#include <stdexcept>

InferenceEngine::InferenceEngine(const std::string& model_path,
                                 const std::string& vocab_path) {
    // TODO: onnxruntime Session aus model_path erstellen,
    //       tokenizer_.load(vocab_path).
    (void)model_path;
    (void)vocab_path;
    throw std::runtime_error("InferenceEngine ctor not implemented");
}

std::string InferenceEngine::generate(const std::string& prompt,
                                      int max_new_tokens,
                                      float temperature,
                                      int top_k) {
    // TODO: prompt encoden -> autoregressiv Tokens samplen (Temperature + Top-K)
    //       -> decoden -> string zurueck.
    (void)prompt;
    (void)max_new_tokens;
    (void)temperature;
    (void)top_k;
    throw std::runtime_error("InferenceEngine::generate not implemented");
}
