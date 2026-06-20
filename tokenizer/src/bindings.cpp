// pybind11-Binding: exportiert BPE als Python-Modul `tokenizer_cpp`.
// Person 2, Woche 4. Siehe plan.md + docs/interface.md.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bpe.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tokenizer_cpp, m) {
    py::class_<BPE>(m, "Tokenizer")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("path"))  // Tokenizer("vocab.bin")
        .def("train",  &BPE::train)
        .def("save",   &BPE::save)
        .def("load",   &BPE::load)
        .def("encode", &BPE::encode)
        .def("decode", &BPE::decode)
        .def_property_readonly("vocab_size",   &BPE::vocab_size)
        .def_property_readonly("pad_token_id", &BPE::pad_token_id)
        .def_property_readonly("eos_token_id", &BPE::eos_token_id)
        .def_property_readonly("bos_token_id", &BPE::bos_token_id)
        .def_property_readonly("unk_token_id", &BPE::unk_token_id);
}
