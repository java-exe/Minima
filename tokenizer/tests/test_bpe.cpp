// Unit-Tests fuer den BPE-Tokenizer (Person 2, Woche 3).
// Einfache assert-basierte Tests (alternativ Catch2). Siehe plan.md.
//
// Abgedeckte Tests:
//   - Spezial-Tokens korrekt (pad=0, eos=1, bos=2, unk=3)
//   - encode beginnt mit <bos> und endet mit <eos>
//   - encode -> decode Roundtrip (bekannter Text)
//   - Unbekannte Zeichen -> unk_token_id
//   - Einrueckungen bleiben erhalten (Roundtrip ueber mehrzeiligen Code)
//   - save/load reproduziert encode-Ergebnis exakt
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "bpe.hpp"

static std::vector<std::string> sample_corpus() {
    return {
        "def add(a, b):\n    return a + b\n",
        "def sub(a, b):\n    return a - b\n",
        "def mul(a, b):\n    return a * b\n",
        "class Stack:\n    def __init__(self):\n        self.items = []\n",
        "for i in range(10):\n    print(i)\n",
    };
}

static void test_special_tokens() {
    BPE tok;
    assert(tok.pad_token_id() == 0);
    assert(tok.eos_token_id() == 1);
    assert(tok.bos_token_id() == 2);
    assert(tok.unk_token_id() == 3);
    assert(tok.vocab_size() == 4);  // frisch: nur die Spezial-Tokens
}

static void test_bos_eos_and_roundtrip() {
    BPE tok;
    tok.train(sample_corpus(), 100);
    assert(tok.vocab_size() > 4);

    const std::string text = "def add(a, b):\n    return a + b\n";
    std::vector<int> ids = tok.encode(text);

    assert(!ids.empty());
    assert(ids.front() == tok.bos_token_id());
    assert(ids.back() == tok.eos_token_id());

    // Roundtrip: bekannter Text muss exakt rekonstruiert werden (inkl. Einrueckung).
    assert(tok.decode(ids) == text);
}

static void test_unknown_char() {
    BPE tok;
    tok.train(sample_corpus(), 50);  // Corpus enthaelt kein '@'

    std::vector<int> ids = tok.encode("@");  // einzelnes, unbekanntes Byte
    // [bos, unk, eos] -- das unbekannte Zeichen wird zu <unk>.
    assert(ids.size() == 3);
    assert(ids[0] == tok.bos_token_id());
    assert(ids[1] == tok.unk_token_id());
    assert(ids[2] == tok.eos_token_id());
}

static void test_indentation_preserved() {
    BPE tok;
    tok.train(sample_corpus(), 100);
    const std::string code = "class Stack:\n    def __init__(self):\n        self.items = []\n";
    assert(tok.decode(tok.encode(code)) == code);
}

static void test_save_load() {
    BPE tok;
    tok.train(sample_corpus(), 100);

    const std::string path = "test_vocab.bin";
    tok.save(path);

    BPE loaded(path);  // Konstruktor mit Pfad
    assert(loaded.vocab_size() == tok.vocab_size());

    const std::string text = "def mul(a, b):\n    return a * b\n";
    assert(loaded.encode(text) == tok.encode(text));
    assert(loaded.decode(loaded.encode(text)) == text);

    std::remove(path.c_str());
}

int main() {
    test_special_tokens();
    test_bos_eos_and_roundtrip();
    test_unknown_char();
    test_indentation_preserved();
    test_save_load();
    std::printf("All BPE tests passed.\n");
    return 0;
}
