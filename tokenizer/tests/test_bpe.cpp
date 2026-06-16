// Unit-Tests fuer den BPE-Tokenizer (Person 2, Woche 3).
// Einfache assert-basierte Tests (alternativ Catch2). Siehe plan.md.
//
// Geplante Tests:
//   - encode -> decode Roundtrip
//   - Unbekannte Zeichen -> unk_token_id
//   - Spezial-Tokens korrekt (pad=0, eos=1, bos=2, unk=3)
#include <cassert>
#include "bpe.hpp"

static void test_special_tokens() {
    BPE tok;
    assert(tok.pad_token_id() == 0);
    assert(tok.eos_token_id() == 1);
    assert(tok.bos_token_id() == 2);
    assert(tok.unk_token_id() == 3);
}

int main() {
    test_special_tokens();
    // TODO: Roundtrip- und Unknown-Token-Tests, sobald train/encode/decode stehen.
    return 0;
}
