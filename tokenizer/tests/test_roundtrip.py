"""Unit-Tests fuer den Tokenizer (Person 1, Woche 4).

Geplante Tests (siehe plan.md):
  - encode(decode(ids)) == ids  (Roundtrip)
  - Unbekannte Zeichen -> <unk>
  - Leerer String -> [2, 1]  (BOS, EOS)
  - Einrueckung bleibt erhalten
"""


def test_placeholder():
    # TODO: echte Tests implementieren, sobald encode/decode stehen
    assert True
