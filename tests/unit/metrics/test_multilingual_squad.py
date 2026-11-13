"""The language-aware official-SQuAD scorer of xquad / multi_wiki_qa and the multilingual normaliser (11-13 review fixes):
(1) the Unicode bidi formatting marks are removed in every language, (2) F1 of Korean is over characters (KorQuAD convention).
Plain functions without fixtures: run with pytest, or `python tests/unit/metrics/test_multilingual_squad.py`. Korean needs the stanza model in
$STANZA_RESOURCES_DIR (the test is skipped without it); Arabic / English / Russian / Chinese need the spaCy blank pipelines of the llm_env.
"""

import os
import unittest

from lighteval.metrics.dynamic_metrics import CHARACTER_LEVEL_F1_LANGUAGES, MultilingualSquadOfficialMetric
from lighteval.metrics.normalizations import BIDI_MARKS, get_multilingual_normalizer, remove_bidi_marks
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def _score(language, score, golds, prediction):
    doc = Doc(query="q", choices=list(golds), gold_index=list(range(len(golds))))
    return MultilingualSquadOfficialMetric(Language(language), score).sample_level_fn.compute(doc, ModelResponse(text=[prediction]))


def test_the_bidi_mark_set_is_the_one_the_owner_named():
    expected = {0x200E, 0x200F, 0x061C} | set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))
    assert {ord(c) for c in BIDI_MARKS} == expected and len(BIDI_MARKS) == len(expected) == 12
    assert remove_bidi_marks("a" + "".join(BIDI_MARKS) + "b") == "ab"
    # not touched: zero-width joiners (meaningful in Persian / Indic scripts), the zero-width space, the soft hyphen, ordinary punctuation
    for keep in ("‌", "‍", "​", "­", "-", " "):
        assert remove_bidi_marks(f"a{keep}b") == f"a{keep}b"


def test_bidi_marks_do_not_change_any_language_except_through_themselves():
    for language, text in (("ara", "هولاكو خان"), ("eng", "The Eiffel Tower, 1889"), ("rus", "Город Москва"), ("zho", "上海市"), ("deu", "Die Stadt")):
        normalise = get_multilingual_normalizer(Language(language))
        plain = normalise(text)
        for mark in BIDI_MARKS:
            assert normalise(mark + text) == plain and normalise(text + mark) == plain, (language, hex(ord(mark)))
            assert normalise(text.replace(" ", " " + mark)) == plain, (language, hex(ord(mark)))
    # the real case: XQuAD ara, the model's answer opens with U+200F
    assert _score("ara", "em", [" هولاكو خان"], " ‏هولاكو خان") == 1
    assert _score("ara", "f1", [" هولاكو خان"], " ‏هولاكو خان") == 1
    assert _score("ara", "em", [" هولاكو خان"], " هولاكو خان") == 1
    assert _score("ara", "em", [" هولاكو خان"], " ‏هولاكو") == 0  # a missing word still counts


def _korean():
    from lighteval.metrics.utils.linguistic_tokenizers import get_word_tokenizer

    if not os.environ.get("STANZA_RESOURCES_DIR"):
        raise unittest.SkipTest("STANZA_RESOURCES_DIR not set (stanza ko model)")
    try:
        get_word_tokenizer(Language("kor")).word_tokenize("테스트")
    except Exception as error:  # model folder missing
        raise unittest.SkipTest(f"stanza ko not available: {type(error).__name__}")


def test_korean_f1_is_over_characters_and_em_is_unchanged():
    _korean()
    assert CHARACTER_LEVEL_F1_LANGUAGES == frozenset({Language("kor")})
    # word-level F1 gave 0 here: the particle / counter is part of the word
    assert _score("kor", "f1", ["1988"], "1988년") > 0.5
    assert abs(_score("kor", "f1", ["1988"], "1988년") - 2 * (4 / 5) * 1 / (4 / 5 + 1)) < 1e-9  # precision 4/5, recall 1
    assert _score("kor", "f1", ["서울"], "서울에서") > 0.5 and _score("kor", "f1", ["줄리에타 마시나"], "줄리에타 마시나는") > 0.8
    assert _score("kor", "f1", ["1988"], "1988") == 1
    assert _score("kor", "f1", ["1988"], "2008") == 0.25  # character-level partial credit (one shared "8"), as in KorQuAD
    assert _score("kor", "f1", ["서울"], "부산") == 0
    assert _score("kor", "f1", ["줄리에타 마시나"], "줄리에타마시나") == 1  # white space does not matter at character level
    # max over the golds, as in the official script
    assert _score("kor", "f1", ["부산", "서울"], "서울에서") == _score("kor", "f1", ["서울"], "서울에서")
    # EM is the equality of the normalised strings, unchanged: the extra particle is a different answer
    assert _score("kor", "em", ["1988"], "1988년") == 0 and _score("kor", "em", ["1988"], "1988") == 1
    # every other language keeps the word-level F1 (ara, tur, rus are agglutinative / inflected too: unchanged)
    assert _score("eng", "f1", ["new york city"], "new york") == 0.8 and _score("eng", "f1", ["1988"], "1988s") == 0
    assert _score("rus", "f1", ["Москва"], "Москве") == 0


def test_tasks_that_use_the_scorer_are_version_1_and_only_korean_f1_is_character_level():
    from lighteval.tasks.multilingual.tasks import multi_wiki_qa, xquad

    for table in (xquad.TASKS_TABLE, multi_wiki_qa.TASKS_TABLE):
        gen = [c for c in table if c.name.endswith(":gen")]
        assert gen and all(c.version == 1 for c in gen), "bidi marks removed (and kor character-level F1): task version 1"
        assert all(c.version == 0 for c in table if c.name.endswith(":bpb")), "the BPB tasks do not use the normaliser"
        for c in gen:
            language = c.name.split(":")[1]
            f1 = next(m for m in c.metrics if m.metric_name == "f1").sample_level_fn
            em = next(m for m in c.metrics if m.metric_name == "em").sample_level_fn
            assert f1.char_level_f1 == (language == "kor") and not em.char_level_f1, c.name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except unittest.SkipTest as skip:
                print("skip", name, skip)
