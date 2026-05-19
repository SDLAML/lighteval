"""
name:
Mgsm

dataset:
CohereLabs/global-mgsm

abstract:
MGSM (Multilingual Grade School Math) is a multilingual benchmark testing
mathematical reasoning across languages, derived from GSM8K.

Uses CoT-style prompts matching English gsm8k.py ("Think step by step /
Problem: / Solution:"), with language-specific translations for all non-English
languages in CohereLabs/global-mgsm.

English is excluded — use gsm8k.py for English evaluation.

Reports both expr_gold_metric (math expression parser) and
MultilingualQuasiExactMatchMetric (language-aware fuzzy match, handles
non-ASCII digit systems like Japanese/Thai).

languages:
amharic, arabic, bengali, catalan, czech, welsh, german, greek, spanish,
basque, french, galician, gujarati, hausa, hungarian, japanese, khmer,
kannada, korean, kyrgyz, ganda, burmese, nepali, russian, sinhala, shona,
serbian, southern sotho, swahili, tamil, telugu, thai, urdu, uzbek,
vietnamese, wolof, xhosa, yoruba, chinese, zulu

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2210.03057
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


# Languages covered by CohereLabs/global-mgsm.
# English is intentionally excluded — use gsm8k.py for English evaluation.
_LANGUAGES = [
    Language.AMHARIC,
    Language.ARABIC,
    Language.BENGALI,
    Language.CATALAN,
    Language.CZECH,
    Language.WELSH,
    Language.GERMAN,
    Language.GREEK,
    Language.SPANISH,
    Language.BASQUE,
    Language.FRENCH,
    Language.GALICIAN,
    Language.GUJARATI,
    Language.HAUSA,
    Language.HUNGARIAN,
    Language.JAPANESE,
    Language.KHMER,
    Language.KANNADA,
    Language.KOREAN,
    Language.KIRGHIZ,
    Language.GANDA,
    Language.BURMESE,
    Language.NEPALI,
    Language.RUSSIAN,
    Language.SINHALA,
    Language.SHONA,
    Language.SERBIAN,
    Language.SOUTHERN_SOTHO,
    Language.SWAHILI,
    Language.TAMIL,
    Language.TELUGU,
    Language.THAI,
    Language.URDU,
    Language.UZBEK,
    Language.VIETNAMESE,
    Language.WOLOF,
    Language.XHOSA,
    Language.YORUBA,
    Language.CHINESE,
    Language.ZULU,
]


# Each entry is:
#   instruction, problem_label, solution_label
#
# The instruction is intentionally close in meaning across languages:
#   "Solve the following math problem. Think step by step before giving
#    the final answer."
#
# NOTE:
# Some low-resource translations should ideally receive native-speaker review
# before being treated as final benchmark wording.
_MGSM_PROMPT_PARTS: dict[Language, tuple[str, str, str]] = {
    Language.AMHARIC: (
        "የሚከተለውን የሂሳብ ችግር ፍታ። የመጨረሻውን መልስ ከመስጠትህ በፊት ደረጃ በደረጃ አስብ።",
        "ችግር:",
        "መፍትሔ:",
    ),
    Language.ARABIC: (
        "حل المسألة الرياضية التالية. فكّر خطوة بخطوة قبل تقديم الإجابة النهائية.",
        "المسألة:",
        "الحل:",
    ),
    Language.BENGALI: (
        "নিম্নলিখিত গণিত সমস্যাটি সমাধান করো। চূড়ান্ত উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করো।",
        "সমস্যা:",
        "সমাধান:",
    ),
    Language.CATALAN: (
        "Resol el següent problema de matemàtiques. Pensa pas a pas abans de donar la resposta final.",
        "Problema:",
        "Solució:",
    ),
    Language.CZECH: (
        "Vyřeš následující matematickou úlohu. Než uvedeš konečnou odpověď, přemýšlej krok za krokem.",
        "Úloha:",
        "Řešení:",
    ),
    Language.WELSH: (
        "Datrysa'r broblem fathemateg ganlynol. Meddylia gam wrth gam cyn rhoi'r ateb terfynol.",
        "Problem:",
        "Datrysiad:",
    ),
    Language.GERMAN: (
        "Löse die folgende Mathematikaufgabe. Denke Schritt für Schritt nach, bevor du die endgültige Antwort gibst.",
        "Aufgabe:",
        "Lösung:",
    ),
    Language.GREEK: (
        "Λύσε το παρακάτω μαθηματικό πρόβλημα. Σκέψου βήμα προς βήμα πριν δώσεις την τελική απάντηση.",
        "Πρόβλημα:",
        "Λύση:",
    ),
    Language.SPANISH: (
        "Resuelve el siguiente problema de matemáticas. Piensa paso a paso antes de dar la respuesta final.",
        "Problema:",
        "Solución:",
    ),
    Language.BASQUE: (
        "Ebatzi honako matematikako problema hau. Pentsatu pausoz pauso azken erantzuna eman aurretik.",
        "Problema:",
        "Ebazpena:",
    ),
    Language.FRENCH: (
        "Résous le problème de mathématiques suivant. Réfléchis étape par étape avant de donner la réponse finale.",
        "Problème :",
        "Solution:",
    ),
    Language.GALICIAN: (
        "Resolve o seguinte problema matemático. Pensa paso a paso antes de dar a resposta final.",
        "Problema:",
        "Solución:",
    ),
    Language.GUJARATI: (
        "નીચે આપેલ ગણિતનો પ્રશ્ન ઉકેલો. અંતિમ જવાબ આપતા પહેલાં પગલું દર પગલું વિચારો.",
        "પ્રશ્ન:",
        "ઉકેલ:",
    ),
    Language.HAUSA: (
        "Warware matsalar lissafi mai zuwa. Yi tunani mataki-mataki kafin ka bayar da amsar ƙarshe.",
        "Matsala:",
        "Magani:",
    ),
    Language.HUNGARIAN: (
        "Oldd meg a következő matematikai feladatot. Gondolkodj lépésről lépésre, mielőtt megadnád a végső választ.",
        "Feladat:",
        "Megoldás:",
    ),
    Language.JAPANESE: (
        "次の数学の問題を解きなさい。最終的な答えを出す前に、順を追って考えなさい。",
        "問題：",
        "解答：",
    ),
    Language.KHMER: (
        "ដោះស្រាយលំហាត់គណិតវិទ្យាខាងក្រោម។ សូមគិតជាជំហានៗ មុននឹងផ្តល់ចម្លើយចុងក្រោយ។",
        "លំហាត់:",
        "ដំណោះស្រាយ:",
    ),
    Language.KANNADA: (
        "ಕೆಳಗಿನ ಗಣಿತ ಸಮಸ್ಯೆಯನ್ನು ಪರಿಹರಿಸಿ. ಅಂತಿಮ ಉತ್ತರವನ್ನು ನೀಡುವ ಮೊದಲು ಹಂತ ಹಂತವಾಗಿ ಯೋಚಿಸಿ.",
        "ಸಮಸ್ಯೆ:",
        "ಪರಿಹಾರ:",
    ),
    Language.KOREAN: (
        "다음 수학 문제를 풀어라. 최종 답을 제시하기 전에 단계별로 생각하라.",
        "문제:",
        "풀이:",
    ),
    Language.KIRGHIZ: (
        "Төмөнкү математикалык маселени чыгар. Акыркы жоопту берүүдөн мурун кадам сайын ойлон.",
        "Маселе:",
        "Чечим:",
    ),
    Language.GANDA: (
        "Gonjoola ekibuuzo kya okubala kino wammanga. Lowooza mutendera ku mutendera nga tonnawa ky’okuddamu ekisembayo.",
        "Ekibuuzo:",
        "Okugonjoola:",
    ),
    Language.BURMESE: (
        "အောက်ပါ သင်္ချာပုစ္ဆာကို ဖြေရှင်းပါ။ နောက်ဆုံးအဖြေ မပေးမီ အဆင့်လိုက် စဉ်းစားပါ။",
        "ပုစ္ဆာ:",
        "ဖြေရှင်းချက်:",
    ),
    Language.NEPALI: (
        "तलको गणित समस्या हल गर। अन्तिम उत्तर दिनुअघि चरणबद्ध रूपमा सोच।",
        "समस्या:",
        "समाधान:",
    ),
    Language.RUSSIAN: (
        "Реши следующую математическую задачу. Рассуждай пошагово, прежде чем дать окончательный ответ.",
        "Задача:",
        "Решение:",
    ),
    Language.SINHALA: (
        "පහත ගණිත ගැටලුව විසඳන්න. අවසාන පිළිතුර ලබාදීමට පෙර පියවරෙන් පියවර සිතන්න.",
        "ගැටලුව:",
        "විසඳුම:",
    ),
    Language.SHONA: (
        "Gadzirisa dambudziko remasvomhu rinotevera. Funga nhanho nenhanho usati wapa mhinduro yekupedzisira.",
        "Dambudziko:",
        "Mhinduro:",
    ),
    Language.SERBIAN: (
        "Реши следећи математички задатак. Размишљај корак по корак пре него што даш коначан одговор.",
        "Задатак:",
        "Решење:",
    ),
    Language.SOUTHERN_SOTHO: (
        "Rarolla bothata bona ba dipalo bo latelang. Nahana mohato ka mohato pele o fana ka karabo ya ho qetela.",
        "Bothata:",
        "Tharollo:",
    ),
    Language.SWAHILI: (
        "Tatua tatizo lifuatalo la hisabati. Fikiri hatua kwa hatua kabla ya kutoa jibu la mwisho.",
        "Tatizo:",
        "Suluhisho:",
    ),
    Language.TAMIL: (
        "பின்வரும் கணிதப் பிரச்சினையைத் தீர்க்கவும். இறுதி விடையை அளிப்பதற்கு முன் படிப்படியாக சிந்திக்கவும்.",
        "பிரச்சினை:",
        "தீர்வு:",
    ),
    Language.TELUGU: (
        "క్రింది గణిత సమస్యను పరిష్కరించండి. తుది సమాధానం చెప్పే ముందు దశలవారీగా ఆలోచించండి.",
        "సమస్య:",
        "పరిష్కారం:",
    ),
    Language.THAI: (
        "จงแก้โจทย์คณิตศาสตร์ต่อไปนี้ คิดทีละขั้นตอนก่อนให้คำตอบสุดท้าย",
        "โจทย์:",
        "วิธีทำ:",
    ),
    Language.URDU: (
        "درج ذیل ریاضی کا مسئلہ حل کریں۔ حتمی جواب دینے سے پہلے مرحلہ وار سوچیں۔",
        "مسئلہ:",
        "حل:",
    ),
    Language.UZBEK: (
        "Quyidagi matematika masalasini yeching. Yakuniy javobni berishdan oldin bosqichma-bosqich o‘ylang.",
        "Masala:",
        "Yechim:",
    ),
    Language.VIETNAMESE: (
        "Giải bài toán sau đây. Hãy suy nghĩ từng bước trước khi đưa ra câu trả lời cuối cùng.",
        "Bài toán:",
        "Lời giải:",
    ),
    Language.WOLOF: (
        "Saafara jafe-jafe xayma bii ci suuf. Xalaatal ndànk-ndànk, jéego bu nekk, bala nga joxe tontu bu mujj bi.",
        "Jafe-jafe:",
        "Saafara:",
    ),
    Language.XHOSA: (
        "Sombulula le ngxaki yezibalo ilandelayo. Cinga inyathelo nenyathelo phambi kokunika impendulo yokugqibela.",
        "Ingxaki:",
        "Isisombululo:",
    ),
    Language.YORUBA: (
        "Yanju iṣoro iṣiro atẹle yii. Ronu ni igbesẹ-nipasẹ-igbesẹ ṣaaju ki o to fun idahun ikẹhin.",
        "Iṣoro:",
        "Ojútùú:",
    ),
    Language.CHINESE: (
        "解答下面的数学题。在给出最终答案前，请一步一步思考。",
        "问题：",
        "解答：",
    ),
    Language.ZULU: (
        "Xazulula inkinga yezibalo elandelayo. Cabanga isinyathelo ngesinyathelo ngaphambi kokunikeza impendulo yokugcina.",
        "Inkinga:",
        "Isixazululo:",
    ),
}


# Extra problem-label variants that models may generate.
# Keep this small: too many stop strings increase accidental truncation risk.
_MGSM_PROBLEM_LABEL_STOP_ALIASES: dict[Language, list[str]] = {
    Language.FRENCH: ["Problème:"],  # no-space variant
    Language.CHINESE: ["问题:"],     # ASCII-colon variant
    Language.JAPANESE: ["問題:"],    # ASCII-colon variant
}


def validate_mgsm_prompt_parts() -> None:
    """Fail fast if a language is missing localized prompt parts."""
    missing_languages = [
        language for language in _LANGUAGES if language not in _MGSM_PROMPT_PARTS
    ]

    if missing_languages:
        missing = ", ".join(language.value for language in missing_languages)
        raise ValueError(f"Missing MGSM prompt parts for: {missing}")


def mgsm_cot_template(language: Language) -> str:
    """Return the localized MGSM CoT template for a language."""
    try:
        instruction, problem_label, solution_label = _MGSM_PROMPT_PARTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Missing MGSM prompt parts for language: {language.value}"
        ) from exc

    return f"{instruction}\n\n{problem_label}\n{{prompt}}\n\n{solution_label}"


def mgsm_stop_sequences(language: Language) -> list[str]:
    """Return stop sequences for a language.

    Stop when the model appears to start a new problem.

    We keep English stops because models sometimes switch back to English even
    when the prompt is localized. We also add only the current language's
    problem label and a few high-value aliases to avoid over-stopping.
    """
    try:
        _, problem_label, _ = _MGSM_PROMPT_PARTS[language]
    except KeyError as exc:
        raise ValueError(
            f"Missing MGSM prompt parts for language: {language.value}"
        ) from exc

    stop_sequences = [
        "Question:",
        "Problem:",
        problem_label,
        *_MGSM_PROBLEM_LABEL_STOP_ALIASES.get(language, []),
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(stop_sequences))


def mgsm_cot_prompt(language: Language):
    """Build a CoT prompt function for MGSM in the given language."""
    template = mgsm_cot_template(language)

    def prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["question"]),
            choices=[line["answer"]],
            gold_index=0,
        )

    return prompt_fn


validate_mgsm_prompt_parts()


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm:{language.value}:gen",
        prompt_function=mgsm_cot_prompt(language),
        hf_repo="CohereLabs/global-mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=1024,
        metrics=[
            Metrics.expr_gold_metric,
            # MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=mgsm_stop_sequences(language),
    )
    for language in _LANGUAGES
]