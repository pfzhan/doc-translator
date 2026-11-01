"""Language detection, matching sameLangCheck and paragraph language detection of Immersive Translate.

- Document level: detect the document language from a sample. Fills {{from}} when the source language is set to
  "自动检测", and warns when it equals the target language.
- Paragraph level: paragraphs already in the target language are not translated (Chinese quoted in an English
  book needs no translation into Chinese).

Uses py3langid (offline, no network). It does not tell Simplified from Traditional Chinese, which is decided by
whether the text can be encoded in GB2312: characters specific to Traditional Chinese are not in GB2312.
"""

import re

from py3langid.langid import MODEL_FILE, LanguageIdentifier

_identifier: LanguageIdentifier | None = None

# py3langid code → language code of the extension
CODE_MAP = {"jv": "jw", "tl": "fil", "nb": "no", "nn": "no", "iw": "he", "zh": "zh-CN"}
# Detection is unreliable on very short paragraphs, which are not judged (local detection in the extension also
# requires at least 50 characters)
MIN_LETTERS = 20
MIN_CONFIDENCE = 0.9

HAN_RE = re.compile(r"[一-鿿]")
KANA_RE = re.compile(r"[぀-ヿ]")
HANGUL_RE = re.compile(r"[가-힯]")
LETTER_RE = re.compile(r"[^\W\d_]")


def _get_identifier() -> LanguageIdentifier:
    global _identifier
    if _identifier is None:
        _identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    return _identifier


def _chinese_variant(text: str) -> str:
    han = HAN_RE.findall(text)
    traditional = 0
    for ch in han:
        try:
            ch.encode("gb2312")
        except UnicodeEncodeError:
            traditional += 1
    # A certain share of characters specific to Traditional Chinese makes the text Traditional; characters shared
    # by both do not count
    return "zh-TW" if han and traditional / len(han) > 0.05 else "zh-CN"


def detect(text: str, min_letters: int = MIN_LETTERS, min_confidence: float = MIN_CONFIDENCE) -> str | None:
    """Return a language code in the extension's format; None when the text is too short or the result uncertain."""
    letters = len(LETTER_RE.findall(text))
    han = len(HAN_RE.findall(text))
    # A CJK character carries more information and counts three times
    if letters + 2 * han < min_letters:
        return None
    # CJK is decided by script first: on short Chinese text the statistical model often guesses Wu or Cantonese
    kana = len(KANA_RE.findall(text))
    hangul = len(HANGUL_RE.findall(text))
    if kana and kana + han > letters * 0.5:
        return "ja"
    if hangul > letters * 0.5:
        return "ko"
    if han > letters * 0.6:
        return _chinese_variant(text)
    lang, conf = _get_identifier().classify(text)
    if conf < min_confidence:
        return None
    code = CODE_MAP.get(lang, lang)
    return _chinese_variant(text) if code == "zh-CN" else code


def detect_document(texts: list[str], sample_chars: int = 4000) -> str | None:
    """Detect the overall language from paragraphs sampled evenly across the document."""
    candidates = [t for t in texts if len(t) >= 20] or texts
    if not candidates:
        return None
    step = max(1, len(candidates) // 40)
    sample, size = [], 0
    for t in candidates[::step]:
        sample.append(t)
        size += len(t)
        if size >= sample_chars:
            break
    return detect("\n".join(sample), min_confidence=0.5)


def same_language(detected: str | None, target: str) -> bool:
    """Whether the paragraph is already in the target language. Simplified and Traditional Chinese translated into
    each other (translationLanguagePairs in the extension) do not count as the same."""
    if not detected:
        return False
    base = target.split("-")[0].lower()
    if detected in ("zh-CN", "zh-TW") or target.startswith("zh") or target in ("yue", "wyw"):
        # Chinese variants (Simplified, Traditional, Cantonese, Classical Chinese, Northeastern Mandarin) count as
        # the same only when the codes are identical
        return detected == target or (target == "zh-HK" and detected == "zh-TW")
    return detected.split("-")[0].lower() == base
