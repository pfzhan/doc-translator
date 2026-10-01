"""语言检测，对应沉浸式翻译的 sameLangCheck / 段落语言检测。

- 文档级：抽样检测文档语言。源语言设为“自动检测”时用来填 {{from}}，与目标语言相同时给出提示。
- 段落级：已经是目标语言的段落跳过不翻（例如英文书里引用的中文，翻成中文时就不用再翻）。

用 py3langid（离线、无需网络）。它不区分简繁中文，再用 GB2312 能否编码来判断：
繁体特有的字在 GB2312 里没有。
"""
import re

from py3langid.langid import MODEL_FILE, LanguageIdentifier

_identifier: LanguageIdentifier | None = None

# py3langid 代码 → 插件语言代码
CODE_MAP = {"jv": "jw", "tl": "fil", "nb": "no", "nn": "no", "iw": "he", "zh": "zh-CN"}
# 段落太短时检测不可靠，不做判断（插件里本地检测也要求至少 50 个字符）
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
    # 有一定比例的繁体特有字就当繁体；简繁通用的字不影响判断
    return "zh-TW" if han and traditional / len(han) > 0.05 else "zh-CN"


def detect(text: str, min_letters: int = MIN_LETTERS, min_confidence: float = MIN_CONFIDENCE) -> str | None:
    """返回插件格式的语言代码；文本太短或不确定时返回 None。"""
    letters = len(LETTER_RE.findall(text))
    han = len(HAN_RE.findall(text))
    # 中日韩文字一个字信息量大，按 3 倍算
    if letters + 2 * han < min_letters:
        return None
    # 中日韩先按文字系统判断：统计模型在短中文上容易误判成吴语、粤语
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
    """从文档里均匀抽样一部分段落检测整体语言。"""
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
    """段落语言是否已经是目标语言。简繁互译（插件的 translationLanguagePairs）不算相同。"""
    if not detected:
        return False
    base = target.split("-")[0].lower()
    if detected in ("zh-CN", "zh-TW") or target.startswith("zh") or target in ("yue", "wyw"):
        # 中文各变体（简体、繁体、粤语、文言文、东北话）只在代码完全一致时才算相同
        return detected == target or (target == "zh-HK" and detected == "zh-TW")
    return detected.split("-")[0].lower() == base
