"""从 LaTeX 源码取行内公式，作标准答案。用法: mathgold.py <论文目录> <输出 json>
只把源码当文本读。每个 $…$、\\(…\\) 记下公式源码和两边各几个正文词（锚点），
评分时按锚点在计划里找到公式落在哪，看它有没有做成占位符。预处理和 gold.py 一致。"""
import json, re, sys, unicodedata
from pathlib import Path

home, out = Path(sys.argv[1]), Path(sys.argv[2])
src = home / "src"
# 锚点每边取几个词；少于 MIN 个的找不准，不计
CONTEXT, MIN = 4, 2


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*\n(?![ \t]*\n)[ \t]*", "", text)
    return re.sub(r"(?<!\\)%.*", "", text)


def expand(path: Path, depth: int = 0) -> str:
    text = strip_comments(read(path))
    if depth > 6:
        return text

    def sub(match: re.Match) -> str:
        name = match.group(2).strip()
        for candidate in (name, name + ".tex"):
            target = (path.parent / candidate).resolve()
            if not target.is_file():
                target = (src / candidate).resolve()
            if target.is_file() and src.resolve() in target.parents:
                return "\n\n" + expand(target, depth + 1) + "\n\n"
        return ""
    return re.sub(r"\\(input|include|subfile)\s*\{([^}]*)\}", sub, text)


def drop_braced(text: str, command: str, repl: str = " ") -> str:
    out, i = [], 0
    while True:
        j = text.find(command + "{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        depth, k = 0, j + len(command)
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        out.append(repl)
        i = k + 1


def words_of(text: str) -> list[str]:
    found = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())
    return found


def trivial(math: str) -> bool:
    """只有数字、百分号、正负号的公式（$10\\%$、$-1$）排出来和正文一样，不计"""
    bare = re.sub(r"\\[,;:! ]|\\%|\\textrm|\\mathrm|[{}\s~]", "", math)
    return re.fullmatch(r"[-+−]?[\d.,]+%?", bare) is not None


mains = [p for p in src.rglob("*.tex") if re.search(r"\\begin\{document\}", read(p))]
if not mains:
    sys.exit("no main tex")
main = max(mains, key=lambda p: len(read(p)))
text = expand(main)
body = text.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
body = re.split(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography", body)[0]
FLOATS = r"\w*figure\*?|\w*table\*?|algorithm\*?|algorithmic|tabular\*?|tikzpicture|lstlisting|verbatim|minted|wrapfigure|subfigure|thebibliography|keywords|IEEEkeywords|CCSXML|acks|acknowledgements|acknowledgments"
body = re.sub(r"\\begin\{(" + FLOATS + r")\}.*?\\end\{\1\}", " \x02 ", body, flags=re.S)
DISPLAY = r"equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|flalign\*?|alignat\*?|split|cases|array"
body = re.sub(r"\\begin\{(" + DISPLAY + r")\}.*?\\end\{\1\}", " \x02 ", body, flags=re.S)
body = re.sub(r"\\\[.*?\\\]", " \x02 ", body, flags=re.S)
body = re.sub(r"\$\$.*?\$\$", " \x02 ", body, flags=re.S)
# 行内公式抽出来，正文里留 \x00序号\x00
maths: list[str] = []


def keep(match: re.Match) -> str:
    maths.append(match.group(1))
    return f" \x00{len(maths) - 1}\x00 "


body = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", keep, body, flags=re.S)
body = re.sub(r"\\\((.+?)\\\)", keep, body, flags=re.S)
# 段落、标题、脚注处断开：锚点不跨过这些地方
body = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", " \x02 ", body)
body = re.sub(r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption|title|author|maketitle|item|par)\b\*?", " \x02 ", body)
body = re.sub(r"\\(begin|end)\{[^}]*\}", " \x02 ", body)
body = re.sub(r"\\(footnote|thanks|footnotetext)\*?\s*(\[[^\]]*\])?\s*\{", r"\\FOOTDROP{", body)
body = drop_braced(body, "\\FOOTDROP", " \x02 ")
body = re.sub(r"\\(cite[a-z]*|ref|eqref|autoref|cref|Cref|label|url|href|includegraphics|vspace|hspace|footnotemark)\*?(\[[^\]]*\])*\{[^}]*\}", " \x03 ", body)
body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
body = re.sub(r"[{}~\\]", " ", body)

# 词和公式排成一串：("w", 词) / ("m", 序号) / ("x", 断开)。引用、交叉引用排出来是编号，也断开锚点
stream: list[tuple[str, object]] = []
for piece in re.split(r"(\x00\d+\x00|\x02|\x03)", body):
    if not piece:
        continue
    if piece.startswith("\x00"):
        stream.append(("m", int(piece.strip("\x00"))))
    elif piece in ("\x02", "\x03"):
        stream.append(("x", None))
    else:
        stream.extend(("w", w) for w in words_of(piece))

items = []
for pos, (kind, value) in enumerate(stream):
    if kind != "m":
        continue
    left: list[str] = []
    k = pos - 1
    while k >= 0 and stream[k][0] == "w" and len(left) < CONTEXT:
        left.insert(0, stream[k][1])
        k -= 1
    right: list[str] = []
    k = pos + 1
    while k < len(stream) and stream[k][0] == "w" and len(right) < CONTEXT:
        right.append(stream[k][1])
        k += 1
    if len(left) < MIN or len(right) < MIN:
        continue
    items.append({"src": maths[value].strip(), "left": left, "right": right, "trivial": trivial(maths[value])})
json.dump({"main": str(main.relative_to(src)), "inline": len(maths), "items": items}, out.open("w"), ensure_ascii=False)
print(home.name, len(maths), "inline,", len(items), "anchored,", sum(1 for i in items if not i["trivial"]), "scored")
