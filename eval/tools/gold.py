"""从 LaTeX 源码取段落边界，作标准答案。用法: gold.py <论文目录> <输出 json>
只把源码当文本读。段落在空行、\\par、章节命令、块级环境、\\item 处断开；
段内的行间公式算「不计」的边界（公式两边拆不拆都行）。"""
import json, re, sys, unicodedata
from pathlib import Path

home, out = Path(sys.argv[1]), Path(sys.argv[2])
src = home / "src"

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def strip_comments(text: str) -> str:
    # % 连行尾一起吃掉，下一行的行首空白也忽略：只有注释的一行不会变成空行（段落分隔）。
    # 下一行本来就是空行时照旧分段
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

mains = [p for p in src.rglob("*.tex") if re.search(r"\\begin\{document\}", read(p))]
if not mains:
    sys.exit("no main tex")
main = max(mains, key=lambda p: len(read(p)))
text = expand(main)
body = text.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
# 源码里的空行是段落分隔，记成 \x02。后面去掉命令、公式剩下的空白行不算分段，插进去的断点也用 \x02
body = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n\x02\n", body)
# 自定义的行间公式宏（\be ... \ee、\beq ... \eeq）换回环境
for name, env in re.findall(r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[\d\])?\s*\{\s*\\begin\{(\w+\*?)\}\s*\}", text) + \
        re.findall(r"\\def\\(\w+)\s*\{\s*\\begin\{(\w+\*?)\}\s*\}", text):
    body = re.sub(r"\\" + name + r"(?![a-zA-Z])", lambda m: "\\begin{" + env + "}", body)
for name, env in re.findall(r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[\d\])?\s*\{\s*\\end\{(\w+\*?)\}\s*\}", text) + \
        re.findall(r"\\def\\(\w+)\s*\{\s*\\end\{(\w+\*?)\}\s*\}", text):
    body = re.sub(r"\\" + name + r"(?![a-zA-Z])", lambda m: "\\end{" + env + "}", body)
# 参考文献、附录之后的部分常是表格和长列表，到参考文献为止
body = re.split(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography", body)[0]

# 浮动体夹在段落中间不断段，段落照旧看两边有没有空行。名字带 figure、table 的自定义环境也算
FLOATS = r"\w*figure\*?|\w*table\*?|algorithm\*?|algorithmic|tabular\*?|tikzpicture|lstlisting|verbatim|minted|wrapfigure|subfigure|thebibliography|keywords|IEEEkeywords|CCSXML|acks|acknowledgements|acknowledgments"
body = re.sub(r"\\begin\{(" + FLOATS + r")\}.*?\\end\{\1\}", " \x01 ", body, flags=re.S)
DISPLAY = r"equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|flalign\*?|alignat\*?|split|cases|array"
body = re.sub(r"\\begin\{(" + DISPLAY + r")\}.*?\\end\{\1\}", " \x00MATH\x00 ", body, flags=re.S)
body = re.sub(r"\\\[.*?\\\]", " \x00MATH\x00 ", body, flags=re.S)
body = re.sub(r"\$\$.*?\$\$", " \x00MATH\x00 ", body, flags=re.S)
# 行内公式换成不是空白的占位符：单独占几行的 $…$、\(…\) 去掉后不能留下空行（变成分段）
body = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " \x01 ", body, flags=re.S)
body = re.sub(r"\\\(.*?\\\)", " \x01 ", body, flags=re.S)
# 块级断点
body = re.sub(r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption|title|author|maketitle|item|par|noindent\s*\\textbf)\b\*?", "\n\x02\n\\g<0>", body)
body = re.sub(r"\\(begin|end)\{[^}]*\}", "\n\x02\n", body)
# 脚注、致谢号在句子中间，去掉后句子照旧接着
body = re.sub(r"\\(footnote|thanks|footnotetext)\*?\s*(\[[^\]]*\])?\s*\{", r"\\INLINEDROP{", body)
body = re.sub(r"\\(caption|title|author|affiliation|address|email|institute|part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(\[[^\]]*\])?\s*\{", r"\\DROP{", body)

def drop_braced(text: str, command: str, repl: str = "\n\x02\n") -> str:
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

# 段首的粗体、斜体小标题（\textbf{Ablation.} 正文）是标题，不算段里的字
body = re.sub(r"(\x02\s*(?:\\noindent\s*)?)\\(textbf|textit|emph|textsc|underline)\s*\{([^{}]{1,80})\}(\s*[.:])?", r"\1\\DROP{\3}", body)
body = drop_braced(body, "\\INLINEDROP", " ")
body = drop_braced(body, "\\DROP")
body = re.sub(r"\\(cite[a-z]*|ref|eqref|autoref|cref|Cref|label|url|href|includegraphics|vspace|hspace|footnotemark|bibliographystyle)\*?(\[[^\]]*\])*\{[^}]*\}", " ", body)
body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
body = re.sub(r"[{}~\\]", " ", body)

segments = []
for para in body.split("\x02"):
    pieces = para.split("\x00MATH\x00")
    for index, piece in enumerate(pieces):
        words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", piece).lower())
        words = [w for w in words if len(w) >= 2]
        segments.append({"words": words, "next": "math" if index + 1 < len(pieces) else "par"})
# 太短的段（标题、公式后的半句）对不准，不计分。两个留下的段之间只要有一处是段落边界，就要分开
kept = []
pending_par = False
for seg in segments:
    if len(seg["words"]) >= 12:
        if kept and pending_par:
            kept[-1]["next"] = "par"
        kept.append(dict(seg))
        pending_par = False
        continue
    pending_par = pending_par or seg["next"] == "par" or (kept and kept[-1]["next"] == "par")
    if kept and kept[-1]["next"] == "par":
        pending_par = True
json.dump({"main": str(main.relative_to(src)), "segments": kept}, out.open("w"))
print(home.name, len(kept), "segments")
