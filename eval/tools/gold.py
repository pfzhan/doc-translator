"""Take paragraph boundaries from the LaTeX source as the gold standard. Usage: gold.py <paper dir> <output json>
The source is read as text only. Paragraphs break at blank lines, \\par, sectioning commands, block environments
and \\item; a display formula inside a paragraph is an ignored boundary (splitting or not at the formula are both
accepted)."""

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import TypedDict

home, out = Path(sys.argv[1]), Path(sys.argv[2])
src = home / "src"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    # % consumes the line end too, and leading whitespace of the next line is ignored, so a line holding only a
    # comment does not become a blank line (a paragraph break). When the next line is blank, the paragraph still
    # breaks
    text = re.sub(r"(?<!\\)%.*\n(?![ \t]*\n)[ \t]*", "", text)
    return re.sub(r"(?<!\\)%.*", "", text)


def expand(path: Path, depth: int = 0) -> str:
    text = strip_comments(read(path))
    if depth > 6:
        return text

    def sub(match: re.Match[str]) -> str:
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
# Blank lines in the source separate paragraphs and are recorded as \x02. Blank lines left after removing commands
# and formulas do not break paragraphs; inserted breaks also use \x02
body = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n\x02\n", body)
# Custom display formula macros (\be ... \ee, \beq ... \eeq) are turned back into environments
for name, env in re.findall(
    r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[\d\])?\s*\{\s*\\begin\{(\w+\*?)\}\s*\}", text
) + re.findall(r"\\def\\(\w+)\s*\{\s*\\begin\{(\w+\*?)\}\s*\}", text):
    body = re.sub(r"\\" + name + r"(?![a-zA-Z])", lambda m: "\\begin{" + env + "}", body)
for name, env in re.findall(
    r"\\(?:re)?newcommand\*?\s*\{?\\(\w+)\}?\s*(?:\[\d\])?\s*\{\s*\\end\{(\w+\*?)\}\s*\}", text
) + re.findall(r"\\def\\(\w+)\s*\{\s*\\end\{(\w+\*?)\}\s*\}", text):
    body = re.sub(r"\\" + name + r"(?![a-zA-Z])", lambda m: "\\end{" + env + "}", body)
# What follows the bibliography and appendices is often tables and long lists; stop at the bibliography
body = re.split(r"\\bibliography\{|\\begin\{thebibliography\}|\\printbibliography", body)[0]

# A float inside a paragraph does not break it; the paragraph still breaks only on blank lines around it. Custom
# environments with figure or table in their names count too
FLOATS = (
    r"\w*figure\*?|\w*table\*?|algorithm\*?|algorithmic|tabular\*?|tikzpicture|lstlisting|verbatim|minted|wrapfigure"
    r"|subfigure|thebibliography|keywords|IEEEkeywords|CCSXML|acks|acknowledgements|acknowledgments"
)
body = re.sub(r"\\begin\{(" + FLOATS + r")\}.*?\\end\{\1\}", " \x01 ", body, flags=re.S)
DISPLAY = r"equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath|flalign\*?|alignat\*?|split|cases|array"
body = re.sub(r"\\begin\{(" + DISPLAY + r")\}.*?\\end\{\1\}", " \x00MATH\x00 ", body, flags=re.S)
body = re.sub(r"\\\[.*?\\\]", " \x00MATH\x00 ", body, flags=re.S)
body = re.sub(r"\$\$.*?\$\$", " \x00MATH\x00 ", body, flags=re.S)
# Inline formulas are replaced by a non-blank placeholder: removing $…$ or \(…\) spanning lines of their own must
# not leave a blank line (which would break the paragraph)
body = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " \x01 ", body, flags=re.S)
body = re.sub(r"\\\(.*?\\\)", " \x01 ", body, flags=re.S)
# Block-level breaks
body = re.sub(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph|caption|title|author|maketitle|item|par|noindent\s*\\textbf)\b\*?",
    "\n\x02\n\\g<0>",
    body,
)
body = re.sub(r"\\(begin|end)\{[^}]*\}", "\n\x02\n", body)
# Footnotes and acknowledgement marks sit inside sentences, which continue after they are removed
body = re.sub(r"\\(footnote|thanks|footnotetext)\*?\s*(\[[^\]]*\])?\s*\{", r"\\INLINEDROP{", body)
body = re.sub(
    r"\\(caption|title|author|affiliation|address|email|institute|part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*(\[[^\]]*\])?\s*\{",
    r"\\DROP{",
    body,
)


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


# A bold or italic run-in heading at the paragraph start (\textbf{Ablation.} text) is a heading, not paragraph text
body = re.sub(
    r"(\x02\s*(?:\\noindent\s*)?)\\(textbf|textit|emph|textsc|underline)\s*\{([^{}]{1,80})\}(\s*[.:])?",
    r"\1\\DROP{\3}",
    body,
)
body = drop_braced(body, "\\INLINEDROP", " ")
body = drop_braced(body, "\\DROP")
body = re.sub(
    r"\\(cite[a-z]*|ref|eqref|autoref|cref|Cref|label|url|href|includegraphics|vspace|hspace|footnotemark|bibliographystyle)\*?(\[[^\]]*\])*\{[^}]*\}",
    " ",
    body,
)
body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
body = re.sub(r"[{}~\\]", " ", body)


class Segment(TypedDict):
    words: list[str]
    next: str  # what ends the segment: "par" (paragraph boundary) or "math" (display formula)


segments: list[Segment] = []
for para in body.split("\x02"):
    pieces = para.split("\x00MATH\x00")
    for index, piece in enumerate(pieces):
        words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", piece).lower())
        words = [w for w in words if len(w) >= 2]
        segments.append({"words": words, "next": "math" if index + 1 < len(pieces) else "par"})
# Very short paragraphs (headings, half sentences after a formula) cannot be located reliably and are not scored.
# Two remaining paragraphs must be separate when there is a paragraph boundary anywhere between them
kept: list[Segment] = []
pending_par = False
for seg in segments:
    if len(seg["words"]) >= 12:
        if kept and pending_par:
            kept[-1]["next"] = "par"
        kept.append(seg.copy())
        pending_par = False
        continue
    pending_par = pending_par or seg["next"] == "par" or (bool(kept) and kept[-1]["next"] == "par")
    if kept and kept[-1]["next"] == "par":
        pending_par = True
json.dump({"main": str(main.relative_to(src)), "segments": kept}, out.open("w"))
print(home.name, len(kept), "segments")
