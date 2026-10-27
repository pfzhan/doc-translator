# 评测脚本共用的路径。各脚本用 . "$(dirname "$0")/../env.sh" 引入，可以用同名环境变量覆盖。
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)    # 项目目录
P=${PY:-$C/.venv/bin/python}                           # 带项目依赖的 Python
E=${EVALSET:-$HOME/doc-translator-evalset}             # 评测数据：论文、标准答案、计划、框缓存
W=${WORK:-$E/work}                                     # 实验产物：int8 模型等
REGRESS_PDFS=${REGRESS_PDFS:-$C/data/jobs}             # 老语料 PDF 所在目录，可以是平铺的，也可以是 data/jobs 那样一份一个子目录
CODE=${CODE:-$C}                                       # 出计划、出框用的代码目录，可以指向某个提交的副本
TOOLS=$C/eval/tools; REG=$C/eval/regress; MODELS=$C/eval/models

# 老语料里名为 $1 的 PDF
regress_pdf() {
  ls "$REGRESS_PDFS/$1.pdf" "$REGRESS_PDFS"/*/"$1.pdf" 2>/dev/null | head -1
}

# 老语料的全部 PDF（跳过翻译输出的双语版、译文版）
regress_pdfs() {
  ls "$REGRESS_PDFS"/*.pdf "$REGRESS_PDFS"/*/*.pdf 2>/dev/null | grep -v -e '\.bilingual\.pdf$' -e '\.translated\.pdf$'
}
