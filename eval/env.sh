# Paths shared by the evaluation scripts. Each script sources this with . "$(dirname "$0")/../env.sh"; every value
# can be overridden by an environment variable of the same name.
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)    # project directory
P=${PY:-$C/.venv/bin/python}                           # Python with the project dependencies
E=${EVALSET:-$HOME/doc-translator-evalset}             # evaluation data: papers, gold standards, plans, box caches
W=${WORK:-$E/work}                                     # experiment outputs: int8 models and so on
REGRESS_PDFS=${REGRESS_PDFS:-$C/data/jobs}             # regression corpus PDFs, flat or one dir each
CODE=${CODE:-$C}                                       # code used for plans and boxes; may be a copy of a commit
TOOLS=$C/eval/tools; REG=$C/eval/regress; MODELS=$C/eval/models

# The PDF named $1 in the regression corpus
regress_pdf() {
  ls "$REGRESS_PDFS/$1.pdf" "$REGRESS_PDFS"/*/"$1.pdf" 2>/dev/null | head -1
}

# All PDFs of the regression corpus (skipping bilingual and translated outputs)
regress_pdfs() {
  ls "$REGRESS_PDFS"/*.pdf "$REGRESS_PDFS"/*/*.pdf 2>/dev/null | grep -v -e '\.bilingual\.pdf$' -e '\.translated\.pdf$'
}
