# 用法: runplans.sh <代码目录> <标签>   不开模型，出老语料的计划到 regress/plans/<文件名>.<标签>.json
. "$(dirname "$0")/../env.sh"
mkdir -p $E/regress/plans
regress_pdfs | while read -r f; do
  n=$(basename "$f" .pdf); rm -f $E/regress/plans/$n.$2.json
  $P -I $REG/plandump.py $1 "$f" $E/regress/plans/$n.$2.json 2>$E/regress/plans/$n.$2.err &
done
wait
echo finished
