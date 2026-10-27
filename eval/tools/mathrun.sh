# 用法: mathrun.sh <代码目录> <标签>   不开模型，出带公式文字的计划（mplans/），再按 mathgold/ 打行内公式的分
code=$1; tag=$2; . "$(dirname "$0")/../env.sh"
mkdir -p $E/mplans
for d in $E/raw/*/; do
  id=$(basename $d); [ -f $E/mathgold/$id.json ] || continue
  [ -f $E/mplans/$id.$tag.json ] && continue
  echo "$P -I $TOOLS/plandump_math.py $code $d/paper.pdf $E/mplans/$id.$tag.json > /dev/null 2>&1"
done | xargs -P 6 -I{} sh -c "{}"
$P -I $TOOLS/mathscore.py $tag
