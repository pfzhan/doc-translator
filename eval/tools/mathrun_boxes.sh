# 用法: mathrun_boxes.sh <代码目录> <标签> <公式框标签>
# 用缓存的公式框 mfd/<id>.<公式框标签>.json 出带公式文字的计划（mplans/），再打行内公式的分，不跑模型
code=$1; tag=$2; ftag=$3; . "$(dirname "$0")/../env.sh"
mkdir -p $E/mplans
for d in $E/raw/*/; do
  id=$(basename $d); [ -f $E/mathgold/$id.json ] && [ -f $E/mfd/$id.$ftag.json ] || continue
  [ -f $E/mplans/$id.$tag.json ] && continue
  echo "$P -I $TOOLS/plandump_mfdcache.py $code $d/paper.pdf $E/mplans/$id.$tag.json $E/mfd/$id.$ftag.json math > /dev/null 2>&1"
done | xargs -P 6 -I{} sh -c "{}"
$P -I $TOOLS/mathscore.py $tag
