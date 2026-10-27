# 用法: run_formula_model.sh <onnx> <标签:边长> ...
# 用一个公式检测模型给评测集出框（mfd/<id>.<标签>.json，已有的不重出），再打行内公式和分段的分。
# 计划标签是 m<标签>（行内公式）和 s<标签>（分段）
. "$(dirname "$0")/../env.sh"
M=$1; shift
mkdir -p $E/mfd
for v in "$@"; do tag=${v%%:*}; size=${v##*:}
  t0=$(date +%s)
  for g in $E/gold/*.json; do id=$(basename $g .json)
    [ -f $E/mfd/$id.$tag.json ] || $P -I $TOOLS/mfd_project.py $CODE $M $E/raw/$id/paper.pdf $E/mfd/$id.$tag.json $size
  done
  echo "$tag boxes $(( $(date +%s) - t0 )) s"
  bash $TOOLS/mathrun_boxes.sh $CODE m$tag $tag | tail -3
  bash $TOOLS/run_boxes.sh $CODE s$tag $tag
done
echo finished
