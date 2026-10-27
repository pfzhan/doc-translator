# 用法: run_layout_model.sh <onnx> <框目录名> [公式框标签=rect]
# 用一个版面模型给评测集出框到 $EVALSET/<框目录名>/（已有的不重出），再配上缓存的公式框打分段的分，计划标签是 r<框目录名>
. "$(dirname "$0")/../env.sh"
M=$1; D=$E/$2; mkdir -p $D
t0=$(date +%s)
for g in $E/gold/*.json; do id=$(basename $g .json)
  [ -f $D/$id.json ] || echo "$P -I $TOOLS/regions_at.py $CODE $M 1024 3 $E/raw/$id/paper.pdf $D/$id.json"
done | xargs -P 2 -I{} sh -c "{}"
echo "$2 regions $(( $(date +%s) - t0 )) s"
bash $TOOLS/run_boxes.sh $CODE r$2 ${3:-rect} $D
