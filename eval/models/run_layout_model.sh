# Usage: run_layout_model.sh <onnx> <box dir name> [formula box tag=rect]
# Produce boxes for the evaluation set with a layout model into $EVALSET/<box dir name>/ (existing ones are kept),
# then score segmentation together with the cached formula boxes. The plan tag is r<box dir name>
. "$(dirname "$0")/../env.sh"
M=$1; D=$E/$2; mkdir -p $D
t0=$(date +%s)
for g in $E/gold/*.json; do id=$(basename $g .json)
  [ -f $D/$id.json ] || echo "$P -I $TOOLS/regions_at.py $CODE $M 1024 3 $E/raw/$id/paper.pdf $D/$id.json"
done | xargs -P 2 -I{} sh -c "{}"
echo "$2 regions $(( $(date +%s) - t0 )) s"
bash $TOOLS/run_boxes.sh $CODE r$2 ${3:-rect} $D
