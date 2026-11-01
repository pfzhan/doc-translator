# Usage: run_formula_model.sh <onnx> <tag:side> ...
# Produce boxes for the evaluation set with a formula detection model (mfd/<id>.<tag>.json, existing ones are kept),
# then score inline formulas and segmentation. Plan tags are m<tag> (inline formulas) and s<tag> (segmentation)
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
