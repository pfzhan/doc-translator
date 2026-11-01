# Usage: run_cal.sh <name:method:pages:seed> ...
# Quantize the formula detection model to int8 ($W/int8/mfd.<name>.onnx) with the given calibration method, page
# count and random seed, then evaluate it as run_formula_model.sh does. Methods are minmax, avg (moving average),
# percentile and entropy. When $W/int8/mfd.<name>.onnx already exists, it is evaluated without quantizing again
. "$(dirname "$0")/../env.sh"
M=$C/data/models/pix2text-mfd-1.5.onnx
mkdir -p $W/int8 $E/mfd
for v in "$@"; do IFS=: read tag method count seed <<< "$v"
  [ -f $W/int8/mfd.$tag.onnx ] || $P -I $MODELS/qcal.py $C $M $W/int8/mfd.$tag.onnx $method $count $seed 2>&1 | grep -v -i warn | tail -2
  t0=$(date +%s)
  for g in $E/gold/*.json; do id=$(basename $g .json)
    [ -f $E/mfd/$id.$tag.json ] || $P -I $TOOLS/mfd_project.py $CODE $W/int8/mfd.$tag.onnx $E/raw/$id/paper.pdf $E/mfd/$id.$tag.json
  done
  echo "$tag boxes $(( $(date +%s) - t0 )) s"
  bash $TOOLS/mathrun_boxes.sh $CODE m$tag $tag | tail -3
  bash $TOOLS/run_boxes.sh $CODE s$tag $tag
done
echo finished
