# 用法: run_cal.sh <名字:方法:页数:seed> ...
# 用指定的校准方法、页数、随机种子把公式检测模型量化成 int8（$W/int8/mfd.<名字>.onnx），再像 run_formula_model.sh 一样评测。
# 方法是 minmax、avg（滑动平均）、percentile、entropy。$W/int8/mfd.<名字>.onnx 已存在时不再量化，直接评测这个文件
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
