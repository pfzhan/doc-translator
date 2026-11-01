# Usage: compare_models.sh <formula|layout> <reference tag>=<onnx> <new tag>=<onnx>
# Compare two files of the same kind of model (such as fp32 and int8) on the regression corpus (the PDFs under
# REGRESS_PDFS): produce boxes and plans with each, then list files whose unit count changed and units whose skip
# flag changed. Cached boxes are reused without inference: formula boxes in regress/mfd/<file name>.<tag>.json,
# layout boxes in regress/regions_<tag>/<file name>.json.
. "$(dirname "$0")/../env.sh"
kind=$1; shift
mkdir -p $E/regress/plans
tags=()
for v in "$@"; do
  tag=${v%%=*}; model=${v#*=}; tags+=("$tag")
  t0=$(date +%s)
  if [ "$kind" = formula ]; then
    mkdir -p $E/regress/mfd
    regress_pdfs | while read -r src; do id=$(basename "$src" .pdf)
      [ -f $E/regress/mfd/$id.$tag.json ] || $P -I $TOOLS/mfd_project.py $CODE $model "$src" $E/regress/mfd/$id.$tag.json < /dev/null
    done
  else
    mkdir -p $E/regress/regions_$tag
    regress_pdfs | while read -r src; do id=$(basename "$src" .pdf)
      [ -f $E/regress/regions_$tag/$id.json ] || echo "$P -I $TOOLS/regions_at.py $CODE $model 1024 3 '$src' $E/regress/regions_$tag/$id.json"
    done | xargs -P 2 -I{} sh -c "{}"
  fi
  echo "$tag boxes $(( $(date +%s) - t0 )) s"
  regress_pdfs | while read -r src; do id=$(basename "$src" .pdf)
    if [ "$kind" = formula ]; then
      echo "$P -I $TOOLS/plandump_mfdcache.py $CODE '$src' $E/regress/plans/$id.c$tag.json $E/regress/mfd/$id.$tag.json seg > /dev/null 2>&1"
    else
      echo "$P -I $REG/plandump_model.py $CODE '$src' $E/regress/regions_$tag/$id.json $E/regress/plans/$id.c$tag.json > /dev/null 2>&1"
    fi
  done | xargs -P 3 -I{} sh -c "{}"
done
$P -I $REG/rowskip.py c${tags[0]} c${tags[1]} 10
echo finished
