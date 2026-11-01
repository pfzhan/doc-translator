# Usage: mathrun_boxes.sh <code dir> <tag> <formula box tag>
# Produce plans with formula text (mplans/) from the cached formula boxes mfd/<id>.<formula box tag>.json, then
# score inline formulas, without running the model
code=$1; tag=$2; ftag=$3; . "$(dirname "$0")/../env.sh"
mkdir -p $E/mplans
for d in $E/raw/*/; do
  id=$(basename $d); [ -f $E/mathgold/$id.json ] && [ -f $E/mfd/$id.$ftag.json ] || continue
  [ -f $E/mplans/$id.$tag.json ] && continue
  echo "$P -I $TOOLS/plandump_mfdcache.py $code $d/paper.pdf $E/mplans/$id.$tag.json $E/mfd/$id.$ftag.json math > /dev/null 2>&1"
done | xargs -P 6 -I{} sh -c "{}"
$P -I $TOOLS/mathscore.py $tag
