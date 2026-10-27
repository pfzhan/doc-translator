# 用法: run_boxes.sh <代码目录> <标签> <公式框标签> [版面框目录]
# 用缓存的公式框 mfd/<id>.<公式框标签>.json 出分段计划并打分，不跑模型；给了版面框目录时再开版面模型（读 <目录>/<id>.json）
code=$1; tag=$2; ftag=$3; . "$(dirname "$0")/../env.sh"
mkdir -p $E/plans $E/results
for d in $E/raw/*/; do
  id=$(basename $d)
  [ -f $E/gold/$id.json ] && [ -f $E/mfd/$id.$ftag.json ] || continue
  [ -f $E/plans/$id.$tag.json ] && continue
  extra=""; [ -n "$4" ] && extra=$4/$id.json
  echo "$P -I $TOOLS/plandump_mfdcache.py $code $d/paper.pdf $E/plans/$id.$tag.json $E/mfd/$id.$ftag.json seg $extra > /dev/null 2>&1"
done | xargs -P 3 -I{} sh -c "{}"
out=$E/results/$tag.txt; : > $out
for g in $E/gold/*.json; do
  id=$(basename $g .json)
  [ -f $E/plans/$id.$tag.json ] || { echo "$id NOPLAN" >> $out; continue; }
  echo "$id $($P -I $TOOLS/score.py $E/plans/$id.$tag.json $g)" >> $out
done
$P -I -c "
import json
from collections import Counter
tot = Counter(segments=0, found=0, split=0, boundaries=0, merged=0)
for line in open('$out'):
    if '{' in line:
        tot.update(json.loads(line.split(' ', 1)[1]))
tot = dict(tot)
print('$tag', tot, 'split%%=%.1f merge%%=%.1f' % (100*tot['split']/max(tot['found'],1), 100*tot['merged']/max(tot['boundaries'],1)))
"
