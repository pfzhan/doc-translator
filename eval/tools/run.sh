# 用法: run.sh <代码目录> <标签>   对 raw/ 下每篇出分段计划并打分，汇总到 results/<标签>.txt
code=$1; tag=$2; . "$(dirname "$0")/../env.sh"
mkdir -p $E/plans $E/gold $E/results
for d in $E/raw/*/; do
  id=$(basename $d)
  [ -f $E/gold/$id.json ] || $P -I $TOOLS/gold.py $d $E/gold/$id.json > /dev/null 2>&1
  [ -f $E/gold/$id.json ] || continue
  [ -f $E/plans/$id.$tag.json ] && continue
  echo "$P -I $TOOLS/plandump2.py $code $d/paper.pdf $E/plans/$id.$tag.json > /dev/null 2>&1"
done | xargs -P 6 -I{} sh -c "{}"
out=$E/results/$tag.txt; : > $out
for g in $E/gold/*.json; do
  id=$(basename $g .json)
  [ -f $E/plans/$id.$tag.json ] || { echo "$id NOPLAN" >> $out; continue; }
  echo "$id $($P -I $TOOLS/score.py $E/plans/$id.$tag.json $g)" >> $out
done
$P -I -c "
import json, sys
from collections import Counter
tot = Counter(segments=0, found=0, split=0, boundaries=0, merged=0)
for line in open('$out'):
    if '{' in line:
        tot.update(json.loads(line.split(' ', 1)[1]))
tot = dict(tot)
print('$tag', tot, 'split%%=%.1f merge%%=%.1f' % (100*tot['split']/max(tot['found'],1), 100*tot['merged']/max(tot['boundaries'],1)))
"
