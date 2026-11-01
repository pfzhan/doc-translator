# Usage: runplans.sh <code dir> <tag>
# Produce plans for the regression corpus without models, into regress/plans/<file name>.<tag>.json
. "$(dirname "$0")/../env.sh"
mkdir -p $E/regress/plans
regress_pdfs | while read -r f; do
  n=$(basename "$f" .pdf); rm -f $E/regress/plans/$n.$2.json
  $P -I $REG/plandump.py $1 "$f" $E/regress/plans/$n.$2.json 2>$E/regress/plans/$n.$2.err &
done
wait
echo finished
