find "$EXPERIMENT" -type f -name return.json \
| awk -F/ '{print $(NF-2) "," $0}' \
| sort -t, -k1,1 -n \
| cut -d, -f2- \
| xargs -I{} bash -c '
  echo {},$(
    jq -r "
      map(select(.step % 1000000 == 0))
      | max_by(.returns_mean)
      | [.step, .returns_mean]
      | @csv
    " {}
  ),$(
    jq ".[-1].returns_mean" {}
  )
'\
| sort -t, -k3,3 -n \
| cut -d, -f1,2 \
| awk -F'[/,]' '{print $(NF-3)","$(NF)}' > src/foundation_policy/checkpoints_$RL_TOOLS_EXTRACK_EXPERIMENT.txt