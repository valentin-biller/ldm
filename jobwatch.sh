#!/usr/bin/env bash

# printf 'jobs=(%s)\n' "$(squeue -u bilv -h -o '%i' | sort -n | xargs)"
# printf 'jobs=(%s)\n' "$(squeue -u ge65mod2 -h -o '%i %j' | awk '$2 ~ /^vb_/ {print $1}' | sort -n | xargs)"

jobs=(90363 91387 91388 91391 91392 91394 91395 91396 91398 91399 91400 91402 91403 91404 91405 91407 91408 91409 91410 91412 91413 91415 91418 91419 91422 91423 91426 91430 91431 91432 91455 91456 91457 91458)
file=""

SLACK_WEBHOOK_URL=
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:?set Slack webhook}

# jobs
declare -A last_state
for job in "${jobs[@]}"; do
  last_state["$job"]=""
done
# file
file_hash_last=""
file_state_last=""
file_check_counter=0
file_check_interval=5

while true; do
  alerts=()
  recoveries=()

  # jobs
  for job in "${jobs[@]}"; do
    state=$(squeue -j "$job" -h -o "%T")
    if [[ -z $state ]]; then
      name=MISSING
    elif [[ $state == RUNNING ]]; then
      name=RUNNING
    else
      name=PENDING
    fi

    if [[ ${last_state["$job"]} != "$name" ]]; then
      if [[ $name == RUNNING ]]; then
        recoveries+=("$job $name")
      elif [[ $name == MISSING ]]; then
        alerts+=("$job $name")
      else
        alerts+=("$job $name")
      fi
      last_state["$job"]="$name"
    fi
  done

  # file
  if [[ -n $file ]]; then
    if (( file_check_counter % file_check_interval == 0 )); then
      file_hash_curr=$(md5sum "$file" 2>/dev/null | awk '{print $1}')
      if [[ -n $file_hash_curr && $file_hash_curr == "$file_hash_last" ]]; then
        file_state="NOT CHANGED"
      elif [[ -n $file_hash_curr && $file_hash_curr != "$file_hash_last" ]]; then
        file_state="CHANGED"
      fi
      file_hash_last="$file_hash_curr"

      if [[ $file_state != "$file_state_last" ]]; then
        if [[ $file_state == "CHANGED" ]]; then
          recoveries+=("FILE $file_state")
        elif [[ $file_state == "NOT CHANGED" ]]; then
          alerts+=("FILE $file_state")
        fi
        file_state_last="$file_state"
      fi
    fi
    file_check_counter=$((file_check_counter + 1))
  fi

  if (( ${#alerts[@]} || ${#recoveries[@]} )); then
    lines=()
    # jobs
    for job in "${jobs[@]}"; do
      lines+=("$job ${last_state[$job]}")
    done
    # file
    if [[ -n $file ]]; then
      lines+=("FILE $file_state")
    fi

    message=$(printf '%s\n' "${lines[@]}")
    printf 'SLURM\n%s\n' "$message"
    message=${message//$'\n'/\\n}
    payload=$(printf '{"text":"SLURM\n%s"}' "$message")
    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null
  fi

  printf 'Sleeping for 60 seconds...\n'
  sleep 60
done

payload=$(printf '{"text":"TMUX : JOBWATCH ERROR"}')
curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null

# ./jobwatch.sh