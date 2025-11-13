#!/usr/bin/env bash
# System-wide CPU% and Memory% sampler (fixed 30s interval).
# Logs CSV: timestamp_iso,cpu_pct,mem_pct
# Run: ./util_monitor.sh   (stops on Ctrl-C)

set -euo pipefail

INTERVAL=30

TS=$(date '+%Y%m%d_%H%M%S')
LOGDIR="$(pwd)/sys_monitor_${TS}"
mkdir -p "$LOGDIR"
MONITOR_CSV="$LOGDIR/monitor.csv"

echo "timestamp_iso,cpu_pct,mem_pct" > "$MONITOR_CSV"
echo "Logging every ${INTERVAL}s to $MONITOR_CSV (press Ctrl-C to stop)"

# read cpu counters -> set globals prev_total prev_idle
read_cpu_counters() {
  # read first line of /proc/stat
  read -r _ user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
  total=$((user + nice + system + idle + iowait + irq + softirq + steal))
  idle_all=$((idle + iowait))
  echo "$total $idle_all"
}

# compute mem percent using MemTotal and MemAvailable
compute_mem_pct() {
  awk '/MemTotal/ {t=$2} /MemAvailable/ {a=$2} END { if (t>0) printf("%.2f", (t - a) / t * 100); else print "0.00" }' /proc/meminfo
}

# initial read
read -r prev_total prev_idle < <(read_cpu_counters)

trap 'echo "Stopping monitor."; exit 0' INT TERM

while true; do
  sleep "$INTERVAL"
  read -r cur_total cur_idle < <(read_cpu_counters)

  dt=$((cur_total - prev_total))
  di=$((cur_idle - prev_idle))

  if [ "$dt" -le 0 ]; then
    cpu_pct="0.00"
  else
    # compute CPU% = 100 * (1 - idle_delta/total_delta)
    cpu_pct=$(awk -v dt="$dt" -v di="$di" 'BEGIN { printf("%.2f", (1 - (di/dt)) * 100) }')
  fi

  mem_pct=$(compute_mem_pct)

  ts_iso=$(date '+%Y-%m-%dT%H:%M:%S%z')
  printf '%s,%s,%s\n' "$ts_iso" "$cpu_pct" "$mem_pct" >> "$MONITOR_CSV"

  prev_total=$cur_total
  prev_idle=$cur_idle
done