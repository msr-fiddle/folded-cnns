#!/bin/bash

get_flops() {
  model=$1
  mode=$2
  fold=$3

  if [ $mode == "fold" ]; then
    fold_flag=--do_fold
  else
    fold_flag=''
  fi
  python3 flop_count.py $model $fold_flag --fold $fold > /tmp/tmp.txt
  tail -n 1 /tmp/tmp.txt
}

run() {
  model=$1
  final_comma=$2
  og_ops=$(get_flops $model none 1)
  fold2_ops=$(get_flops $model fold 2)
  fold3_ops=$(get_flops $model fold 3)
  fold4_ops=$(get_flops $model fold 4)
  echo "  \"${model}\": {"
  echo "    \"og_ops\": ${og_ops},"
  echo "    \"fold2_ops\": ${fold2_ops},"
  echo "    \"fold3_ops\": ${fold3_ops},"
  echo "    \"fold4_ops\": ${fold4_ops}"

  if [ ${final_comma} -eq 1 ]; then
    echo "  },"
  else
    echo "  }"
  fi
}

run_fold2() {
  model=$1
  final_comma=$2
  og_ops=$(get_flops $model none 1)
  fold2_ops=$(get_flops $model fold 2)
  echo "  \"${model}\": {"
  echo "    \"og_ops\": ${og_ops},"
  echo "    \"fold2_ops\": ${fold2_ops}"

  if [ ${final_comma} -eq 1 ]; then
    echo "  },"
  else
    echo "  }"
  fi
}


echo "{"
run lol/goldnumber-fraction 1
run apex/squad_count_v2 1
run sot/coin_count-digits_3_4-v2 1
run sot/timer3 1
run noscope_coral 1
run noscope_night 1
run noscope_taipei 1
run noscope_roundabout 1
run_fold2 efficientnet-game_lol/goldnumber-fraction 1
run_fold2 efficientnet-game_apex/squad_count_v2 1
run_fold2 efficientnet-game_sot/coin_count-digits_3_4-v2 1
run_fold2 efficientnet-game_sot/timer3 1
run lol/goldnumber-int 1
run lol/timer-minutes 0
echo "}"
