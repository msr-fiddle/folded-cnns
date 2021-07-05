#!/bin/bash

NUMARG=1
if [ "$#" -ne "$NUMARG" ]; then
  echo "Usage: $0 <outfile>"
  exit 1
fi

outfile=$1

num_w=100
num_it=10000
num_trials=10
fp16=1
do_trt=1

inference_script=inference.py
flops_script=util/flops_from_tput.py

header_str="Model,Trial,Batch Size,Mode,Fold,Throughput,FLOPs/sec,TputNorm,FLOPNorm"
echo $header_str
echo $header_str >> $outfile
depth_mult=1.2
width_mult=1.1
res_mult=1.15
for trial in $(seq $num_trials); do
  for batch_size in 1024; do
    for prefix in game noscope efficientnet-game; do
      if [ $prefix == "noscope" ]; then
        cfg_list="coral night taipei roundabout"
      elif [ $prefix == "efficientnet-game" ]; then
        cfg_list="lol/goldnumber-fraction apex/squad_count_v2 sot/coin_count-digits_3_4-v2 sot/timer3"
      elif [ $prefix == "game" ]; then
        cfg_list="lol/goldnumber-fraction apex/squad_count_v2 sot/coin_count-digits_3_4-v2 sot/timer3 lol/goldnumber-int lol/timer-minutes"
      else
        echo "Unrecognized prefix '${prefix}'"
        exit 1
      fi

      for cfg in $cfg_list; do
        # Get original model results
        model=${prefix}_${cfg}
        og_out=$(python3 $inference_script $model 1 \
          --num_warmup $num_w --num_iteration $num_it \
          --batch_size $batch_size --trt $do_trt)
        og_flops=$(python3 $flops_script $og_out $model og)
        result_str="${cfg},${trial},${batch_size},Original,1,${og_out},${og_flops},1,1"
        echo $result_str
        echo $result_str >> $outfile

        for fold in 2 3 4; do
          # Only run fold 2 for EfficientNet
          if [ $fold -gt 2 ] && [ $prefix == "efficientnet-game" ]; then
            continue
          fi

          # Do FoldedCNN batch size reduction, if running a FoldedCNN
          if [ $prefix == "efficientnet-game" ]; then
            title=EfficientNet
            fold_batch_size=$batch_size
          else
            title=Folded
            fold_batch_size=$(($batch_size / $fold))
          fi

          fold_out=$(python3 $inference_script $model $fold --do_fold \
            --num_warmup $num_w --num_iteration $num_it --batch_size $fold_batch_size \
            --depth_mult $depth_mult --width_mult $width_mult --res_mult $res_mult --trt $do_trt)
          fold_flops=$(python3 $flops_script $fold_out $model fold${fold})
          norm_tput=$(python3 -c "print('{:.2f}'.format(${fold_out} / ${og_out}))")
          norm_flops=$(python3 -c "print('{:.2f}'.format(${fold_flops} / ${og_flops}))")
          result_str="${cfg},${trial},${batch_size},${title},${fold},${fold_out},${fold_flops},${norm_tput},${norm_flops}"
          echo $result_str
          echo $result_str >> $outfile
        done # fold
      done # model
    done # prefix (game, noscope)
  done # batch_size
done # trials
