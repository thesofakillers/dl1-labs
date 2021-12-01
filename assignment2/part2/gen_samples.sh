#!/bin/bash

# checkpoint dir
if [ ! -z $1 ]; then
  checkpoint_dir=$1
else
  checkpoint_dir="./"
fi

# text file
if [ ! -z $2 ]; then
  txt_file=$2
else
  txt_file="assets/book_EN_grimms_fairy_tails.txt"
fi

for TEMPERATURE in 0.0 0.5 1.0 2.0; do
  echo Sampling with temperature $TEMPERATURE
  output_file="lstm_res_t${TEMPERATURE}.txt"
  rm -f $output_file
  python train.py --sample --txt_file $txt_file \
    --checkpoint_dir $checkpoint_dir --temperature $TEMPERATURE > $output_file
  echo Done
done
