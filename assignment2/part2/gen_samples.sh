#!/bin/bash

# checkpoint dir
if [ ! -z $1 ]; then
  checkpoint_dir=$1
else
  checkpoint_dir="./"
fi


# sequence length dir
if [ ! -z $2 ]; then
  seq_len=$2
else
  seq_len="30"
fi

# text file
if [ ! -z $3 ]; then
  txt_file=$3
else
  txt_file="assets/book_EN_grimms_fairy_tails.txt"
fi

source activate dl1

for TEMPERATURE in 0.0 0.5 1.0 2.0; do
  echo Sampling with temperature $TEMPERATURE
  output_file="output/lstm_res_t${TEMPERATURE}_l${seq_len}.txt"
  rm -f $output_file
  python train.py --sample --txt_file $txt_file \
    --checkpoint_dir $checkpoint_dir --temperature $TEMPERATURE \
    --input_seq_length $seq_len > $output_file
  echo Done
done
