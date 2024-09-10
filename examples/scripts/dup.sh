#!/bin/bash

# 输入文件名
input_file=$1
# 输出文件名
output_file=$2

# 清空输出文件
> $output_file

# 读取输入文件的每一行并复制5次 
while IFS= read -r line
do
  for i in {1..5}
  do  
    echo "$line" >> $output_file
  done
done < "$input_file"

