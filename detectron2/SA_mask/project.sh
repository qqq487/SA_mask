id=$1
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

echo $free_mem

while true; do
  while [ $free_mem -lt 40000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
    echo $free_mem
    sleep 1

  done

python train_net.py --config-file "./configs/sa_mask/SA_mask_rcnn_R_50_FPN_3x.yaml" --num-gpus 1

done


