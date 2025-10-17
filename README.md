FVE on cortical surface area data using Graph Convolutional Network model


Run the following for FVE_GCN_script.py

```
python FVE_GCN_script.py --job_nickname "FVE_GCN_test" --model_type "base" \
  --lr 0.001 --weight_decay 0.0001 --epochs 100 --patience 5  \
  --partial_dat --norm_y --scaler_name "standard" --batch_size 20
```
