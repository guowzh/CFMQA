# CFMQA
Official code for ACL 2023 paper [Counterfactual Multihop QA: A Cause-Effect Approach for Reducing Disconnected Reasoning](https://arxiv.org/abs/2210.07138)

## Preprocess

* please see and run `prepocess.sh` to get the raw data as well as the prepocessd data feature.
* `cd dire_evalute` and run `download_scripts/*.sh` to obtain the probe dataset.
* preprocess the probe dataset to get the data feature. You can refer to the file 'prepocess.sh' in the first step

## Train

```cmd
python train.py --config_file configs/train.bert.json
```

## Predict 
```cmd
python predict.py --config_file configs/predict.bert.json
