# FedRefiner

Official code for the paper "FedRefiner: Robust Federated Learning with Noisy Label Refinement for Remote Sensing".

## Installation

```bash
pip install -r requirements.txt
```

## Example Usage

```bash
python FL_train.py --alg fedrefiner --model resnet18 --dataset RS-5 --noise_rate 0.8 --noise_pattern symmetric
```
More settings can be found in `args.py`. Please load the dataset (refer to [FedRS](https://fedrs-bench.github.io/) to download these datasets) into `./dataset`. In `./dataset`, there should be two sub folders `FedRS` and `FedRS-5`.


## Model Scalability

![Model Scalability](./model_scale.png)


## Remote Sensing Imagery
![Remote Sensing Imagery](./raw_rs.png)


## Citation

TBD.

## Acknowledgments
We sincerely appreciate the authors of the FedRS datasets, and more information can be found in the [FedRS](https://fedrs-bench.github.io/).
