# Importance Weighted Auto Encoder
A tensorflow implementation of _Importance Weighted Auto Encoder_ <sup>[1]</sup>

## Requirements

* tensorflow
* numpy
* matplotlib

## Usage
```
python main.py  --dataset {mnist,omniglot} \
                --k <# of particles for training> \
                --test_k <# number of particles for testing> \
                --n_steps <# of steps> \
                --batch_size <batch size>
```

### Datasets

* MNIST - automatically downloaded by tensorflow
* OMNIGLOT - run `download_omniglot.sh`

## Results
The following are the log-likelihood values after training for 400,000 steps with a batch size of 100 for different number of particles (`k`) and `test_k = 5000`.

|k| NLL (MNIST) | NLL (OMNIGLOT) |
|:----:|:----:|:----:|
| 1  | 90.26 | 114.68 |
| 5  | 88.49 | 112.25 |
| 50 | 87.34 | 110.31 |

### References
[1] Burda, Y., Grosse, R. and Salakhutdinov, R., 2015. Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.

