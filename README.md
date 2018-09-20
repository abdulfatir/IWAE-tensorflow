# Importance Weighted Auto Encoder
A tensorflow implementation of _Importance Weighted Auto Encoder_ <sup>[1]</sup>

## Requirements

* tensorflow
* numpy
* matplotlib

## Usage

`python main.py -k 5 --test_k 5000 --n_steps 200000 --batch_size 100`

## Results
The following are the log-likelihood values after training for 200,000 steps with a batch size of 100 for different number of particles (`k`) and the number of test particles `test_k = 5000`.

|k| NLL (MNIST) |
|:----:|:----:|
| 1  | 91.65 |
| 5  | 89.93 |
| 50 | 88.51 |

### References
[1] Burda, Y., Grosse, R. and Salakhutdinov, R., 2015. Importance Weighted Autoencoders. arXiv preprint arXiv:1509.00519.

