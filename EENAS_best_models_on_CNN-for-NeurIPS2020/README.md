# This code is for reproducibility of NeurIPS2020 submission "Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement", the searched best models on CNN with common DARTS search space.


## Requirements
```
Python == 3.6.5, PyTorch == 1.0.1.post2
```


## Pretrained models

### Test on CIFAR10
#### model with 600 training epochs

```
cd CNN && python test.py --auxiliary --model_path ./trained_models/EENAS_CIFAR10_600.pt
```
* Expected result: 2.53% test error rate with 3.32M model params.


#### model with 1500 training epochs
```
cd CNN && python test.py --auxiliary --model_path ./trained_models/EENAS_CIFAR10_1500.pt

```
* Expected result: 2.49% test error rate with 3.32M  model params.



#### model with 50 initial filters and 1500 training epochs
```
cd CNN && python test.py --auxiliary --init_channels 50 --model_path ./trained_models/EENAS_CIFAR10_50_1500.pt

```
* Expected result: 2.40% test error rate with 6.26M  model params.




### Test on CIFAR100

#### model with 600 training epochs
```
cd CNN && python test_CIFAR100.py --auxiliary --model_path ./trained_models/EENAS_CIFAR100.pt

```
* Expected result: 16.99% test error rate with 3.37M  model params.



#### model with 50 initial filters
```
cd CNN && python test_CIFAR100.py --auxiliary --model_path ./trained_models/EENAS_CIFAR100_50.pt

```
* Expected result: 16.45% test error rate with 3.37M  model params.



#### model with 50 initial filters and 1500 training epochs 
```
cd CNN && python test_CIFAR100.py --auxiliary --init_channels 50   --model_path ./trained_models/EENAS_CIFAR100_50_1500.pt

```
* Expected result: 15.71% test error rate with 6.33M  model params.


### Test on IMAGENET
```
cd CNN && python test_imagenet.py --auxiliary --model_path ./trained_models/EENAS_IMAGENET/model_best.pth.tar
```
* Expected result: 26.36% test error rate with 4.69M  model params.



## Architecture evaluation (using full-sized models)
To evaluate our best cells by training from scratch, run

### CIFAR-10
```
cd CNN && python train.py --auxiliary --cutout  
```
### CIFAR-100
```
cd CNN && python train_CIFAR100.py --auxiliary --cutout           
```
### IMAGENET
```
cd CNN && python train_imagenet.py --auxiliary  
```
               



## New searched high-performance models. All results and training log files could be found in ./CNN/trained_models/EENAS_C_XXX.
### Test on CIFAR10, model with 600 training epochs
```
cd CNN && python test.py --auxiliary --model_path ./trained_models/EENAS_C_cifar10/weights.pt  --arch EENAS_C
```
* Expected result: 2.50% test error rate with 3.86M model params.

### Test on CIFAR100, model with 600 training epochs
```
cd CNN && python test_CIFAR100.py --auxiliary --model_path ./trained_models/EENAS_C_cifar100/weights.pt  --arch EENAS_C
```
* Expected result: 15.84% test error rate with 3.91M model params.

### Test on IMAGENET, with 250 training epochs and 48 initial filters
```
cd CNN && python test_imagenet_multi_gpu.py --auxiliary --model_path ./trained_models/EENAS_C_imagenet/model_best.pth.tar  --arch EENAS_C
```
* Expected result: 24.37% test error rate with 5.37M  model params. this model is trained with two GPUs with batchsize 256, and initial learning rate 0.2. Comparison results with peer algorithms could be found in  ./Comparision_results_on_DARTS_search_space.jpg

