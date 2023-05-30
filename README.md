[ResNet](https://arxiv.org/abs/1512.03385) re-implementation using PyTorch

#### Steps

* Configure `imagenet` path by changing `data_dir` in `main.py`
* `python main.py --benchmark` for model information
* `bash ./main.sh $ --train` for training model, `$` is number of GPUs
* `python main.py --test` for testing

### Results

|  Version  | Top@1 | Top@5 | Parameters |                                                                               Download |
|:---------:|------:|------:|-----------:|---------------------------------------------------------------------------------------:|
| ResNet18  |  69.8 |  89.1 |      11.7M |  [model](https://github.com/jahongir7174/ResNet/releases/download/v0.0.1/resnet_18.pt) |
| ResNet34  |  75.1 |  92.3 |      21.8M |  [model](https://github.com/jahongir7174/ResNet/releases/download/v0.0.1/resnet_34.pt) |
| ResNet50  |  80.1 |  94.5 |      25.5M |  [model](https://github.com/jahongir7174/ResNet/releases/download/v0.0.1/resnet_50.pt) |
| ResNet101 |  81.9 |  95.7 |      44.5M | [model](https://github.com/jahongir7174/ResNet/releases/download/v0.0.1/resnet_101.pt) |
| ResNet152 |  82.6 |  96.1 |      60.1M | [model](https://github.com/jahongir7174/ResNet/releases/download/v0.0.1/resnet_152.pt) |
| ResNet200 |     - |     - |      64.6M |                                                                                      - |

#### Note

* The default version is `ResNet50` and weights are ported from `timm==0.6.13`

```
Number of parameters: 25530472
Time per operator type:
        99.0258 ms.    89.2101%. Conv
        4.53153 ms.    4.08236%. Add
        4.13168 ms.    3.72214%. Relu
        2.21462 ms.     1.9951%. MaxPool
        1.07225 ms.   0.965965%. FC
      0.0239915 ms.  0.0216134%. AveragePool
     0.00304198 ms. 0.00274045%. Flatten
        111.003 ms in Total
FLOP per operator type:
        8.17427 GFLOP.    99.8825%. Conv
     0.00551936 GFLOP.  0.0674418%. Add
       0.004097 GFLOP.  0.0500618%. FC
              0 GFLOP.          0%. Relu
        8.18389 GFLOP in Total
Feature Memory Read per operator type:
        136.575 MB.    60.0677%. Conv
        44.1549 MB.    19.4199%. Add
        38.4348 MB.    16.9041%. Relu
        8.20419 MB.    3.60831%. FC
        227.369 MB in Total
Feature Memory Written per operator type:
        44.4559 MB.    42.3502%. Conv
        38.4348 MB.    36.6143%. Relu
        22.0774 MB.    21.0317%. Add
          0.004 MB. 0.00381053%. FC
        104.972 MB in Total
Parameter Memory per operator type:
        93.8196 MB.    91.9659%. Conv
          8.196 MB.    8.03406%. FC
              0 MB.          0%. Add
              0 MB.          0%. Relu
        102.016 MB in Total
```

