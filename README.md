## Code for Generative Text Modeling through Short Run Inference


## Pretrained Models
Pretrained models are available in ```cache/ckpt```.

We also provide cached posterior z samples in ```cache```. They can be used to evaluate the clustering and semi-supervised classification performance.

## Run Trained Models
```
python train_sri.py

python unsupervised_cluster.py

python semi_sup_classification.py

```


## Train New Models
```
python train_sri.py --eval_only False
```

## Contact
Feel free to contact Bo Pang (bopang@ucla.edu) or Erik Nijkamp (enijkamp@ucla.edu) for questions.
