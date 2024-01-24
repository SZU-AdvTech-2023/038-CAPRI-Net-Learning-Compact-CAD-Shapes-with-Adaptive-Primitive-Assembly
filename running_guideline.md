# Pretrain

1. pretrain model

```
python train.py -e {experiment_name} -g {gpu_id} -p 0
```

## Guideline

1. use initial model to generate finetuned model

```
python fine-tuning.py -e shapenet_voxel -g 1 --test --voxel --start 0 --end 10

python fine-tuning.py -e abc_voxel_transformer_decoder -g 1 --test --voxel --start 0 --end 10
```

2. use finetuned model to reconstruct

```
python test.py -e shapenet_voxel -g 1 -p 0 -c best_stage2 --test --voxel --start 0 --end 10 --mc_threshold 0.5

python test.py -e abc_voxel_transformer_decoder -g 0 -p 0 -c best_stage2 --test --voxel --start 0 --end 10 --mc_threshold 0.5
```

or directly use initial model to reconstruct

```
python test_pretrain.py -e shapenet_voxel -g 1 -p 0 -c initial --test --voxel --start 0 --end 10 --mc_threshold 0.5
```



```
python test_pretrain.py -e shapenet_voxel -g 1 -p 0 -c initial --test --voxel --start 0 --end 10 --mc_threshold 0.5 --csg


4563 4
```

