# Rethinking_of_PAR with python to train, test and export
forked from [Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR)

## Train
```python
python3 train.py --cfg ./configs/pedes_baseline/pa100k1.yaml
```

## Test
```python
python3 demo.py --checkpoint_path ckpt_max_2023-06-21_18\:53\:44.pth --test_img ../attr_recog/images/ --cfg ./configs/pedes_baseline/pa100k1.yaml
```

## Export model to onnx
```python
python3 export_onnx.py
```

# Reference
[1] https://github.com/valencebond/Rethinking_of_PAR
