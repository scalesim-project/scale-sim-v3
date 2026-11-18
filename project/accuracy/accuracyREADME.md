# Accuracy

Accuracy testing for ResNet-18 using baseline, PoT, APoT, and Mitchell multiplication.

Run commands from the repository root with Python 3.11.

## Install

```bash
python -m pip install -r project/accuracy/requirements.txt
```

## Test

```bash
python -m unittest project.accuracy.test_shiftmac -v
```

## Run

Show available options:

```bash
python -m project.accuracy.run --help
```

Run an accuracy experiment with a labeled ImageNet validation dataset and matching class map:

```bash
python -m project.accuracy.run --data path/to/imagenet-val --class-map path/to/class-map.json --pes baseline pot apot --max-images 100 --download-weights
```

Results are saved to `project/results/accuracy.csv` by default. `--download-weights` is needed only when pretrained model weights are not already cached. Mitchell is also available with `--pes mitchell`, but is substantially slower.
