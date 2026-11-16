# Cesar's accuracy work — CS 441

Extract the archive into the root of the team's `scale-sim-v3` repository on the `CO_Accuracy` branch. Accuracy files now live in `project/accuracy/`. Keep the existing `project/sweep.py`, `project/energy.py`, and other teammates' files unchanged.

## Installation

Use the existing Python 3.11 virtual environment. The project's earlier NumPy requirement remains `numpy<2`.

```bash
source scalesim-env/Scripts/activate
python -m pip install -r project/accuracy/requirements.txt
```

On your Windows PC, set `TORCH_HOME=D:/torch-cache` to store pretrained weights on D. Also keep `PIP_CACHE_DIR`, `TEMP`, and `TMP` on D if C is full.

## Quick tests

From the repository root:

```bash
python -m project.accuracy.shiftmac
python -m unittest project.accuracy.test_shiftmac -v
python -m project.accuracy.run --help
```

These do not require an ImageNet dataset or pretrained downloads.

## Dataset setup

Use a licensed, labeled ImageNet-1k validation subset. The script expects:

```text
D:/Datasets/imagenet-val/
  n01440764/
    image1.JPEG
  n01443537/
    image2.JPEG
```

A folder containing all **1000** ImageNet synset class directories can use the default sorted class indices. For any smaller subset (or custom folder names) pass `--class-map D:/Datasets/class-map.json` with entries like `{"n01440764": 0, "n01443537": 1}`. These are example mappings; check against the **exact** index mapping of your pretrained torchvision ResNet-18. Wrong class labels invalidate reported accuracy.

Standard ResNet-18 was trained on ImageNet-1k, so do not use raw CIFAR-10 labels for its top-1 evaluation.

## Run baseline, PoT, and APoT

```bash
export TORCH_HOME=D:/torch-cache
python -m project.accuracy.run --data D:/Datasets/imagenet-val --class-map D:/Datasets/class-map.json --pes baseline pot apot --max-images 100 --batch-size 8 --bits 4 --download-weights
```

The `--download-weights` switch is needed **only if** the weights have not yet been cached. Delete that switch on later runs. The script will **not** automatically download a dataset.

## Run Mitchell (very slow)

Mitchell multiplication is implemented with actual approximate products for **Conv2d and Linear**, not only changed weights. Start with **one image**, using the same fixed subset seed for baseline and Mitchell:

```bash
python -m project.accuracy.run --data D:/Datasets/imagenet-val --class-map D:/Datasets/class-map.json --pes baseline mitchell --max-images 1 --batch-size 1 --bits 4 --out project/results/accuracy_mitchell.csv
```

It may take considerable time even for one image on a CPU. Increase `--max-images` only after measuring performance. The implementation is a correctness-first proof of concept rather than a fast inference engine. Comparisons must use the *same subset* for baseline and Mitchell.

## Output

The CSV includes `workload,pe,bits,weights_id,dataset,subset_seed,num_images,num_correct,top1_pct`. The PE name for the exact method is `baseline` to match `project/sweep.py`. Energy and cycles are produced separately by your teammates. The accuracy script does **not** modify the sweep pipeline.

The PoT and APoT experiments use the starter's floating-point weight approximations followed by ordinary PyTorch multiplication. Mitchell uses floating-point approximate multiplication. **These are not yet faithful INT8 hardware arithmetic simulations.** The hardware's energy and area estimates are separate models; document that limitation in the paper.

The original `project/shiftmac.py` entry point remains as a small compatibility wrapper; the arithmetic now lives in `project/accuracy/shiftmac.py`. The original arithmetic is included with small fixes for all-zero tensors and an APoT residual offset. Check that you want these changes before merging with your team's code.

## Moving from the earlier layout

This ZIP adds the new `project/accuracy/` directory. After extraction, **remove the old** `project/accuracy.py`, `project/mitchell_layers.py`, `project/test_shiftmac.py`, and `project/requirements-accuracy.txt` if those files still exist. Do not remove unrelated teammates' files. The included `project/shiftmac.py` is a compatibility wrapper for old imports and commands.

Run commands from the repository root using `python -m project.accuracy.run` or `python -m project.accuracy.shiftmac` so that relative imports work correctly.
