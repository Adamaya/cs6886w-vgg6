# cs6886w-vgg6
Asssignment 1 for cs6886w-vgg6


# Environment Setup to train and test the model
- Download the anaconda software and install it. [Download Link](https://www.anaconda.com/download)
- Run the following commands to configure the runtime environment:
```
conda create -n vgg6 python=3.10 -y
conda activate vgg6
pip install torch torchvision matplotlib pandas scikit-learn tqdm seaborn wandb
```


# Baseline run (Q1)
```
python train.py --out_dir results/baseline --epochs 200 --batch_size 128  --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4   --scheduler cosine --activation relu --wandb --seed 42
```

# Activation sweep (Q2a)
```
for act in relu gelu silu tanh sigmoid; do
  python train.py --out_dir results/act_$act --activation $act --wandb --seed 42
done
```

# Optimizer sweep (Q2b)
```
python train.py --out_dir results/opt_sgd        --optimizer sgd --lr 0.1  --momentum 0.9 --scheduler cosine --wandb --seed 42
python train.py --out_dir results/opt_nesterov   --optimizer sgd --lr 0.1  --momentum 0.9 --nesterov --scheduler cosine --wandb --seed 42
python train.py --out_dir results/opt_adam       --optimizer adam --lr 1e-3 --scheduler cosine --wandb --seed 42
python train.py --out_dir results/opt_rmsprop    --optimizer rmsprop --lr 1e-3 --momentum 0.9 --scheduler cosine --wandb --seed 42
python train.py --out_dir results/opt_adagrad    --optimizer adagrad --lr 1e-2 --scheduler none --wandb --seed 42
python train.py --out_dir results/opt_nadam      --optimizer nadam --lr 1e-3 --scheduler cosine --wandb --seed 42
```

Batch size / epochs / LR (Q2c) (windows cmd)
```
for %B in (64 128 256) do python train.py --out_dir results/bs_%B --batch_size %B --wandb --seed 42

for %E in (50 100 200) do python train.py --out_dir results/ep_%E --epochs %E --wandb --seed 42

for %L in (0.01 0.05 0.1) do (set "lr=%L" & call set "lrmod=%%lr:.=%%" & call python train.py --out_dir results/lr_%%lrmod%% --lr %L --wandb --seed 42)
```


### Basic Usage
```bash
# Test the baseline model
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data"

# Test with custom batch size
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --batch_size 64

# Test without plots (faster)
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --no_plots

# Test with detailed analysis
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --detailed_analysis
```

### Save Results
```bash
# Save plots and summary to files
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --save_plots --plot_dir "./test_results"
```

### Best Configuration test
```bash
python test.py --model_path "./results/bs_256/best.pt"  --data_root "../../data" --detailed_analysis
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | `./results/baseline/best.pt` | Path to the trained model checkpoint |
| `--data_root` | `./data` | Path to CIFAR-10 dataset |
| `--batch_size` | `128` | Batch size for testing |
| `--save_plots` | `False` | Save plots to files |
| `--plot_dir` | `./test_results` | Directory to save plots |
| `--seed` | `42` | Random seed for reproducibility |
| `--no_plots` | `False` | Skip plotting visualizations |
| `--detailed_analysis` | `False` | Perform detailed prediction analysis |
