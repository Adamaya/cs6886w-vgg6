# cs6886w-vgg6
Asssignment 1 for cs6886w-vgg6

Commands for your README (copy/paste)

Environment

conda create -n vgg6 python=3.10 -y
conda activate vgg6
pip install torch torchvision matplotlib pandas
# optional:
pip install wandb


Baseline run (Q1)
```
python train.py --out_dir results/baseline --epochs 200 --batch_size 128  --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 5e-4   --scheduler cosine --activation relu --seed 42
```

Activation sweep (Q2a)
```
for act in relu gelu silu tanh sigmoid; do
  python train.py --out_dir results/act_$act --activation $act --seed 42
done
```

Optimizer sweep (Q2b)
```
python train.py --out_dir results/opt_sgd        --optimizer sgd --lr 0.1  --momentum 0.9 --scheduler cosine --seed 42
python train.py --out_dir results/opt_nesterov   --optimizer sgd --lr 0.1  --momentum 0.9 --nesterov --scheduler cosine --seed 42
python train.py --out_dir results/opt_adam       --optimizer adam --lr 1e-3 --scheduler cosine --seed 42
python train.py --out_dir results/opt_rmsprop    --optimizer rmsprop --lr 1e-3 --momentum 0.9 --scheduler cosine --seed 42
python train.py --out_dir results/opt_adagrad    --optimizer adagrad --lr 1e-2 --scheduler none --seed 42
python train.py --out_dir results/opt_nadam      --optimizer nadam --lr 1e-3 --scheduler cosine --seed 42
```

Batch size / epochs / LR (Q2c)
```
for bs in 64 128 256; do
  python train.py --out_dir results/bs_$bs --batch_size $bs --seed 42
done

for ep in 50 100 200; do
  python train.py --out_dir results/ep_$ep --epochs $ep --seed 42
done

for lr in 0.01 0.05 0.1; do
  python train.py --out_dir results/lr_${lr//./} --lr $lr --seed 42
done
```

W&B logging (Q3, optional)

wandb login
python train.py --wandb --out_dir results/baseline --seed 42
# in W&B UI: make parallel-coordinates & scatter plots, export images and paste into PDF

What you still need to do locally

Run the commands (baseline + sweeps).

Paste your best validation accuracy config (Q4) and the test top-1.

Insert the plots (from results/*/*.png or W&B exports) into the PDF.

Upload the trained best.pt, CSV logs, and code to your GitHub repo; put the link in the PDF.

If you want, I can also generate a short README.md and a results table template you can drop into your repo.
