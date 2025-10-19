import argparse, os, math, random, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------- Utils ----------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def top1(outputs, targets):
    return (outputs.argmax(1) == targets).float().mean().item() * 100.0

# ---------- Augmentations ----------
class Cutout(object):
    def __init__(self, n_holes=1, length=16): self.n_holes, self.length = n_holes, length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

def cifar10_loaders(data_root, batch_size, use_cutout, cutout_length):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if use_cutout:
        train_tfms.append(Cutout(1, cutout_length))
    trans_t = transforms.Compose(train_tfms)
    trans_e = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR10(data_root, train=True, transform=trans_t, download=True)
    test  = datasets.CIFAR10(data_root, train=False, transform=trans_e, download=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# ---------- Model (VGG6) ----------
def make_layers(cfg, batch_norm=True, activation=nn.ReLU):
    layers, in_ch = [], 3
    Act = activation
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=not batch_norm)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), Act(inplace=True)]
            else:
                layers += [conv2d, Act(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)

class VGG6(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
        self.features = make_layers(cfg_vgg6, batch_norm=batch_norm, activation=activation)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(128, num_classes))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ---------- Train / Eval ----------
def create_optimizer(name, params, lr, weight_decay, momentum, nesterov):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def get_activation(name):
    name = name.lower()
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[name]

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    train_loader, val_loader = cifar10_loaders(args.data_root, args.batch_size, args.cutout, args.cutout_length)
    model = VGG6(num_classes=10, batch_norm=not args.no_bn, activation=get_activation(args.activation), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum, args.nesterov)
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)
    else:
        scheduler = None

    # Optional: Weights & Biases
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"[WARN] W&B not available: {e}")

    # CSV logger
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    best_val, best_path = 0.0, os.path.join(args.out_dir, "best.pt")
    for epoch in range(1, args.epochs+1):
        # Train
        model.train()
        total, right, loss_sum = 0, 0, 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += y.size(0)
            right += (out.argmax(1)==y).sum().item()
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum/total
        train_acc = 100.0*right/total

        # Eval
        model.eval()
        v_total, v_right, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = model(x)
                v_loss_sum += criterion(out, y).item() * y.size(0)
                v_total += y.size(0)
                v_right += (out.argmax(1)==y).sum().item()
        val_loss = v_loss_sum / v_total
        val_acc  = 100.0 * v_right / v_total

        if scheduler: scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{lr_now:.6f}"])

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now})

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best_val, "args": vars(args)}, best_path)

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:6.2f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:6.2f} | lr {lr_now:.5f}")

    # Quick plots (Matplotlib)
    import pandas as pd
    df = pd.read_csv(csv_path)
    for col, outname, ylabel in [
        ("train_loss","train_loss.png","Train Loss"),
        ("val_loss","val_loss.png","Val Loss"),
        ("train_acc","train_acc.png","Train Acc (%)"),
        ("val_acc","val_acc.png","Val Acc (%)"),
    ]:
        plt.figure()
        plt.plot(df["epoch"], df[col]); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(outname[:-4])
        plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, outname)); plt.close()

    print(f"[DONE] Best val_acc={best_val:.2f}. Artifacts at: {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./results/baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","adam","rmsprop","adagrad","nadam"])
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none","cosine","multistep"])
    p.add_argument("--activation", type=str, default="relu", choices=["relu","gelu","silu","tanh","sigmoid"])
    p.add_argument("--no_bn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--cutout", action="store_true")
    p.add_argument("--cutout_length", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs6886w-vgg6")
    args = p.parse_args()
    train(args)
