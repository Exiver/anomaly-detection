import json
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from anomaly.config import Settings
from anomaly.data.mvtec import MVTecDataset
from anomaly.models.autoencoder import ConvAutoencoder
from anomaly.utils import seed_everything

log = structlog.get_logger()

@torch.no_grad()
def _validate(model: ConvAutoencoder, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        total += F.mse_loss(model(x), x) * x.size(0)
        count += x.size(0)
    return (total / count).item()


def train_autoencoder(category: str, settings: Settings, device: torch.device) -> Path:
    seed_everything(settings.seed)

    fully = MVTecDataset(
        settings.data_root, category, "train", settings.image_size, settings.crop_size
    )
    n_val = max(1, int(len(fully) * settings.val_fraction))
    generator = torch.Generator().manual_seed(settings.seed)
    train_set, val_set = random_split(
        fully, [len(fully) - n_val, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=settings.batch_size, 
        shuffle= True,
        num_workers=settings.num_workers,
        drop_last=len(train_set) > settings.batch_size,
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=settings.batch_size, 
        shuffle=True,
        num_workers=0
    )

    out_dir = settings.artifacts_dir / "ae"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{category}.pt"

    model = ConvAutoencoder(settings.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)

    best_val = float("inf")
    bad_epochs = 0
    history : list[dict] = []

    log.info(
        "train_start",
        category= category,
        train_images = len(train_set),
        val_images = len(val_set),
        device = str(device)
    )

    for epoch in range(1, settings.epochs + 1):
        model.train()
        running, count = 0.0 , 0
        for batch in train_loader:
            x = batch["image"].to(device)
            loss = F.mse_loss(model(x), x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            count += x.size(0)

        train_loss = running / count
        val_loss = _validate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log.info("epoch", category=category, epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint)

        else: 
            bad_epochs += 1
            if bad_epochs >= settings.patience:
                log.info("early_stop", category=category, epoch= epoch)
                break

    (out_dir / f"{category}_history.json").write_text(json.dumps(history, indent=2))
    log.info("train_done", category= category, best_val_loss= round(best_val, 5))
    return checkpoint
