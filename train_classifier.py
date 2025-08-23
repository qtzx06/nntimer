import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import torch.multiprocessing as mp
from tqdm import tqdm
import time

# --- config ---
DATA_DIR = "datasets/cube_data/state"
MODEL_PATH = "cube_classifier.pth"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # using 0 to disable multiprocessing and prevent crashes


def train():
    """
    trains the cube state classifier model.
    """
    print("starting classifier training...")

    # --- 1. data augmentation and preprocessing ---
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # --- 2. load and split dataset ---
    print(f"loading data from '{DATA_DIR}'...")
    full_dataset = datasets.ImageFolder(DATA_DIR)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # apply the respective transforms by creating new dataset instances
    train_dataset.dataset.transform = data_transforms["train"]
    val_dataset.dataset.transform = data_transforms["val"]

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        ),
        "val": DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        ),
    }
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"found {num_classes} classes: {class_names}")
    print(
        f"important: class to index mapping: {full_dataset.class_to_idx}"
    )  # <-- added this line
    print(
        f"training set size: {dataset_sizes['train']}, validation set size: {dataset_sizes['val']}"
    )

    # --- 3. define the model ---
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"using device: {device}")

    # --- 4. define loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. training loop ---
    best_acc = 0.0
    since = time.time()

    for epoch in range(EPOCHS):
        print(f"epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # wrap the dataloader with tqdm for a progress bar
            progress_bar = tqdm(
                dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}"
            )
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # update progress bar description
                progress_bar.set_postfix(
                    loss=running_loss / (progress_bar.n + 1),
                    acc=running_corrects.double() / (progress_bar.n + 1) / BATCH_SIZE,
                )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            # save the model if it has the best validation accuracy so far
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                print(f"new best validation accuracy: {best_acc:.4f}. saving model...")
                torch.save(model.state_dict(), MODEL_PATH)

    time_elapsed = time.time() - since
    print(f"training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"best val acc: {best_acc:4f}")


if __name__ == "__main__":
    # set the start method for multiprocessing to 'spawn' to avoid deadlocks with cuda.
    # this must be in the __name__ == '__main__' block.
    try:
        mp.set_start_method("spawn", force=True)
        print("set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass

    if not os.path.isdir(DATA_DIR):
        print(f"error: data directory '{DATA_DIR}' not found.")
    else:
        train()
