import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# The Dataset Class

class MyTabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


if __name__ == '__main__':

    # Some dummy data
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    y = [0, 1, 0, 1]

    dataset = MyTabularDataset(X, y)

    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")

    # DataLoader — The Real MVP
    train_loader = DataLoader(
        dataset,  # your Dataset object
        batch_size=2,  # how many samples per batch
        shuffle=True,  # randomize order each epoch? YES for training!
        num_workers=2,  # parallel data loading (0 = main thread only)
        drop_last=True  # drop the last incomplete batch? Usually True for training
    )

    # Iterate over batches
    for batch_features, batch_labels in train_loader:
        print(f"Batch features shape: {batch_features.shape}")  # torch.Size([2, 2])
        print(f"Batch labels shape: {batch_labels.shape}")  # torch.Size([2])
        break

    # torchvision.datasets
    # Downloads MNIST to './data'
    train_dataset = torchvision.datasets.MNIST(
        root='./data',  # where to store
        train=True,  # training set (60,000 images)
        transform=transforms.ToTensor(),  # convert PIL Image → Tensor
        download=True  # download if not present
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    print(f"Training samples: {len(train_dataset)}")  # 60000
    print(f"Test samples: {len(test_dataset)}")  # 10000
    print(f"Image shape: {train_dataset[0][0].shape}")  # torch.Size([1, 28, 28])
    print(f"Label: {train_dataset[0][1]}")
    
# Transforms — Your Preprocessing Pipeline

# Transforms let you chain preprocessing steps together. Think of them as an assembly line for your data:

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    ),
])

# Complete Working Example: MNIST with DataLoader

# ---- Step 1: Define transforms ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---- Step 2: Load datasets ----
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# ---- Step 3: Create DataLoaders ----
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

# ---- Step 4: Verify ----
for images, labels in train_loader:
    print(f"Batch of images:{images.shape}")   # [64, 1, 28, 28]
    print(f"Batch of labels:{labels.shape}")    # [64]
    print(f"Pixel range: [{images.min():.2f},{images.max():.2f}]")
    break

# Quick stats
print(f"\nTotal training batches:{len(train_loader)}")   # 60000/64 ≈ 938
print(f"Total test batches:{len(test_loader)}")           # 10000/1000 = 10