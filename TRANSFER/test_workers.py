import torch
from torch.utils.data import DataLoader, TensorDataset

# Windows shared-memory fix
torch.multiprocessing.set_sharing_strategy("file_system")

def main():
    # Dummy dataset
    X = torch.randn(1000, 21, 128, 256)
    y = torch.randint(0, 2, (1000,))
    ds = TensorDataset(X, y)

    for n in [0, 1, 2, 4]:   # Avoid 6 on Windows with limited RAM
        try:
            print(f"\nTesting num_workers={n}")

            loader = DataLoader(
                ds,
                batch_size=64,
                num_workers=n,
                pin_memory=False,   # IMPORTANT for Windows stability
                persistent_workers=False
            )

            for i, (xb, yb) in enumerate(loader):
                if i == 5:
                    break

            print(f"✅ Works with num_workers={n}")

        except Exception as e:
            print(f"⚠️ Failed with num_workers={n}: {e}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
