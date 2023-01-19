import numpy as np
import time
from build_features import prepare_data, FoodDataset
from torch.utils.data import DataLoader


X_train, _, y_train, _ = prepare_data(10000, 128)
train_dataset = FoodDataset(X_train, y_train)
batch_size = 128
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
)

if __name__ == "__main__":
    res = []
    for _ in range(1):
        start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 100:
                break
        end = time.time()

        res.append(end - start)

    res = np.array(res)
    print("Execution time: {mean} +- {std}".format(mean=np.mean(res), std=np.std(res)))
