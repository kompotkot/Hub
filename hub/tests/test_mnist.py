from warnings import filterwarnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from hub.api.dataset import Dataset
from hub.constants import MIN_FIRST_CACHE_SIZE
from hub.core import MemoryProvider, LocalProvider
from hub.util.cache_chain import get_cache_chain


def test_mnist_to_hub():
    root = "/Users/shashank/Projects/activeloop/activeloop/datasets/test"
    storage_providers = [MemoryProvider(root), LocalProvider(root)]
    cache_sizes = [MIN_FIRST_CACHE_SIZE]
    provider = get_cache_chain(storage_providers, cache_sizes)
    ds = Dataset(provider=provider)

    # Downloaded dataset from here: https://www.kaggle.com/zalando-research/fashionmnist
    df_train = pd.read_csv("/Users/shashank/Downloads/archive/fashion-mnist_train.csv")
    train_label = df_train["label"]
    train_data = np.array(df_train[df_train.columns[1:]])
    train_label_np = np.array(train_label)

    train_data = train_data.reshape(train_data.shape[0], 28, 28)

    len(train_data[1])
    len(train_label)
    print(train_data.shape)
    print(train_data.nbytes)
    ds["image"] = train_data
    ds["label"] = train_label_np
    ds.provider.flush()


def test_read_mnist():
    root = "/Users/shashank/Projects/activeloop/activeloop/datasets/test"
    storage_providers = [MemoryProvider(root), LocalProvider(root)]
    cache_sizes = [MIN_FIRST_CACHE_SIZE]
    provider = get_cache_chain(storage_providers, cache_sizes)

    ds = Dataset(provider=provider)
    for i in tqdm(range(0, len(ds))):
        ds["image"][i].numpy()
        ds["label"][i].numpy()
