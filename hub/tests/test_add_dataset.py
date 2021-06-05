import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from hub import Dataset

ds_root = "s3://shashank-activeloop/"


def add_tfds(tfds_name, split):
    dataset_name = tfds_name + split
    ds = tfds.load(tfds_name, split=split).batch(100)
    ds_hub = Dataset(ds_root + dataset_name)
    ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
    n = 0
    for ex in tqdm(ds_numpy):
        # print(ex)
        for col in ex:
            if col not in ds_hub.tensors:
                print("Creating tensor with name: " + str(col))
                ds_hub.create_tensor(col)
            # if not isinstance(ex[col], np.ndarray):
            #     ex[col] = np.array(ex[col])
            # if not isinstance(ex[col], bytes):
            # TODO handle strings
            # t = list(ex[col])
            # ex[col] = np.array([t])
            if not isinstance(ex[col], np.ndarray):
                print("Converting to array")
                try:
                    ds_hub[col].extend(np.array([ex[col]]))
                except Exception as e:
                    print(e)
                    print("Didn't work converting to np.array")
            else:
                ds_hub[col].extend(ex[col])
        n += 1
        # if n > 10:
        #     break
    ds_hub.storage.flush()


def test_load_tfds():
    datasets_to_load = [
        ["mnist", "train"],
        ["mnist", "test"],
        ["beans", "test"],
        ["beans", "train"],
        ["beans", "validation"],
        ["bigearthnet", "train"],
        ["binary_alpha_digits", "train"],
        ["caltech101", "test"],
        ["caltech101", "train"],
        ["caltech_birds2010", "train"],
        ["caltech_birds2010", "test"],
        ["caltech_birds2011", "train"],
        ["caltech_birds2011", "test"],
        ["cars196", "train"],
        ["cars196", "test"],
        ["cassava", "train"],
        ["cassava", "test"],
        ["cassava", "validation"],
        ["cats_vs_dogs", "train"],
        ["cifar10", "train"],
        ["cifar10", "test"],
        ["cifar100", "train"],
        ["cifar100", "test"],
        ["cifar10_1", "test"],
        ["cifar10_corrupted", "test"],
        ["citrus_leaves", "train"],
        ["cmaterdb", "train"],
        ["cmaterdb", "test"],
    ]
    for dataset, split in datasets_to_load:
        print("Trying to convert dataset: " + str(dataset) + ", with split: " + split)
        print(dataset)
        print(split)
        try:
            add_tfds(dataset, split)
            print("Successfully loaded dataset")
        except Exception as e:
            print(e)

test_load_tfds()