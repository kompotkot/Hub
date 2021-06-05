import tensorflow_datasets as tfds
from tqdm import tqdm

from hub import Dataset

ds_root = "/Users/shashank/Projects/activeloop/activeloop/datasets/"


def add_tfds(tfds_name):
    ds = tfds.load(tfds_name, split='train').batch(100)
    ds_hub = Dataset(ds_root + tfds_name)
    ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
    n = 0
    for ex in tqdm(ds_numpy):
        # print(ex)
        for col in ex:
            if col not in ds_hub.tensors:
                print("Creating tensor with name: " + str(col))
                ds_hub.create_tensor(col)
            ds_hub[col].extend(ex[col])
        n += 1
        if n > 10:
            break
    ds_hub.storage.flush()


def test_load_tfds():
    name = "coco"
    add_tfds(name)
