from hub import Dataset


def test_read_dataset():
    root = "s3://shashank-activeloop/"
    dataset_name = "cifar10_train"
    path = root + dataset_name
    ds = Dataset(path)
    print(ds["image"][0].numpy())
