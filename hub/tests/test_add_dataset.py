import boto3
import botocore
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from hub import Dataset

bukcket_name = "shashank-activeloop"
ds_root = "s3://shashank-activeloop/"


def check_dataset_exists(dataset_name):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bukcket_name, dataset_name + "/dataset_meta.json").load()
        return True
    except botocore.exceptions.ClientError as e:
        print(e)
        return False


def add_tfds(ds, tfds_name, split):
    dataset_name = tfds_name + "_" + split
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

    ...


def test_load_tfds():
    datasets_to_load = [
        'beans',
        'binary_alpha_digits',
        'caltech101',
        'caltech_birds2010',
        'caltech_birds2011',
        'cars196',
        'cassava',
        'cats_vs_dogs',
        'chexpert',
        'cifar10',
        'cifar100',
        'cifar10_1',
        'cifar10_corrupted',
        'citrus_leaves',
        'cmaterdb',
        'colorectal_histology',
        'colorectal_histology_large',
        'curated_breast_imaging_ddsm',
        'cycle_gan',
        'deep_weeds',
        'diabetic_retinopathy_detection',
        'dmlab',
        'dtd',
        'emnist',
        'eurosat',
        'fashion_mnist',
        'food101',
        'geirhos_conflict_stimuli',
        'horses_or_humans',
        'i_naturalist2017',
        'imagenet2012',
        'imagenet2012_corrupted',
        'imagenet2012_real',
        'imagenet2012_subset',
        'imagenet_a',
        'imagenet_r',
        'imagenet_resized',
        'imagenet_v2',
        'imagenette',
        'imagewang',
        'kmnist',
        'lfw',
        'malaria',
        'mnist',
        'mnist_corrupted',
        'omniglot',
        'oxford_flowers102',
        'oxford_iiit_pet',
        'patch_camelyon',
        'pet_finder',
        'places365_small',
        'plant_leaves',
        'plant_village',
        'plantae_k',
        'quickdraw_bitmap',
        'resisc45',
        'rock_paper_scissors',
        'siscore',
        'smallnorb',
        'so2sat',
        'stanford_dogs',
        'stanford_online_products',
        'stl10',
        'sun397',
        'svhn_cropped',
        'tf_flowers',
        'uc_merced',
        'vgg_face2',
        'visual_domain_decathlon'
    ]

    for dataset in datasets_to_load:

        print("\n\n\n.....................")
        print("Trying to convert dataset: " + str(dataset))
        try:
            # ds = tfds.load(tfds_name, split=split).batch(100)
            ds = tfds.load(dataset)
            for split in ds:
                try:
                    if not check_dataset_exists(dataset + "_" + split):
                        ds = tfds.load(dataset, split=split)
                        print("Trying to convert dataset: " + str(dataset) + ", with split: " + str(split))
                        add_tfds(ds, dataset, split).batch(100)
                        print("Successfully loaded dataset")
                    else:
                        print("dataset already exists with name: " + dataset + " and split: " + split)
                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)
        print("\n\n\n.....................")


test_load_tfds()
