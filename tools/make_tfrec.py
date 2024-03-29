'''
This script aims to create tfrecord tar shards with multi-processing.
'''

import os
import random
import datetime
from multiprocessing import Process
from torchvision.datasets.folder import ImageFolder

import struct
import tfrecord

# def find_classes(directory: str):
#     """Finds the class folders in a dataset.

#     See :class:`DatasetFolder` for details.
#     """
#     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
#     if not classes:
#         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

#     class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
#     return classes, class_to_idx

# classes, classes_2_idx = find_classes("/home/chenyaofo/datasets/imagenet-c/fog/1")
# subclasses, subclasses_2_idx = find_classes("/home/chenyaofo/datasets/imagenet-r")

# d = {}
# for k, v in subclasses_2_idx.items():
#     d[v] = classes_2_idx[k]

def create_index(tfrecord_file: str, index_file: str) -> None:
    """
    refer to https://github.com/vahidk/tfrecord/blob/master/tfrecord/tools/tfrecord2idx.py
    Create index from the tfrecords file.
    Stores starting location (byte) and length (in bytes) of each
    serialized record.
    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.
    index_file: str
        Path where to store the index file.
    """
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()
    outfile.close()


def make_wds_shards(pattern, num_shards, num_workers, samples, map_func, **kwargs):
    random.shuffle(samples)
    samples_per_shards = [samples[i::num_shards] for i in range(num_shards)]
    shard_ids = list(range(num_shards))
    processes = [
        Process(
            target=write_partial_samples,
            args=(
                pattern,
                shard_ids[i::num_workers],
                samples_per_shards[i::num_workers],
                map_func,
                kwargs
            )
        )
        for i in range(num_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_partial_samples(pattern, shard_ids, samples, map_func, kwargs):
    for shard_id, samples in zip(shard_ids, samples):
        write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs)


def write_samples_into_single_shard(pattern, shard_id, samples, map_func, kwargs):
    fname = pattern % shard_id
    print(f"[{datetime.datetime.now()}] start to write samples to shard {fname}")
    # stream = TarWriter(fname, **kwargs)
    writer = tfrecord.TFRecordWriter(fname)
    size = 0
    for i, item in enumerate(samples):
        raw_data = map_func(item)
        size += len(raw_data["image"][0])
        writer.write(raw_data)

        if i % 1000 == 0:
            print(f"[{datetime.datetime.now()}] complete to write {i:06d} samples to shard {fname}")
    writer.close()
    print(f"[{datetime.datetime.now()}] complete to write samples to shard {fname}!!!")
    create_index(fname, fname+".idx")
    print(f"[{datetime.datetime.now()}] complete tfrecord2idx to shard {fname}!!!")
    return size


def main(source, dest, num_shards, num_workers):
    root = source
    items = []
    dataset = ImageFolder(root=root, loader=lambda x: x)
    for i in range(len(dataset)):
        items.append(dataset[i])

    def map_func(item):
        name, class_idx = item
        # class_idx = d[class_idx]
        # import ipdb; ipdb.set_trace()
        with open(os.path.join(name), "rb") as stream:
            image = stream.read()
        sample = {
            "fname": (bytes(os.path.splitext(os.path.basename(name))[0], "utf-8"), "byte"),
            "image": (image, "byte"),
            "label": (class_idx, "int")
        }
        return sample
    make_wds_shards(
        pattern=dest,
        num_shards=num_shards,  # 设置分片数量
        num_workers=num_workers,  # 设置创建wds数据集的进程数
        samples=items,
        map_func=map_func,
    )



if __name__ == "__main__":
    source = [
        "/home/chenyf/datasets/ImageNet-C/gaussian_noise",
        "/home/chenyf/datasets/ImageNet-C/impulse_noise",
        "/home/chenyf/datasets/ImageNet-C/shot_noise",
        # ---
        "/home/chenyf/datasets/ImageNet-C/defocus_blur",
        "/home/chenyf/datasets/ImageNet-C/glass_blur",
        "/home/chenyf/datasets/ImageNet-C/motion_blur",
        "/home/chenyf/datasets/ImageNet-C/zoom_blur",
        # ---
        "/home/chenyf/datasets/ImageNet-C/snow",
        "/home/chenyf/datasets/ImageNet-C/frost",
        "/home/chenyf/datasets/ImageNet-C/fog",
        "/home/chenyf/datasets/ImageNet-C/brightness",
        # ---
        "/home/chenyf/datasets/ImageNet-C/contrast",
        "/home/chenyf/datasets/ImageNet-C/elastic_transform",
        "/home/chenyf/datasets/ImageNet-C/pixelate",
        "/home/chenyf/datasets/ImageNet-C/jpeg_compression",
    ]

    source = [
        item.replace("/home/chenyf/datasets/ImageNet-C", "/gpfs01/dataset/TTA/imagenet-c") \
        for item in source
    ]

    # dest_prefix = "/home/chenyf/webdataset/ImageNet-C-tfrec/"
    dest_prefix = "/home/chenyaofo/datasets/imagenet-c-tfrec"

    for s in source:
        for i in [1, 2, 3, 4, 5]:
            print(f"[{datetime.datetime.now()}] start transfer {s}/{i}")
            os.makedirs(os.path.join(dest_prefix, os.path.basename(s), f"{i}"), exist_ok=False)
            main(
                source=os.path.join(s, str(i)),
                dest=os.path.join(dest_prefix, os.path.basename(s), f"{i}", "%06d.tfrec"),
                num_shards=32,
                num_workers=4
            )
    
    # source = "/home/chenyaofo/datasets/imagenet-r"
    # dest = "/home/chenyaofo/datasets/imagenet-r-tfrec"
    # os.makedirs(os.path.join(dest), exist_ok=False)
    # main(
    #     source=source,
    #     dest=os.path.join(dest, "%06d.tfrec"),
    #     num_shards=32,
    #     num_workers=4
    # )

    # source = "/home/chenyaofo/datasets/imagenet-c/gaussian_noise/3"
    # dest = "/home/chenyaofo/datasets/imagenet-r-tfrec/gaussian_noise/3"
    # os.makedirs(os.path.join(dest), exist_ok=False)
    # main(
    #     source=source,
    #     dest=os.path.join(dest, "%06d.tfrec"),
    #     num_shards=32,
    #     num_workers=4
    # )