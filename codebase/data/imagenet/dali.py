import os
import random
import warnings
import pathlib

try:
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.tfrecord as tfrec
except ImportError:
    warnings.warn("NVIDIA DALI library is unavailable, cannot load and preprocess dataset with DALI.")

from codebase.torchutils.distributed import world_size, rank


def _glob_by_suffix(path, pattern, percent):
    tars = list(map(str, pathlib.Path(path).glob(pattern)))
    tars = sorted(tars)
    tars = tars[:int(len(tars)*percent)]
    return tars

def _random_shuffle(*items):
    temp = list(zip(*items))
    random.shuffle(temp)
    res = zip(*temp)
    return [list(r) for r in res]

def glob_by_suffix(path, pattern, percent, mixed=False):
    if isinstance(path, (list, tuple)):
        mixed=True
    if mixed:
        rev = [list() for _ in pattern]
        for r, pt in zip(rev, pattern):
            for pa in path:
                r += _glob_by_suffix(pa, pt, percent)
        return _random_shuffle(*rev)
    else:
        return [_glob_by_suffix(path, pt, percent) for pt in pattern]

class DALIWrapper:
    def __init__(self, daliiterator):
        self.daliiterator = daliiterator

    def __iter__(self):
        self._iter = iter(self.daliiterator)
        return self

    def __next__(self):
        datas = next(self._iter)
        inputs = datas[0]["images"]
        targets = datas[0]["targets"].squeeze(-1)
        return inputs, targets

    def __len__(self):
        return len(self.daliiterator)


def build_dali_imagenet_loader(root, image_size, mean, std, batch_size, num_workers, dali_gpu, percent):
    # import ipdb; ipdb.set_trace()
    path, index_path = glob_by_suffix(root, ["*.tfrec", "*.idx"], percent=percent)
    # import ipdb; ipdb.set_trace()
    reader = fn.readers.tfrecord(
        path=path,
        index_path=index_path,
        features={
            "fname": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image": tfrec.FixedLenFeature((), tfrec.string, ""),
            "label": tfrec.FixedLenFeature([1], tfrec.int64,  -1),
        },
        shard_id=rank(),
        num_shards=world_size(),
        random_shuffle=True,
        initial_fill=int(os.environ.get("DALI_BUFFER_SIZE", 5000)),
        pad_last_batch=True,
        dont_use_mmap=True,  # If set to True, the Loader will use plain file I/O
        # instead of trying to map the file in memory. Mapping provides a small
        # performance benefit when accessing a local file system, but most network
        # file systems, do not provide optimum performance.
        prefetch_queue_depth=2,
        read_ahead=True,
        name="Reader")
    images_raw = reader["image"]
    labels = reader["label"]
    pipe = Pipeline(batch_size, num_workers, device_id=0)

    decoder_device = 'mixed'if dali_gpu else 'cpu'
    images = fn.decoders.image(images_raw, device=decoder_device, output_type=types.RGB)

    if not dali_gpu:
        images = images.gpu()
        labels = labels.gpu()

    # images = fn.resize(images, resize_shorter=int(image_size/7*8),)

    crops = fn.crop_mirror_normalize(images,
                                     dtype=types.FLOAT,
                                     output_layout="CHW",
                                     crop=(image_size, image_size),
                                     mean=[item * 255 for item in mean],
                                     std=[item * 255 for item in std],
                                     mirror=False)
    labels = fn.cast(labels, dtype=types.INT64)
    pipe.set_outputs(crops, labels)
    loader = DALIGenericIterator(pipe,
                                 output_map=["images", "targets"],
                                 #  output_map=["images"],
                                 auto_reset=True,
                                 last_batch_policy=LastBatchPolicy.PARTIAL,
                                 reader_name="Reader")

    loader = DALIWrapper(loader)

    return loader
