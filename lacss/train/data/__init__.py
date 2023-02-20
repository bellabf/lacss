from .data_handler import DataHandler
from .dataset import DataLoader, Dataset
from .utils import (
    map_append,
    map_structure,
    train_validation_split,
    unpack_x_y_sample_weight,
)

from .array_adapter import ArrayDataAdapter
from .dataset import DataLoaderAdapter
from .generator_adapter import GeneratorDataAdapter
from .list_adapter import ListsOfScalarsDataAdapter

try:
    from .tf_dataset_adapter import TFDatasetAdapter
except ImportError:
    TFDatasetAdapter = None
try:
    from .torch_dataloader_adapter import TorchDataLoaderAdapter
except ImportError:
    TorchDataLoaderAdapter = None

ALL_ADAPTER_CLS = [
    ArrayDataAdapter,
    GeneratorDataAdapter,
    ListsOfScalarsDataAdapter,
    DataLoaderAdapter,
]

