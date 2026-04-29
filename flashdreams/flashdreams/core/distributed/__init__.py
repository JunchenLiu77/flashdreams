import ctypes
import math
import os
from datetime import timedelta

import pynvml
import torch
import torch.distributed as dist
from loguru import logger


class Device:
    # TODO: fill in docstring.

    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)  # type: ignore

    def __init__(self, device_idx: int):
        # TODO: fill in docstring.
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        # TODO: fill in docstring.
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        # TODO: fill in docstring.
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, Device._nvml_affinity_elements
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        return [i for i, e in enumerate(affinity_list) if e != 0]


def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except (pynvml.NVMLError, OSError) as e:
        logger.warning(f"Failed to set device affinity: {e}")
    # Set up NCCL communication.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timeout_timedelta,
            device_id=local_rank,
        )
        logger.critical(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
        )
    # Increase the L2 fetch granularity for faster speed.
    # For oss, we need to search for the library in site-packages.
    # if INTERNAL:
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    logger.info(f"Training with {dist.get_world_size()} GPUs.")
