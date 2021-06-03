import torch.distributed as dist


def check_dist_init():
    return dist.is_available() and dist.is_initialized()
