import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import numpy as np

import tensorrt as trt
import torch_tensorrt as torchtrt
import torch_tensorrt.ts.ptq as PTQ

import timm

from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser(description="TRT_PRAC", add_help=False)
    parser.add_argument(
        "--model",
        default="vit_base",
        choices=[
            "vit_small",
            "vit_base",
            "deit_tiny",
            "deit_small",
            "deit_base",
            "swin_tiny",
            "swin_small",
        ],
        help="model",
    )
    parser.add_argument("--dataset", default="data/imagenet/", help="path to dataset")
    parser.add_argument(
        "--precision",
        default="int8",
        choices=[
            "org",
            "fp32",
            "fp16",
            "int8",
        ],
        help="precision",
    )
    parser.add_argument(
        "--calib-batchsize", default=32, type=int, help="batchsize of validation set"
    )
    parser.add_argument(
        "--calib-length",
        default=256,
        type=int,
        help="number of images for calibration",
    )
    parser.add_argument(
        "--val-batchsize", default=1, type=int, help="batchsize of validation set"
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="number of data loading workers (default: 1)",
    )
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=10000, type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    return parser


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def quantize_model(model, precision="fp16", calib_loader=None, args=None):
    if precision == "org":
        return model

    SHAPE = (1, 3, 224, 224)
    INPUT = [torchtrt.Input(SHAPE, dtype=torch.float32)]

    print("Precision: ", precision)

    model = torch.jit.script(model, example_inputs=INPUT).eval().cuda()
    assert torchtrt.ts.check_method_op_support(model)

    compile_spec = {
        "inputs": INPUT,
        # "workspace_size": 1 << 22,
        "truncate_long_and_double": True,
        "device": {
            "device_type": torchtrt.DeviceType.GPU,
            "gpu_id": 0,
            "dla_core": 0,
            "allow_gpu_fallback": False,
        },
    }

    if precision == "fp32":
        # Average throughput: 1246.62 images/second
        compile_spec["enabled_precisions"] = {torch.float32}
        trt_model_fp32 = torchtrt.compile(model, **compile_spec)
        return trt_model_fp32

    elif precision == "fp16":
        # Average throughput: 1970.76 images/second
        compile_spec["enabled_precisions"] = {torch.float16}
        trt_model_fp16 = torchtrt.compile(model, **compile_spec)
        return trt_model_fp16

    elif precision == "int8":
        assert calib_loader is not None
        # Average throughput: 1996.48 images/second
        # TODO: Implement int8 quantization
        # now using fp16...
        # https://github.com/pytorch/TensorRT/tree/main/tests/py/ts
        # https://github.com/pytorch/TensorRT/blob/main/tests/py/ts/ptq/test_ptq_dataloader_calibrator.py

        calibrator = PTQ.DataLoaderCalibrator(
            calib_loader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=PTQ.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )
        compile_spec["calibrator"] = calibrator
        compile_spec["enabled_precisions"] = {torch.int8}

        # with torchtrt.logging.debug():
        # torch.compile -> determine the ir len (torchscript) -> torch.ts.compile -> compiled C lang::CompileGraph
        # https://pytorch.org/TensorRT/ts/ptq.html#ptq
        # https://pytorch.org/TensorRT/_cpp_api/function_namespacetorch__tensorrt_1_1torchscript_1a6e19490a08fb1553c9dd347a5ae79db9.html#exhale-function-namespacetorch-tensorrt-1-1torchscript-1a6e19490a08fb1553c9dd347a5ae79db9
        trt_model_int8 = torchtrt.compile(model, **compile_spec)
        print("calibration is done")

        return trt_model_int8

    elif precision == "int4":
        raise NotImplementedError


def main():
    print(args)
    seed(args.seed)

    model_zoo = {
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "deit_tiny": "deit_tiny_patch16_224",
        "deit_small": "deit_small_patch16_224",
        "deit_base": "deit_base_patch16_224",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
    }

    device = torch.device(args.device)

    # Build dataloader
    print("Building dataloader ...")
    train_loader, val_loader = build_dataset(args)

    indices = random.sample(range(len(train_loader)), args.calib_length)
    subset = Subset(train_loader.dataset, indices)
    calib_loader = DataLoader(subset, batch_size=1)

    # Build model
    print("Building model ...")
    #  * Prec@1 72.138 Prec@5 91.128 Time 366.984
    # Average throughput: 291.55 images/second
    model = timm.create_model(model_zoo[args.model], pretrained=True).cuda().eval()
    # benchmark(model=model, input_shape=(1, 3, 224, 224), dtype="int8", nruns=100)

    out_model = quantize_model(
        model,
        precision=args.precision,
        calib_loader=calib_loader,
        args=args,
    )

    benchmark(model=out_model, input_shape=(1, 3, 224, 224), dtype="int8", nruns=100)

    print("Validating ...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, out_model, nn.CrossEntropyLoss().to(device), device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TRT_PRAC", parents=[get_args_parser()])
    args = parser.parse_args()
    main()
