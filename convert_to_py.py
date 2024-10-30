import torch
import torch_tensorrt
from torch_tensorrt.ts import ptq
from torchvision import datasets, transforms
import os
from utils import *
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Subset


# INT8 후처리 양자화(PTQ)를 위한 모델 컴파일 함수
def compile_int8_model(data_dir, model, batch_size):
    # CIFAR-10 데이터셋 설정
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    indices = random.sample(range(len(train_loader)), 1024)
    subset = Subset(train_loader.dataset, indices)
    calib_loader = DataLoader(subset, batch_size=batch_size)

    # 캘리브레이션 캐시 파일 설정
    calibration_cache_file = "/tmp/vgg16_TRT_ptq_calibration.cache"

    # INT8 캘리브레이터 생성
    calibrator = ptq.DataLoaderCalibrator(
        calib_loader, cache_file=calibration_cache_file, use_cache=False
    )

    # 모델 컴파일을 위한 설정
    inputs = [torch_tensorrt.Input((batch_size, 3, 32, 32), dtype=torch.float)]
    compile_spec = {
        "inputs": inputs,
        # "workspace_size": 1 << 22,
        "calibrator": calibrator,
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float16, torch.int8},  # FP16, INT8 사용
    }
    print("Compiling and quantizing module")
    trt_model = torch_tensorrt.ts.compile(model, **compile_spec)
    return trt_model


def evaluate_model(model, val_loader, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2000 == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(
        " * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
            top1=top1, top5=top5, time=val_end_time - val_start_time
        )
    )

    return losses.avg, top1.avg, top5.avg


def main(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = torch.jit.load(model_path).to(device).eval()

    # 평가 데이터셋 로드
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    BATCHSIZE = 1
    eval_dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=transform, download=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8
    )

    # JIT 모델 정확도 계산
    benchmark(model=model, input_shape=(BATCHSIZE, 3, 32, 32), dtype="int8", nruns=100)
    print("Validating ...")
    acc = evaluate_model(model, eval_loader, nn.CrossEntropyLoss(), device)

    # exit()

    # INT8 모델 컴파일
    trt_model = compile_int8_model(data_dir, model, BATCHSIZE)
    benchmark(
        model=trt_model, input_shape=(BATCHSIZE, 3, 32, 32), dtype="int8", nruns=100
    )
    print("Validating ...")
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=8
    )
    acc = evaluate_model(model, eval_loader, nn.CrossEntropyLoss(), device)

    # torch.jit.save(trt_model, "/tmp/ptq_vgg16.trt.ts")


if __name__ == "__main__":
    import sys

    # if len(sys.argv) < 3:
    #     print("Usage: python ptq.py <path-to-module> <path-to-cifar10>")
    #     sys.exit(-1)

    sys.argv = [
        "convert_to_py.py",
        ###
        # "examples/int8/training/vgg16/trained_vgg16_batch32_trace.jit.pt",
        # # batch 32: 90.590 16319.16 -> 90.570 112461.96
        ###
        # "examples/int8/training/vgg16/trained_vgg16_batch32_script.jit.pt",
        # # batch 32: 92.790 19227.88 -> 92.790 75539.02
        ###
        "examples/int8/training/vgg16/trained_vgg16_batch1_trace.jit.pt",
        # batch 1: 47.580 876.63 -> 48.000  4893.49
        ###
        # "examples/int8/training/vgg16/trained_vgg16_batch1_script.jit.pt",
        # # batch 1: 92.790 1030.26 -> 92.790 4958.69
        ###
        "~/Datasets",
    ]
    model_path = sys.argv[1]
    data_dir = sys.argv[2]
    main(model_path, data_dir)


"""
##
##
"examples/int8/training/vgg16/trained_vgg16_batch1_script.jit.pt",
        # # batch 1: 92.790 1030.26 -> 92.790 4958.69
##
## The Log

Files already downloaded and verified
Warm up ...
Start timing ...
Iteration 20/100, avg batch time 0.96 ms
Iteration 40/100, avg batch time 0.96 ms
Iteration 60/100, avg batch time 0.96 ms
Iteration 80/100, avg batch time 0.96 ms
Iteration 100/100, avg batch time 0.96 ms
Input shape: torch.Size([1, 3, 32, 32])
Average throughput: 1039.95 images/second
Validating ...
Test: [0/10000] Time 0.146 (0.146)      Loss 0.0001 (0.0001)    Prec@1 100.000 (100.000)        Prec@5 100.000 (100.000)
Test: [2000/10000]      Time 0.002 (0.002)      Loss 0.0000 (0.2695)    Prec@1 100.000 (92.754) Prec@5 100.000 (99.850)
Test: [4000/10000]      Time 0.002 (0.002)      Loss 7.7180 (0.2958)    Prec@1 0.000 (92.802)   Prec@5 100.000 (99.750)
Test: [6000/10000]      Time 0.002 (0.002)      Loss 0.0020 (0.2921)    Prec@1 100.000 (92.768) Prec@5 100.000 (99.717)
Test: [8000/10000]      Time 0.002 (0.002)      Loss 0.0000 (0.2944)    Prec@1 100.000 (92.688) Prec@5 100.000 (99.750)
 * Prec@1 92.790 Prec@5 99.760 Time 18.411
Files already downloaded and verified
Compiling and quantizing module
WARNING: [Torch-TensorRT] - Detected and removing exception in TorchScript IR for node:  = prim::If(%351) # <string>:5:2  block0():    -> ()  block1():     = prim::RaiseException(%298, %297) # <string>:5:2    -> ()
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
Warm up ...
Start timing ...
Iteration 20/100, avg batch time 0.20 ms
Iteration 40/100, avg batch time 0.20 ms
Iteration 60/100, avg batch time 0.20 ms
Iteration 80/100, avg batch time 0.20 ms
Iteration 100/100, avg batch time 0.20 ms
Input shape: torch.Size([1, 3, 32, 32])
Average throughput: 4958.69 images/second
Validating ...
Test: [0/10000] Time 0.327 (0.327)      Loss 0.0001 (0.0001)    Prec@1 100.000 (100.000)        Prec@5 100.000 (100.000)
Test: [2000/10000]      Time 0.002 (0.002)      Loss 0.0000 (0.2695)    Prec@1 100.000 (92.754) Prec@5 100.000 (99.850)
Test: [4000/10000]      Time 0.002 (0.002)      Loss 7.7180 (0.2958)    Prec@1 0.000 (92.802)   Prec@5 100.000 (99.750)
Test: [6000/10000]      Time 0.002 (0.002)      Loss 0.0020 (0.2921)    Prec@1 100.000 (92.768) Prec@5 100.000 (99.717)
"""
