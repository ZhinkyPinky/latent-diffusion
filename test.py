import torch

if __name__ == "__main__":
    # available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())