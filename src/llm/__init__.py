def print_num_params(model):
    params = [
        (param.numel(), param.numel() if param.requires_grad else 0)
        for _, param in model.named_parameters()
    ]
    all, train = map(sum, zip(*params))
    print(f"{train=} / {all=} {train/all:f}")