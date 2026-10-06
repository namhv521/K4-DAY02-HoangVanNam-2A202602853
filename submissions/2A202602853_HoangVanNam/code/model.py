"""timm backbones and native AdamW groups."""
import torch
import timm

SUGGESTED_BACKBONES = {'resnet50': 'resnet50', 'resnext50': 'resnext50_32x4d', 'convnext_tiny': 'convnext_tiny', 'deit_small': 'deit_small_patch16_224', 'swin_tiny': 'swin_tiny_patch4_window7_224', 'efficientnet_b0': 'efficientnet_b0', 'mobilenetv3': 'mobilenetv3_large_100'}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init='finetune'):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(f'Unknown init: {init}')
    model = timm.create_model(SUGGESTED_BACKBONES.get(name, name), pretrained=pretrained and init != 'scratch', num_classes=num_classes, drop_rate=drop_rate)
    if init == 'frozen':
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad_(True)


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in model.get_classifier().parameters()}
    no_decay = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else set()
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_head if id(p) in head else lr_backbone
        decay = 0. if p.ndim <= 1 or name.endswith('.bias') or name in no_decay else weight_decay
        groups.setdefault((lr, decay), []).append(p)
    return [{'params': params, 'lr': lr, 'weight_decay': wd} for (lr, wd), params in groups.items()]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    """Optional fvcore estimate; never silently report unsupported operations as zero."""
    from fvcore.nn import FlopCountAnalysis
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        counter = FlopCountAnalysis(model, torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device))
        result = counter.total() / 1e9
        if counter.unsupported_ops():
            raise ValueError(f'Incomplete GMAC estimate: {dict(counter.unsupported_ops())}')
        return result
    finally:
        for module, training in modes.items():
            module.training = training
