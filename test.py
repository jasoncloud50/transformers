import torch
print(torch.cuda.is_available()) # 应返回 True
print(torch.__version__)
print(torch.version.cuda)        # 应显示 13.0
