#conda env remove --name bio-env
#conda create --clone py35 --name py35-2
import torch

bAvailable = torch.cuda.is_available()
print(bAvailable)