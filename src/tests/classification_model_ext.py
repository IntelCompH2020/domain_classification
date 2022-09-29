import numpy as np
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

class ClassificationModelExt(ClassificationModel):
  def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        **kwargs,
  ):
    #import pdb
    #pdb.set_trace()
    self.device =  'cuda' if cuda_device == -1 else 'cpu'
    super().__init__(model_type = model_type,model_name = model_name,tokenizer_type = tokenizer_type,tokenizer_name = tokenizer_name,num_labels = num_labels,
                     weight = weight,args = args, use_cuda = use_cuda,cuda_device = cuda_device,onnx_execution_provider = onnx_execution_provider, **kwargs)

  def doStatistics(self,df_train):
    counts = np.zeros(200)
    for i,val in enumerate(np.arange(0,200)*0.01-1):
        counts[i] = np.sum((df_train.labels>val) & (df_train.labels<(val+0.01)))

    #prevent division 0 error
    counts[counts==0] = 10**-20

    #real part
    weights = (len(df_train)/(counts>(10**-20)).sum())/counts
    weights[weights>(10**10)] = np.max(weights[weights<(10**10)])
    
    #smooth data
    kernel_size = 2
    kernel = np.ones(kernel_size) / kernel_size
    weights = np.convolve(weights, kernel)
    self.lossWeights = torch.Tensor(weights).to(self.device)


  def _calculate_loss(self, model, inputs, loss_fct, num_labels, args):
    outputs = model(**inputs)
    # model outputs are always tuple in pytorch-transformers (see doc)
    loss = outputs[0]
    if loss_fct:
        logits = outputs[1]
        labels = inputs["labels"]

        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

    if 1 == 1:
      # import pdb
      # pdb.set_trace()
      predict = outputs[1][:,0]
      labels = inputs["labels"]
      #myLoss = ((predict - labels)**2).mean()
      
      idx = ((labels+1)*100).long().to(self.device)
      myAdjustedLoss = (((predict - labels)**2) * self.lossWeights[idx]).mean()
      loss = myAdjustedLoss
      #print(f'loss:{ loss }, myLoss:{ myLoss }, myAdjustedLoss:{ myAdjustedLoss }')

    return (loss, *outputs[1:])