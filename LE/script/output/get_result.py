import os
import json
import shutil
import argparse
import random
import my_data_util
from my_data_util import ClothSample
import numpy as np
import torch
import time
from pytorch_pretrained_bert.modeling import BertForCloth
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import functools

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def answer_trans(a):
    r=[]
    for i in a:
        r.append(chr(i+ord("A")))
    # result=str(r).replace("'","\"")
    return r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

data_dir='./test_data/test.pt'
save_dir='./answers/LE_test.json'
cache_size=256
test_batch_size=1
test_data = my_data_util.Loader(data_dir, cache_size, test_batch_size, device)
print(data_dir+' loaded')

# Load target model
model_dir='./model.pt'
model=torch.load(model_dir)
model.to(device)
model = torch.nn.DataParallel(model)
model.eval()
print(model_dir+' loaded')

# Get predict result
result={}
for inp, tgt, name in test_data.data_iter(shuffle=False):
    with torch.no_grad():
        out, _, _, _, _ = model(inp, tgt)
    out = np.array(torch.argmax(out, -1).cpu())

    name=name[-13:-5]
    if name in result:
        print(name)
        carry=result[name]
        result[name]=carry+answer_trans(out)
    else:
        result[name]=answer_trans(out)

carry=sorted(result)
final={}
for i in carry:
    final[i]=result[i]

# Write into json file
j_data=json.dumps(final, indent=4)

file_obj=open(save_dir, 'w')
file_obj.write(j_data)
file_obj.close()
print('Convert to result successfully')