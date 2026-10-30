"""共享候选评分器的合成结构验证；不代表LLM或游戏王能力。"""
import argparse
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(1)
torch.manual_seed(42)
head=torch.nn.Sequential(torch.nn.Linear(3,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
optimizer=torch.optim.Adam(head.parameters(),lr=.03)

def batch(count,n):
    target=torch.rand(count,1)
    candidates=torch.rand(count,n)
    difference=(candidates-target).abs()
    features=torch.stack([candidates,target.expand_as(candidates),difference],-1)
    return features,difference.argmin(-1)

for step in range(300):
    features,labels=batch(64,3)
    optimizer.zero_grad()
    loss=torch.nn.functional.cross_entropy(head(features).squeeze(-1),labels)
    loss.backward()
    optimizer.step()
reports=[]
with torch.no_grad():
    for n in (2,3,5,9):
        features,labels=batch(512,n)
        scores=head(features).squeeze(-1)
        order=torch.randperm(n)
        reordered=head(features[:,order]).squeeze(-1)
        reports.append({'candidates':n,'accuracy':float((scores.argmax(-1)==labels).float().mean()),
            'fixed_first_accuracy':float((labels==0).float().mean()),
            'permutation_max_logit_error':float((reordered-scores[:,order]).abs().max()),
            'semantic_choice_consistency':float((order[reordered.argmax(-1)]==scores.argmax(-1)).float().mean())})
result={'scope':'合成数值任务，含人工绝对差特征；只验证共享评分结构，不证明LLM语义能力',
        'seed':42,'updates':300,'reports':reports}
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
