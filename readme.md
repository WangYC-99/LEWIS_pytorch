# LEWIS reconstruct pytorch code

> by WangYC
>
> @NWPU @Zhipu.AI

## usage

### train
```
python roberta-classifier.py
```

### test
cmd:
```
python roberta-classifier.py --ckpt_path <ckpt_path> --instruction test
```

### infer
cmd:
```
python roberta-classifier.py --ckpt_path <ckpt_path> --instruction infer
# e.g. python roberta-classifier.py --ckpt_path /zhangpai25/wyc/lewis/lewis_wyc/saved_ckpts/1-11-13-39/best.pkl --instruction infer
```
outputs:
```
please input the sentence: (type END to exit) : 妹妹你倒是评评理
prediction result:tensor([[0.6786, 0.3214]], grad_fn=<SoftmaxBackward0>), cla_result:红楼风格
please input the sentence: (type END to exit) : 特朗普竞选美国总统
prediction result:tensor([[0.2946, 0.7054]], grad_fn=<SoftmaxBackward0>), cla_result:普通风格
please input the sentence: (type END to exit) :
```

## origin paper & repo

Reid, Machel, and Victor Zhong. “LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer.” In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, 3932–44. Online: Association for Computational Linguistics, 2021. https://doi.org/10.18653/v1/2021.findings-acl.344.

https://github.com/machelreid/lewis