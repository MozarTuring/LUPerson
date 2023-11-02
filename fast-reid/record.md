WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b1_head.classifier.weight' has shape (751, 256
) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b2_head.classifier.weight' has shape (751, 256
) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b21_head.classifier.weight' has shape (751, 25
6) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b22_head.classifier.weight' has shape (751, 25
6) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b3_head.classifier.weight' has shape (751, 256
) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b31_head.classifier.weight' has shape (751, 25
6) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b32_head.classifier.weight' has shape (751, 25
6) in the checkpoint but (0, 256) in the model! Skipped.
WARNING [06/06 15:08:18 fastreid.utils.checkpoint]: 'b33_head.classifier.weight' has shape (751, 25
6) in the checkpoint but (0, 256) in the model! Skipped.

[06/06 15:08:18 fastreid.utils.checkpoint]: Some model parameters are not in the checkpoint:
  b1_head.classifier.weight
  b2_head.classifier.weight
  b21_head.classifier.weight
  b22_head.classifier.weight
  b3_head.classifier.weight
  b31_head.classifier.weight
  b32_head.classifier.weight
  b33_head.classifier.weight


* mgn.py 136 行, self.b1_head 在非训练阶段 forward 的时候不需要 b1_head.classifier，所以以上参数不需要load
* 另外需要注意的是 self.b1_head 的输入shape 是 [128, 2048, 24, 8] ， 输出shape是 [128, 256]


* fast-reid/fastreid/data/build.py 里的 test_set = CommDataset(test_items, test_transforms, relabel=False) 里的 __getitem__  可以看到对数据的处理
这个 getitem 里如果用read_image的返回值作为返回字典"images"键的值,那么实际load出来的会是None

* pre_models/market.pth
  * 原始模型测试结果
  [06/08 16:35:08 fastreid.evaluation.testing]: Evaluation results in csv format:
  | Datasets   | Rank-1   | Rank-5   | Rank-10   | mAP    | mINP   |
  |:-----------|:---------|:---------|:----------|:-------|:-------|
  | CMDM       | 96.26%   | 98.93%   | 99.41%    | 91.08% | 69.62% |
  * onnx 测试结果
  [06/08 16:31:12 fastreid.evaluation.testing]: Evaluation results in csv format:
  | Datasets   | Rank-1   | Rank-5   | Rank-10   | mAP    | mINP   |
  |:-----------|:---------|:---------|:----------|:-------|:-------|
  | CMDM       | 96.26%   | 98.93%   | 99.41%    | 91.04% | 69.56% |

  
* 从 fast-reid/fastreid/evaluation/evaluator.py 的 results = evaluator.evaluate() 可以看到对模型结果的后处理

[06/09 19:23:01 fastreid.data.common]: Compose(
297     Resize(size=[384, 128], interpolation=PIL.Image.BICUBIC)
298     ToTensor()
299 )

ipdb> self.transform
Compose(
    Resize(size=[384, 128], interpolation=PIL.Image.BICUBIC)
    ToTensor()
)


body_feature
'datasets/market1501/query/0001_c1s1_001051_00.jpg'


lup
image
[[116, 115, 113],
        [110, 109, 107],
        [105, 104, 102],
        ...,
        [108, 107, 105],
        [106, 105, 103],
        [108, 107, 105]],

       [[104, 103, 101],
        [ 97,  96,  94],
        [ 91,  90,  88],
        ...,
        [ 97,  96,  94],
        [107, 106, 104],
        [122, 121, 119]]], dtype=uint8)

after transform
 [[ 64.,  65.,  68.,  ..., 128., 109.,  99.],
         [ 64.,  65.,  68.,  ..., 128., 110., 101.],
         [ 65.,  66.,  68.,  ..., 128., 113., 106.],
         ...,
         [107., 105., 102.,  ..., 109., 115., 118.],
         [104., 102.,  98.,  ..., 110., 118., 122.],
         [103., 101.,  97.,  ..., 111., 119., 124.]],

        [[ 69.,  69.,  71.,  ..., 119.,  99.,  90.],
         [ 69.,  69.,  71.,  ..., 119., 101.,  92.],
         [ 70.,  70.,  71.,  ..., 119., 104.,  97.],
         ...,
         [105., 103., 100.,  ..., 107., 113., 116.],
         [102., 100.,  96.,  ..., 108., 116., 120.],
         [101.,  99.,  95.,  ..., 109., 117., 122.]]])
