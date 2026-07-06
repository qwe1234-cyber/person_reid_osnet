\# OSNet Person Re-Identification System



本项目基于 ICCV 2019 论文 Omni-Scale Feature Learning for Person Re-Identification，实现了一个行人再认证（Person Re-Identification, ReID）模式识别系统。



\## 项目功能



\- Market-1501 数据集读取与预处理

\- OSNet 模型训练

\- Rank-k 和 mAP 测试评估

\- Query-Gallery Top-K 检索

\- Gradio 网页演示程序

\- Re-ranking 改进测试



\## 数据集



数据集目录：



data/market1501/Market-1501-v15.09.15/



主要包含：



bounding\_box\_train

bounding\_box\_test

query



由于 Market-1501 数据集体积较大，本项目不包含数据集文件。请自行下载并放置到上述目录。



\## 项目结构



person\_reid\_osnet/

├── app/

│   └── demo\_gradio.py

├── scripts/

│   ├── train\_osnet.py

│   ├── evaluate\_osnet.py

│   ├── evaluate\_osnet\_rerank.py

│   └── reid\_search.py

├── requirements.txt

├── start\_demo.sh

└── README.md



\## 主要文件



scripts/train\_osnet.py：训练脚本

scripts/evaluate\_osnet.py：测试评估脚本

scripts/evaluate\_osnet\_rerank.py：Re-ranking 改进测试脚本

scripts/reid\_search.py：Top-K 检索脚本

app/demo\_gradio.py：Gradio 网页演示程序

start\_demo.sh：一键启动演示脚本



\## 环境安装



安装项目依赖：



pip install -r requirements.txt



如果环境中尚未安装 deep-person-reid / torchreid，需要先安装开源框架：



git clone https://github.com/KaiyangZhou/deep-person-reid.git

cd deep-person-reid

python setup.py develop

cd ..



\## 训练模型



python scripts/train\_osnet.py --epochs 60 --batch-size 64



\## 测试模型



python scripts/evaluate\_osnet.py --weights results/osnet\_train\_60epoch/model/model.pth.tar-60



\## 单张图片检索



QUERY=$(find data/market1501/Market-1501-v15.09.15/query -name "\*.jpg" | head -n 1)

python scripts/reid\_search.py --query "$QUERY"



检索结果默认保存在：



results/retrieval\_examples/retrieval\_result.jpg



\## 启动网页演示



bash start\_demo.sh



启动后，打开控制台显示的 Gradio 访问地址即可使用网页演示程序。



如果在 AutoDL 环境中运行，可打开 AutoDL 控制台中 6006 端口对应的访问链接。



\## 当前实验结果



基础 OSNet 测试结果：



mAP：44.9%

Rank-1：67.6%

Rank-5：86.3%

Rank-10：91.1%

Rank-20：94.6%



\## Re-ranking 改进实验



本项目在基础 OSNet 检索结果上加入了 k-reciprocal Re-ranking 后处理方法。



Re-ranking 不需要重新训练模型，而是在测试阶段利用 query-gallery 和 gallery-gallery 之间的邻居关系重新调整检索排序。



运行 Re-ranking 测试：



python scripts/evaluate\_osnet\_rerank.py --weights results/osnet\_train\_60epoch/model/model.pth.tar-60



Re-ranking 测试结果：



mAP：63.6%

Rank-1：72.3%

Rank-5：83.5%

Rank-10：88.6%

Rank-20：92.2%



对比普通 OSNet 结果，Re-ranking 将 mAP 从 44.9% 提升到 63.6%，Rank-1 从 67.6% 提升到 72.3%。说明 Re-ranking 能够明显改善整体检索排序质量。



\## 快速运行说明



由于 Market-1501 数据集、deep-person-reid 框架以及运行环境体积较大，本项目不打包为 exe 可执行程序。



1\. 解压项目



tar -zxvf person\_reid\_osnet\_package.tar.gz

cd person\_reid\_osnet



2\. 安装依赖



pip install -r requirements.txt



如未安装 deep-person-reid / torchreid，执行：



git clone https://github.com/KaiyangZhou/deep-person-reid.git

cd deep-person-reid

python setup.py develop

cd ..



3\. 准备数据集



将 Market-1501 数据集放到以下目录：



data/market1501/Market-1501-v15.09.15/



目录中应包含：



bounding\_box\_train

bounding\_box\_test

query



4\. 测试模型



python scripts/evaluate\_osnet.py --weights results/osnet\_train\_60epoch/model/model.pth.tar-60



5\. 运行 Re-ranking 改进测试



python scripts/evaluate\_osnet\_rerank.py --weights results/osnet\_train\_60epoch/model/model.pth.tar-60



6\. 单张图片检索



QUERY=$(find data/market1501/Market-1501-v15.09.15/query -name "\*.jpg" | head -n 1)

python scripts/reid\_search.py --query "$QUERY"



检索结果保存在：



results/retrieval\_examples/retrieval\_result.jpg



7\. 启动网页演示程序



bash start\_demo.sh



\## 注意事项



\- 本仓库不包含 Market-1501 数据集。

\- 本仓库不建议上传模型权重文件。

\- 如需运行测试，请自行准备模型权重文件，并放置到：



results/osnet\_train\_60epoch/model/model.pth.tar-60



\- 如果使用 GitHub 上传项目，建议在 .gitignore 中忽略以下内容：



data/

datasets/

results/

weights/

checkpoints/

\_\_pycache\_\_/

\*.pyc

\*.pth

\*.pt

\*.onnx

\*.pkl

\*.tar-\*

.env



\##License



本项目仅用于课程学习、模式识别实验和行人再认证方法研究。

