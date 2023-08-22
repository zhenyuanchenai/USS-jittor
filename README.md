| 第三届计图人工智能挑战赛

| Jittor 大规模无监督语义分割 USS

| 大规模无监督语义分割是计算机视觉领域的一个活跃研究领域，在自动驾驶、遥感、医学成像和视频监控等领域有许多潜在的应用，其涉及不使用有标签训练数据的情况下自动将图像内的相似区域或对象分组在一起。该任务的目标是生成一个语义分割图，将图像中的每个像素分配给特定的语义类别，例如“车辆”、“建筑”或“天空”等。

| 环境配置
首先，安装jittor环境（两种选择）：
1.通过conda安装环境
conda env create -f requirement.yaml 
2.直接复制已有的conda环境包到自己的conda目录。
cd /USS-jittor/weights
unzip jittor.zip 
mv /USS-jittor/weights/jittor conda目录/evns/
修改自己电脑的conda环境变量

注: 环境晚装说明
本代码使用的环境是基于python==3.7.12,  cude=10.2,  cudnn=v8.0.4, 请根据自己的显卡配置安装对应版本cuda和cudnn，并修改某些包的版本情况。

测试B榜结果：
cd /USS-jittor/
一个GPU推理：
CUDA_VISIBLE_DEVICES='0' python test.py

注：测试说明
将在USS-jittor这一根目录下创建result文件夹，其中包含所有testB测试结果和result.zip文件。
默认的数据集路径为：data_path/mode，请在test.py文件中修改自己的数据集。其中data_path为ImageNetS50的路径，mode=testB。
预训练模型存放在./weight/checkpoint.pth.tar
因为sam模型需要占用很大显存，可能会出现显存不足占用内容的情况（此时可能会调用服务器内存，是正常现象不用担心），并且可能出现测试程序终止的情况，或者运行过程中卡住不动的情况（很久没有下一张图片的推理时，请停止程序重新运行，会接在上一次生成后继续生成），因此请多次运行test.py直至测试结束,两个代码的运行结果可以共享，在保存数据集路径相同的情况下。


训练代码：
训练代码共有四份，依次运行bash train_1.sh、bash train_2.sh、bash train_3.sh、bash train_4.sh

注：为了减轻训练时间过长带来的影响，提供中间过程的模型训练权重（不需要执行bash train_1.sh和bash train_2.sh）。
train_3.sh中./weights/pass50/pixel_attention/checkpoint.pth.tar可替换为./weights/infer_pixel_attnetion_checkpoint.pth.tar
train_3.sh中./weights/pass50/pixel_attention/cluster/centroids.npy可替换为./weights/centroids.npy
train_4.sh中./weights/pass50_resnet50/checkpoint.pth.tar可替换为./weights/train_pixelattention_checkpoint.pth.tar
train_4.sh运行完毕后将生成B榜提交结果，保存在./weights/pass50_sam_upper50/pixel_finetuning/testB中。
