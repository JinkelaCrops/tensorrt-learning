# 使用tensorrt进行infer

## tensorrt下载

nvidia官网链接[nvidia-tensorrt-download](http://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/4.0/rc1/TensorRT-4.0.0.3.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.0.tar.gz)

解压到`/usr/local/TensorRT4`, 然后在`~/.bashrc`中添加`export LD_LIBRARY_PATH=/usr/local/TensorRT4/lib:$LD_LIBRARY_PATH`, 并把`/usr/local/TensorRT4/python/tensorrt-4.0.0.3-cp35-cp35m-linux_x86_64.whl`和`/usr/local/TensorRT4/uff/uff-0.3.0-py2.py3-none-any.whl`安装到自己的`python`虚拟环境.

```python
import tensorrt as trt
import tensorflow as tf
```
测试应当可以通过.

## tensorrt教程(sample nmt)

nvidia深度学sdk文档[3.11. SampleNMT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#nmt_sample)

```shell
$ mkdir ~/wmt16
$ cd ~/wmt16
$ git clone https://github.com/tensorflow/nmt
$ cp ~/wmt16/nmt/nmt/scripts/wmt16_en_de.sh ~/wmt16/wmt16_en_de.sh
$ bash wmt16_en_de.sh
```
自动下载, 生成BPE等, 需要一段时间

下载Nvidia训练好的`weights`
```shell
$ vim /usr/local/TensorRT4/samples/sampleNMT/README.txt
$ wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/models/sampleNMT_weights.tar.gz
$ tar -xvf sampleNMT_weights.tar.gz samples/nmt/deen/weights/
$ cp wmt16_de_en/newstest2015.tok.bpe.32000.de samples/nmt/deen
$ cp wmt16_de_en/newstest2015.tok.bpe.32000.en samples/nmt/deen
$ cp wmt16_de_en/vocab.bpe.32000.de samples/nmt/deen
$ cp wmt16_de_en/vocab.bpe.32000.en samples/nmt/deen
```

要使用[3.11.3. Running The Sample](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#nmt_run)首先要`make` `simpleNMT`

修改`/usr/local/TensorRT4/samples/Makefile.config`

```shell
CUDA_INSTALL_DIR?=/usr/local/cuda
CUDNN_INSTALL_DIR?=/usr/local/cuda
```
这里`/usr/local/cuda`是`cuda`和`cudnn`的安装位置

```shell
$ cd /usr/local/TensorRT4/samples/sampleNMT
$ sudo make clean
$ sudo make
```
等待结束

```shell
$ cd /usr/local/TensorRT4/bin
$ ./sample_nmt --help
$ ./sample_nmt --data_dir=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/samples/nmt/deen --data_writer=text
```

可以看到生成了`translation_output.txt`
```shell
$ less translation_output.txt
```

```shell
$ ./sample_nmt --data_dir=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/samples/nmt/deen --data_writer=bleu

data_dir: /media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/samples/nmt/deen
data_writer: bleu
Component Info:
- Data Reader: Text Reader, vocabulary size = 36548
- Input Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
- Output Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
- Encoder: LSTM Encoder, num layers = 2, num units = 512
- Decoder: LSTM Decoder, num layers = 2, num units = 512
- Alignment: Multiplicative Alignment, source states size = 512, attention keys size = 512
- Context: Ragged softmax + Batch GEMM
- Attention: SLP Attention, num inputs = 1024, num outputs = 512
- Projection: SLP Projection, num inputs = 512, num outputs = 36548
- Likelihood: Softmax Likelihood
- Search Policy: Beam Search Policy, beam = 5
- Data Writer: BLEU Score Writer, max order = 4
End of Component Info
BLEU score = 10.5962
```
```shell
$ ./sample_nmt --data_dir=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/samples/nmt/deen --data_writer=benchmark

data_writer: benchmark
Component Info:
- Data Reader: Text Reader, vocabulary size = 36548
- Input Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
- Output Embedder: SLP Embedder, num inputs = 36548, num outputs = 512
- Encoder: LSTM Encoder, num layers = 2, num units = 512
- Decoder: LSTM Decoder, num layers = 2, num units = 512
- Alignment: Multiplicative Alignment, source states size = 512, attention keys size = 512
- Context: Ragged softmax + Batch GEMM
- Attention: SLP Attention, num inputs = 1024, num outputs = 512
- Projection: SLP Projection, num inputs = 512, num outputs = 36548
- Likelihood: Softmax Likelihood
- Search Policy: Beam Search Policy, beam = 5
- Data Writer: Benchmark Writer
End of Component Info
2169 sequences generated in 9.56708 seconds, 226.715 samples/sec
```