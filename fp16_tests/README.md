# Tests to see if half precision (fp16) is working

### Measure performance in notebook
Run measure_fp16.ipynb
The results of a v100 should be similar these:
https://discuss.pytorch.org/t/solved-titan-v-on-pytorch-0-3-0-cuda-9-0-cudnn-7-0-is-much-slower-than-1080-ti/11320/10

### Profile CUDA to make sure it is utilizing half precision
1. `pip install nvprof`
2. `/usr/local/cuda/bin/nvprof --log-file nvprof_output.txt python fp16_test.py`
3. `cat nvprof_output.txt | grep fp16_s884`

You should see some 884 calls