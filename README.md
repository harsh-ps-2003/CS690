# CS690

### Commands :

* Run `source /data1/cs690_env/bin/activate` at the root CS690
* To test GPUs
```
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```
* Use python --- whatver to run the scMODAL code!