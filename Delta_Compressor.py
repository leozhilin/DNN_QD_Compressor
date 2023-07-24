import copy
import gzip
import pickle
def Compressor(state_dict, compressed_file_name):
    """
    This is a function which can compress state dict of model.

    Parameters:
     param1 - model's state dict
     param2 - saved path of compressed file

    Returns:
     none
    """
    # 提取模型参数
    model_parameters = state_dict
    # 将模型参数转换为字节数据
    parameters_bytes = pickle.dumps(model_parameters)
    # 压缩模型参数
    compressed_parameters = gzip.compress(parameters_bytes)
    # 保存压缩后的模型参数到文件
    with open(compressed_file_name, 'wb') as f:
        f.write(compressed_parameters)
def Decompressor(model, decompressed_file_name):
    with open(decompressed_file_name, 'rb') as f:
        compressed_parameters = f.read()
    # 解压缩模型参数
    decompressed_parameters = gzip.decompress(compressed_parameters)
    # 将解压缩后的字节数据转换回模型参数
    model_parameters = pickle.loads(decompressed_parameters)
    #将解压后的参数载入到模型中
    state_dict = model_parameters
    model.load_state_dict(state_dict)

def Delta_Calculator(modelx, modely):
    delta = copy.deepcopy(modelx)
    for i, (delta_param, paramx, paramy) in enumerate(zip(delta.parameters(), modelx.parameters(), modely.parameters())):
        delta_param.data = (paramx.data - paramy.data) % 2**8
        # print(paramy.data)
    return delta

def Delta_Restore(modelx, delta):
    modely = copy.deepcopy(modelx)
    for i, (delta_param, paramx, paramy) in enumerate(zip(delta.parameters(), modelx.parameters(), modely.parameters())):
        paramy.data = (paramx.data + delta_param.data) % 2**8
        print(paramy.data)
    return modely

def QD_Compressor(quantized_model_last, quantized_model_current, path):
    """
    This is a function which can compress delta of neighbor-version models and save compressed file by path.

    Parameters:
     param1 - last quantized model
     param2 - current quantized model
     param3 - save path of compressed file

    Returns:
     none
    """
    print("Compressing...")
    delta = Delta_Calculator(quantized_model_current, quantized_model_last)
    Compressor(delta, compressed_file_name = path)