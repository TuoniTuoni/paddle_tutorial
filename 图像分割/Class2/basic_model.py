#basic_model.py

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
import numpy as np
np.set_printoptions(precision=2)

class BasicModel(fluid.dygraph.Layer):
    def __init__(self,num_classes=59):
        super(BasicModel,self).__init__()
        self.pool=Pool2D(pool_size=2,pool_stride=2)
        #self.conv=Conv2D(num_channels=3,num_filters=1,filter_size=1)#输入3个维度，输出1个维度..通常conv默认padding是0，stride是1
        self.conv=Conv2D(num_channels=3,num_filters=num_classes,filter_size=1)
    def forward(self,inputs):
        x=self.pool(inputs)
        x=fluid.layers.interpolate(x,out_shape=inputs.shape[2::])
        x=self.conv(x)
    
        return x
   

def main():
    place=paddle.fluid.CPUPlace()
    #place=paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model=BasicModel(num_classes=59)
        model.eval()#model.train()
        input_data=np.random.randn(1,3,8,8).astype(np.float32)#np_array_type
        print(input_data)
        print("Input data shape:",input_data.shape)
        input_data=to_variable(input_data)#转换为paddle的tensor类型
        output_data=model(input_data)#output也为tensor类型
        #print(output_data)
        output_data=output_data.numpy()#转换为numpy的array
        print('Output data shape:',output_data.shape)

if __name__=="__main__":
    main()