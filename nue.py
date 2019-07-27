#-*-coding:utf8-*-
import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes, outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化输入节点和隐藏节点间的权重
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))

        # 初始化隐藏节点和输出节点间的权重
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # 初始化生成S（激活）函数：y=1/(1+e^(-x))
        self.active_function = lambda x:scipy.special.expit(x)

    # 数据训练函数
    def train(self,input_list,target_list):
        inputs = numpy.array(input_list,ndmin=2).T

        # 期望输出数据
        targets = numpy.array(target_list,ndmin=2).T

        # 输入节点数据与输入和隐藏间权重相乘作为隐藏层的输入数据
        hidden_input = numpy.dot(self.wih,inputs)

        # 隐藏层的输入数据作为参数经过S函数后可得隐藏层的输出数据
        hidden_out = self.active_function(hidden_input)

        # 隐藏层的输出数据与隐藏和输出间权重相乘作为输出层的输入数据
        final_input = numpy.dot(self.who,hidden_out)

        # 输出层的输入数据作为参数经过S函数后可得输出层的输出数据，即最后输出数据
        final_out = self.active_function(final_input)

        # 计算期望数据与输出数据间的误差
        output_error = targets - final_out

        # 反向将误差值作为隐藏层的输入数据
        hiden_error = numpy.dot(self.who.T,output_error)

        # 更新三个节点间的权重，使输出数据更接近期望数据，根据误差函数的斜率分析进行，新权重公式为：ΔW=lr * E * O₂ * (1- O₂) · O₁,
        # 其中lr为学习率，E为误差， O₂  为最后输出数据，O₁为上个节点的输出数据
        self.who += self.lr * numpy.dot((output_error*final_out*(1-final_out)),numpy.transpose(hidden_out))
        self.wih += self.lr * numpy.dot((hiden_error * hidden_out * (1-hidden_out)) , numpy.transpose(inputs))

    # 查询函数，根据训练好的模型进行理想数据输出
    def query(self,input_list):
        inputs = numpy.array(input_list,ndmin=2).T
        hidden_input = numpy.dot(self.wih,inputs)
        hidden_out = self.active_function(hidden_input)
        final_input = numpy.dot(self.who,hidden_out)
        final_out = self.active_function(final_input)
        return final_out

if __name__ == '__main__':
    recognition_rate = input('请输入期望的识别率：')
    #初始识别率
    r_rate = 0


    # 读取训练集数据
    with open('train_set_10000.csv') as f:
        train_data_list = f.readlines()
    # 读取测试集数据
    with open('test_set_1000.csv') as f:
        test_data_list = f.readlines()

    # 实例化一个神经网络对象，输入数据数值一般根据该图片的像素进行确定
    n=NeuralNetwork(784,100,10,0.1)

    while r_rate < recognition_rate:

        for line in train_data_list:
            all_values = line.split(',')
            # 将输入数据转化为小于1的形式，方便激活函数的计算
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001

            # 将对应的期望数据转为对应0.01-0.09的列表
            targets = numpy.zeros(10)+0.001
            targets[int(all_values[0])] = 0.999

            # 开始训练模型
            n.train(inputs,targets)

        #开始测试数据并获取识别率
        right_list = []
        for line in test_data_list:
            all_values = line.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.999) + 0.001
            out = n.query(inputs)
            # 获取该列表中最大数值的索引
            label = numpy.argmax(out)
            if int(all_values[0]) == label:
                right_list.append(1)
            else:
                right_list.append(0)

        r_rate = float( right_list.count(1))/float(len(right_list))
        print '当前识别率为：%f'%(r_rate)