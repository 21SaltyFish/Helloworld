import numpy as np
import pandas as pd
class BpNet:

    study_rate=0.1

    def __init__(self,input_x,hidden_x,output_x):
        self.input_x=input_x
        self.hidden_x=hidden_x
        self.output_x=output_x
        self.init_matrix()

    def get_input(self):#读数据


        data = pd.read_excel('datafile.xlsx')
        self.ip_layer=data.values
        '''
        s=input("请输入输入层数据及类别（使用空格隔开）:")
        value = s.split(' ')
        ip_layer = [float(i) for i in value]
        self.ip_layer= np.array(ip_layer).reshape(int(len(value)/(self.input_x+self.output_x)),self.input_x+self.output_x)'''
        self.ip_layer=self.ip_layer.transpose()
        self.label=self.ip_layer[-self.output_x:,:]
        self.ip_layer=np.delete(self.ip_layer,np.linspace(-1,-self.output_x,num=self.output_x,dtype=int),axis=0)

    def init_hidde(self):
        self.hid_layer=np.random.random([self.hidden_x,self.input_x])
        self.hid_layer_theta=np.random.random([self.hidden_x,1])

    def init_output(self):
        self.op_layer=np.random.random([self.output_x,self.hidden_x])
        self.op_layer_theta=np.random.random([self.output_x,1])

    def init_matrix(self):
        self.get_input()
        self.init_hidde()
        self.init_output()

    def sigmoid(self,matrix):
        return 1/(1+np.exp(-matrix))

    def print_result(self):
        self.calculate_result()

    def calculate_result(self):
       result_hid_befsig= np.dot(self.hid_layer,self.ip_layer)-self.hid_layer_theta
       self.hidde_out=self.sigmoid(result_hid_befsig)

       result_op_befsig=self.op_layer.dot(self.hidde_out)-self.op_layer_theta
       self.result=self.sigmoid(result_op_befsig)
       print(self.result)
       self.bias=self.result-self.label

    def derivative_sigmoid(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def correct_op_theta(self,number):
        self.deta_op_theta=self.bias[:,number].reshape(self.output_x,1)*self.derivative_sigmoid(self.result[:,number].reshape(self.output_x,1))
        self.op_layer_theta=self.op_layer_theta+BpNet.study_rate*self.deta_op_theta

    def correct_op_weight(self,number):
        self.deta_op_weight=self.deta_op_theta.dot(self.hidde_out[:,number].reshape(1,self.hidden_x))
        self.op_layer=self.op_layer-BpNet.study_rate*self.deta_op_weight

    def correct_hi_theta(self,number):
        self.deta_hi_theta=np.sum(self.deta_op_theta)*self.derivative_sigmoid(self.hidde_out[:,number].reshape(self.hidden_x,1))
        self.hid_layer_theta=self.hid_layer_theta+BpNet.study_rate*self.deta_hi_theta

    def correct_hi_weight(self,number):
        self.deta_hi_weight=self.deta_hi_theta.dot(self.ip_layer[:,number].reshape(1,self.input_x))
        self.hid_layer=self.hid_layer-BpNet.study_rate*self.deta_hi_weight

    def corret(self,num):
        self.correct_op_theta(num)
        self.correct_op_weight(num)
        self.correct_hi_theta(num)
        self.correct_hi_weight(num)

    def show(self):
        print(self.ip_layer)
        print(self.label)
        print(self.hid_layer)
        print(self.hid_layer_theta)
        print(self.op_layer)
        print(self.op_layer_theta)
