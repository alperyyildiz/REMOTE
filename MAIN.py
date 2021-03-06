  
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time

from torch import optim
from sklearn import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
cuda = torch.device('cuda')

print('IMPORTS ARE DONE!!!')




class PARAMETERS():
    
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

        
    def GET_OTHERS(self):
        self.OTHERS  =  {
                        'windowlength': 24,
                        'out_size': 3,
                        'period': 24,
                        'lrate': 0.0005,
                        'batchsize': 32,
                        'epoch': 100
                        }
        
    def GET_DICT(self):
        self.DICT = {'CONV': {
                  '1': {'FIL': 128, 
                        'KER': 8,
                        'dropout': [True, 0.5],
                        'batchnorm': True
                       },
                  
                  '2': {'FIL': 64, 
                        'KER': 4,
                        'dropout': [True, 0.2],
                        'batchnorm': True
                       }
                             },
         
                  'flatten': {'1': {'nofilter':0 , 'nonothing':0 }},
         
                  'DENSE': {
                  
                  '1': {'FIL': 48,
                        'dropout' : [True,0.3]
                       },
                  '2': {'FIL':self.OTHERS['out_size']}
                  }
        }
        
        
def SEARCH_SPACE(self):
  space = {}
  TO_CHNG = {'CON': {'1': {'KER': (2,14),
                          'dropout': (0.2,0.8)},
                    '2': {'KER': (2,13),
                          'dropout': (0.2,0.8)}
                    },
            'DENSE': {'1':{ 'FIL': (32,256),
                            'dropout':(0.2,0.6)}
                      }
            }
  HYP_DICT ={}
  d_count = 0
  for TYPE in  list(TO_CHNG.keys()):
    HYP_DICT_LAYER = {}
    for layernum in list(TO_CHNG[TYPE].keys()):
      HYP_DICT_PARAMS = {}
      for PARAM in list(TO_CHNG[TYPE][layernum].keys()):
        if PARAM == 'dropout':
          d_count = d_count + 1
          HYP_DICT_PARAMS[PARAM + str(d_count)] = hp.uniform(PARAM + str(d_count),TO_CHNG[TYPE][layernum][PARAM][0],TO_CHNG[TYPE][layernum][PARAM][1])
        else:
          HYP_DICT_PARAMS[PARAM + layernum] = hp.uniform(PARAM + layernum,TO_CHNG[TYPE][layernum][PARAM][0],TO_CHNG[TYPE][layernum][PARAM][1])
      HYP_DICT_LAYER[layernum] =  HYP_DICT_PARAMS
    HYP_DICT[TYPE] =  HYP_DICT_LAYER

  self.space = hp.choice('paramz', [HYP_DICT])

    #CREATE SUBDIR OF ABOVE NAMED WITH 
    #EXPERIMENT DATE AND START TIME
    def CREATE_DIR(self):
        first_Con = True
        SAVE_DIR = 'storage/'

        for con in range(len(self.dict['CON']['list'])):
            if first_Con:
                SAVE_DIR = SAVE_DIR +'CON_' + str(self.dict['CON']['list'][con])
                first_Con = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['CON']['list'][con])

        first_LS = True

        for ls in range(len(self.dict['LST']['list'])):
            if first_LS:
                SAVE_DIR = SAVE_DIR + '_LS_' + str(self.dict['LST']['list'][ls])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['LST']['list'][ls])

        first_DEN = True

        for dense in range(len(self.dict['DEN']['list'])):
            if first_DEN:
                SAVE_DIR = SAVE_DIR + '_D_' + str(self.dict['DEN']['list'][dense])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['DEN']['list'][dense])
        try: 
            os.mkdir(SAVE_DIR)
            self.save_DIR = SAVE_DIR
        except:
            pass
        
        SAVE_DIR = SAVE_DIR + '/' +str(datetime.now())[:-10]
        SAVE_DIR = list(SAVE_DIR)
        SAVE_DIR[-6] = '--'
        SAVE_DIR = ''.join(SAVE_DIR)
        self.save_DIR = SAVE_DIR
        try:
            os.mkdir(SAVE_DIR)
        except:
            pass

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self):
        first_con_f = True
        save_NAME = ''
        key_VAR = ''
        for Layer_TYP in list(self.VARS_EX.keys()):
            key_VAR = key_VAR + '\n'
            for kk,layer_NUM in enumerate(list(self.VARS_EX[Layer_TYP].keys())):
                if kk==0:
                    key_VAR = key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = save_NAME + '---'+  Layer_TYP[0]  + layer_NUM
                else:
                    key_VAR = '---' + key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = '---' + save_NAME + '---'+  Layer_TYP[0]  + layer_NUM                
                VALUES = '\n'
                save_VALUES = ''
                for VAR in list(self.VARS_EX[Layer_TYP][layer_NUM].keys()):
                        print(str(self.dict[Layer_TYP][layer_NUM][VAR]))
                        save_VALUES = save_VALUES  +  Layer_TYP[0]  + layer_NUM + '-' + VAR + '-' + str(self.dict[Layer_TYP][layer_NUM][VAR]) + '---'
                        VALUES = VALUES +  VAR + ' = ' +  str(self.dict[Layer_TYP][layer_NUM][VAR]) + '\t'
        key_VAR = key_VAR + VALUES
        save_NAME_PRED = save_NAME + 'PRED_' + save_VALUES
        save_NAME = save_NAME + save_VALUES
        return key_VAR, save_NAME, save_NAME_PRED
    
    #SAVE CONSTANT HYPERPARAMETERS OF EXPERIMENT AS TXT
    def WRITE_CONSTANTS(self):
        first_con_f = True
        SAVE_CON = ''
        key_CONST = ''
        for lt, Layer_TYP in enumerate(list(self.dict.keys())):
            if len(self.dict[Layer_TYP]['list']) > 0:
                if Layer_TYP != 'OTHERS':
                    for i,LAYER_NUM in enumerate(list(self.dict[Layer_TYP]['list'])):
                        if LAYER_NUM != 'list':
                            key_CONST = key_CONST[:-3] + ' \n '
                            key_CONST = key_CONST + Layer_TYP[0] + LAYER_NUM + ': '
                            for var in list(self.dict[Layer_TYP][LAYER_NUM].keys()):
                                if Layer_TYP in self.VARS_EX.keys() and LAYER_NUM in self.VARS_EX[Layer_TYP].keys():
                                    if var not in list(self.VARS_EX[Layer_TYP][LAYER_NUM]):
                                        key_CONST  = key_CONST + var + ': ' + str(self.dict[Layer_TYP][LAYER_NUM][var]) + ' -- '
                                else:
                                    key_CONST  = key_CONST + var + ': ' + str(self.dict[Layer_TYP][LAYER_NUM][var]) + ' -- '
        key_CONST = key_CONST + '\n'
        for other_KEY in self.dict['OTHERS']['1']:
            if other_KEY not in (self.VARS_EX['OTHERS']['1'].keys()):
                key_CONST = key_CONST + other_KEY + ': ' + str(self.dict['OTHERS']['1'][other_KEY]) + '\n'
        self.key_CONST = key_CONST
        save_NAME_CONST = self.save_DIR + '/CONSTANT_HyperParameters.txt'
        text_file = open(save_NAME_CONST , 'w')
        text_file.write(self.key_CONST)
        text_file.close()

    #SAVES HIST AND PRED PLOTS
    def SAVE_PLOTS(self):
        key_VAR, save_NAME, save_NAME_PRED = self.CREATE_SAVE_NAME()
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(key_VAR)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( self.save_DIR + '/' + save_NAME + '.png')
        
        fig2 = self.plotz()
        plt.savefig(self.save_DIR + '/' + save_NAME_PRED + '.png')
        plt.close('all')
        self.keyz.append(save_NAME)

    
    #SET INITIAL MODEL CHANGING PARAMETERS AS FIRST OF THEIR LIST VALUES
    def dict_UPDATE(self):
        for param in list(mm.VARS_EX.keys()):
            for sec_param in list(self.VARS_EX[param].keys()):
                for VAR in list(self.VARS_EX[param][sec_param].keys()):
                    self.dict[param][sec_param][VAR] = self.VARS_EX[param][sec_param][VAR][0]


        
    def GET_ALL(self):
        self.GET_OTHERS()
        self.GET_DICT()
        self.preprocess(split = 220)
        self.DICT_TO_LIST()

    def DICT_TO_LIST(self):
        prev_out_ch = 0
        self.LIST = list()
        self.seq_len_left = self.OTHERS['windowlength']
        for tt, key in enumerate(list(self.DICT.keys())):
            if key == 'flatten':
                self.LIST.append([key, ['nothing','nothing']])
                prev_out_ch = prev_out_ch * self.seq_len_left
            elif key != 'OTHERS':
                for ttt, layer in enumerate(list(self.DICT[key].keys())):
                    p_list = list()
                    for tttt, param in enumerate(list(self.DICT[key][layer].keys())):
                        if param not in ['dropout','batchnorm']:
                            if param == 'KER':
                                self.seq_len_left = self.seq_len_left - self.DICT[key][layer][param] + 1
                            if tt is 0 and ttt is 0:
                                if tttt is 0:
                                    p_list.append(self.featuresize)       
                                p_list.append(self.DICT[key][layer][param])
                            else:
                                if tttt is 0:
                                    p_list.append(prev_out_ch)       
                                p_list.append(self.DICT[key][layer][param])
                                
                    self.LIST.append([key, p_list])
                    prev_out_ch = p_list[1]
                    if 'batchnorm' in list(self.DICT[key][layer].keys()) and self.DICT[key][layer]['batchnorm']:
                        self.LIST.append(['batchnorm',[prev_out_ch,True]])

                    if 'dropout' in list(self.DICT[key][layer].keys()) and self.DICT[key][layer]['dropout'][0]:
                        self.LIST.append(['dropout',[self.DICT[key][layer]['dropout'][1],False]])
                
        
        
    def preprocess(self,split):
        data = pd.read_excel('clean.xlsx').dropna()
        windowlength = self.OTHERS['windowlength']
        outsize = self.OTHERS['out_size']
        arr = np.asarray(data['sales'])
        vv =pd.read_csv('vix.csv',sep=',')

        vix = np.array(vv['Şimdi'])
        for i in range(len(vix)):
            vix[i] = float(vix[i].replace(',','.'))

        dol =pd.read_csv('dollar.csv',sep=',')
        dollars = np.array(dol['Şimdi'])
        for i in range(len(dollars)):
            dollars[i] = float(dollars[i].replace(',','.'))
            
            
        res = STL(arr,period = self.OTHERS['period'] ,seasonal = 23 , trend = 25).fit()
        observed = res.observed
        a = np.concatenate([np.array(res.observed).reshape(res.observed.shape[0],1),np.array(res.seasonal).reshape(observed.shape[0],1),np.array(res.trend).reshape(observed.shape[0],1),np.array(res.resid).reshape(observed.shape[0],1).reshape(observed.shape[0],1),np.array(vix).reshape(observed.shape[0],1),np.array(dollars).reshape(observed.shape[0],1)],axis=1)
        dataz = np.swapaxes(np.array([res.observed,res.seasonal,res.trend,res.resid,vix,dollars]),0,1)
        train = dataz[:split]
        test = dataz[split:]
                
        MAX_window = self.OTHERS['windowlength']
        scaler = StandardScaler()
        sclr = scaler.fit(train)
        train =  scaler.transform(train)
        test =  scaler.transform(test)
        
        self.scaler.fit(arr[:split].reshape(-1,1))
        TR_OUT = np.asarray([[np.array(train[:,0])[i+k+windowlength] for i in range(outsize)] for k in range(split - outsize - MAX_window)])
        for feat in range(train.shape[1]):
            if feat == 0:
                TR_INP = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
            else:
                TR_new = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
                TR_INP = np.concatenate((TR_INP,TR_new),axis=1)

        TST_OUT = np.asarray([[np.array(test[:,0])[i+k+windowlength] for i in range(outsize)] for k in range(len(arr) - split - outsize - windowlength)])
        for feat in range(test.shape[1]):
            if feat == 0:
                TST_INP = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
            else:
                TST_new = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
                TST_INP = np.concatenate((TST_INP,TST_new),axis=1)

        #TR_INP = np.swapaxes(TR_INP,1,2)
        #TST_INP = np.swapaxes(TST_INP,1,2)
        self.pagez = test.shape[0]-outsize-windowlength
        self.test_actual = self.scaler.inverse_transform(TST_OUT)
        self.featuresize = TR_INP.shape[1]
        
        
        TR_INP = torch.Tensor(TR_INP).to(device = cuda)
        TST_INP = torch.Tensor(TST_INP).to(device = cuda)
        TR_OUT = torch.Tensor(TR_OUT).to(device = cuda)
        TST_OUT = torch.Tensor(TST_OUT).to(device = cuda)
        
        TRA_DSet = TensorDataset(TR_INP, TR_OUT)
        VAL_DSet = TensorDataset(TST_INP, TST_OUT)
        self.train_DL = DataLoader(TRA_DSet, batch_size=self.OTHERS['batchsize'])
        self.val_DL = DataLoader(VAL_DSet, batch_size=self.OTHERS['batchsize']*2)
        
        
        
print('PARAMETERS DEFINED !!!!')




class Model(nn.Module, PARAMETERS):
    def __init__(self, LIST, OTHERS, SCLR, TRAIN, VAL):
        super().__init__()

        self.scaler = StandardScaler()
        self.LIST = LIST
        self.OTHERS = OTHERS
        self.epoch = self.OTHERS['epoch']
        self.preprocess(split=216)
        self.layers = nn.ModuleList()
        self.Loss_FUNC = F.mse_loss

        for elem in self.LIST:
            key = elem[0]
            args = elem[1]
            self.layer_add(key,*args)
        
    def layer_add(self,key,*args):
        self.layers.append(self.layer_set(key,*args))

        
    def layer_set(self,key,*args):
        ## push args into key layer type, return it
        ## push args into key layer type, return it
        if key == 'CONV':
            return nn.Conv1d(*args)
        elif key == 'LSTM':
            return nn.LSTM(*args)
        elif key == 'DENSE':
            return nn.Linear(*args)
        elif key == 'dropout':
            return nn.Dropout(*args)
        elif key == 'batchnorm':
            return nn.BatchNorm1d(*args)
        elif key == 'flatten':
            return nn.Flatten()
        elif key == 'OTHERS':
            pass

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
    def loss_batch(self, TR_INP, TR_OUT, opt=None):
        loss = self.Loss_FUNC(self(TR_INP), TR_OUT)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(TR_INP)

    def fit(self):
        min_loss = 10
        for epoch in range(self.epoch):
            self.train()
            for TR_INP, TR_OUT in self.train_DL:
                self.loss_batch(TR_INP, TR_OUT, self.optimizer)

            self.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(TR_INP, TR_OUT) for TR_INP, TR_OUT in self.val_DL]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            if val_loss < min_loss:
                min_loss = val_loss
            print(epoch, val_loss)
        return min_loss
      



def fit(epochs, model, opt, train_dl, valid_dl):
    min_loss = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            model.loss_batch(xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[model.loss_batch(xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if val_loss < min_loss:
            min_loss = val_loss
        print(epoch, val_loss)
    return min_loss
        
def GET_Model(dictz):
    param_OBJ = PARAMETERS() 
    param_OBJ.GET_ALL()
    param_OBJ.OTHERS  =  {
                    'windowlength': 24,
                    'out_size': 3,
                    'period': 24,
                    'lrate': 0.0005,
                    'batchsize': 32,
                    'epoch': 100
                    }

    for TYPE in  list(DD.keys()):
      print(TYPE)
      for layernum in list(DD[TYPE].keys()):
        for PARAM in list(DD[TYPE][layernum].keys()):
          param_OBJ[TYPE][layernum][PARAMS] = dictz[TYPE][layernum][PARAMS]  
    
    model = Model(param_OBJ.LIST,param_OBJ.OTHERS, param_OBJ.scaler, param_OBJ.train_DL, param_OBJ.val_DL)
    
    
    
        
def GET_Model(DD):
    param_OBJ = PARAMETERS() 
    param_OBJ.GET_ALL()

    for PARAM in list(param_OBJ.keys()):
      if PARAM in list(OTHER_PARAM_TO_CHANGE)
      param_OBJ[]
    param_OBJ.OTHERS  =  {
                    'windowlength': 24,
                    'out_size': 3,
                    'period': 24,
                    'lrate': 0.0005,
                    'batchsize': 32,
                    'epoch': 100
                    }

    for TYPE in  list(DD.keys()):
      print(TYPE)
      for layernum in list(DD[TYPE].keys()):
        for PARAM in list(DD[TYPE][layernum].keys()):
          param_OBJ[TYPE][layernum][PARAMS] = DD[TYPE][layernum][PARAMS]  
    
    model = Model(param_OBJ.LIST,param_OBJ.OTHERS, param_OBJ.scaler, param_OBJ.train_DL, param_OBJ.val_DL)
    model.to(device = cuda)
    model.optimizer = optim.Adam(model.parameters(),lr=param_OBJ.OTHERS['lrate'])

    minloss = model.fit()
    torch.cuda.empty_cache()
    return {
        'loss': minloss,
        'status': STATUS_OK,
        'attachments':
            {'time_module': pickle.dumps(time.time)}
          }
  
  
  
    model.to(device = cuda)
    model.optimizer = optim.Adam(model.parameters(),lr=param_OBJ.OTHERS['lrate'])

    minloss = model.fit()
    torch.cuda.empty_cache()
    return {
        'loss': minloss,
        'status': STATUS_OK,
        'attachments':
            {'time_module': pickle.dumps(time.time)}
          }
  
  
  
PARAM_TO_CHANGE = {'CONV': {
        '1': {
              'KER': (2,14),
              'dropout': (0.2, 0.8),
              'batchnorm': True
             },

        '2': { 
              'KER': (2,11),
              'dropout': (0.2, 0.8),
              'batchnorm': True
             }
                   },


        'DENSE': {

        '1': {'FIL': (32,256),
              'dropout' : (0.2,0.3)
             },
        '2': {'FIL':self.OTHERS['out_size']}
        }
}






