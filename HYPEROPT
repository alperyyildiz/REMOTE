import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
import shutil
from torch import optim
from sklearn import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
#cuda = torch.device('cuda')

print('IMPORTS ARE DONE!!!')




class PARAMETERS():
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def DICT_TO_LIST(self):
        prev_out_ch = 0
        self.LIST = list()
        self.seq_len_left = self.DICT['OTHERS']['1']['windowlength']
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

                    if 'batchnorm' in list(self.DICT[key][layer].keys()):
                        if self.DICT[key][layer]['batchnorm'] is True:
                            self.LIST.append(['batchnorm',[prev_out_ch,True]])

                    print(self.DICT[key][layer]['dropout'])
                    if self.DICT[key][layer]['dropout'][0] is True:
                        self.LIST.append(['dropout',[self.DICT[key][layer]['dropout'][1],False]])
           
    def GET_OTHERS(self,OTHERS=None):
        if OTHERS is None:
            OTHERS  =  {
                            'windowlength': 24,
                            'out_size': 3,
                            'period': 24,
                            'lrate': 0.0005,
                            'batchsize': 32,
                            'epoch': 100
                            }
        return OTHERS
        
    def GET_DICT(self,DICT=None):
        #USES: GET_OTHERS


        #CREATES: self.DICT

        try:
            _ = list(DICT.keys())
        except:



            OTHERS = self.GET_OTHERS()
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
                                '2': {'FIL':OTHERS['out_size'],
                                      'dropout' : [False,0]}
                              },
                        
                      'OTHERS': {'1':OTHERS}
            }
            
          
    def CREATE_SEARCH_SPACE(self,TO_CHNG= None):
        #USES: GET_PARAMS_TO_CHANGE()

        #CREATES: self.space

        space = {}
        self.PARAMS_TO_CHANGE = self.GET_PARAMS_TO_CHANGE()
        HYP_DICT ={}
        d_count = 0
        for TYPE in  list(self.PARAMS_TO_CHANGE.keys()):
            HYP_DICT_LAYER = {}
            for layernum in list(self.PARAMS_TO_CHANGE[TYPE].keys()):
                HYP_DICT_PARAMS = {}
                for PARAM in list(self.PARAMS_TO_CHANGE[TYPE][layernum].keys()):
                    if PARAM == 'dropout':
                        d_count = d_count + 1
                        HYP_DICT_PARAMS[PARAM + str(d_count)] = hp.uniform(PARAM + str(d_count),self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    else:
                        HYP_DICT_PARAMS[PARAM + layernum] = hp.uniform(PARAM + layernum,self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    HYP_DICT_LAYER[layernum] =  HYP_DICT_PARAMS
                HYP_DICT[TYPE] =  HYP_DICT_LAYER

            self.space = hp.choice('paramz', [HYP_DICT])

    def GET_PARAMS_TO_CHANGE(self,PARAMS_TO_CHANGE=None):
        if PARAMS_TO_CHANGE is None:
              
            PARAMS_TO_CHANGE = {'CONV': {
                                          '1': {
                                                'KER': (2,14),
                                                'dropout': (0.2, 0.8),
                                              },

                                          '2': { 
                                                'KER': (2,11),
                                                'dropout': (0.2, 0.8),
                                              }
                                                    },
                              'DENSE': {

                                          '1': {'FIL': (32,256),
                                                'dropout' : (0.2,0.3)
                                                }
                                          }
                              }
        return PARAMS_TO_CHANGE

    #CREATE SUBDIR OF ABOVE NAMED WITH 
    #EXPERIMENT DATE AND START TIME
    def CREATE_DIR(self):
        #INPUT: self.DICT

        first_Con = True
        SAVE_DIR = ''
        for KEY in list(self.dict.keys()):
            if KEY != 'OTHERS':
                SAVE_DIR = SAVE_DIR + '_' +KEY + '-'
                for LAYER in list(self.dict[KEY].keys()):
                    SAVE_DIR = SAVE_DIR + LAYER + '-'
        SAVE_DIR = 'storage/' + SAVE_DIR[1:]
        try: 
            os.mkdir(SAVE_DIR)
            self.save_DIR = SAVE_DIR
        except:
            pass
        
        SAVE_DIR = SAVE_DIR + '/' +str(datetime.now())[:-10]
        self.save_DIR = SAVE_DIR

        try:
            os.mkdir(SAVE_DIR)
        except:
            pass

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self,DDD):
        save_DIR = ''
        plot_header = ''
        for KEY in list(DDD.keys()):
            save_DIR = save_DIR + '_' + KEY  
            plot_header = plot_header + '\n' + KEY + '--- ' 
            for LAYER in list(DDD[KEY].keys()):
                save_DIR = save_DIR + '-' + LAYER  
                plot_header = plot_header + LAYER + ': ' 
                for PARAM in list(DDD[KEY][LAYER].keys()):
                    save_DIR = save_DIR + PARAM + '-' + str(DDD[KEY][LAYER][PARAM])[:5] + '---'
                    plot_header = plot_header + PARAM + '=' + str(DDD[KEY][LAYER][PARAM])[:5] + '   '
            plot_header = plot_header + '\n'
        return save_DIR, plot_header


    
    #SAVE CONSTANT HYPERPARAMETERS OF EXPERIMENT AS TXT
    def WRITE_CONSTANTS(self):
        key_CONST = ''
        key_VAR = ''
        for KEY in list(self.DICT.keys()):
            exist = True
            if KEY is not 'flatten':
                if KEY not in list(self.PARAMS_TO_CHANGE.keys()):
                    exist = False
                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'
                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        for PARAM in list(self.DICT[KEY][LAYER].keys()):
                            key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])
                        key_CONST = key_CONST + '\n'
                else:
                    key_VAR = key_VAR + '\n\n TYPE:   {}\n\n'.format(KEY)
                    key_VAR = key_VAR + 'LAYER:'

                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'

                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        if LAYER in list(self.PARAMS_TO_CHANGE[KEY].keys()):
                            key_VAR =  key_VAR + '\n{} \t---\t'.format(LAYER)
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                if PARAM in list(self.PARAMS_TO_CHANGE[KEY][LAYER].keys()):
                                    key_VAR =  key_VAR + '{}: {} \t\t'.format(PARAM,self.PARAMS_TO_CHANGE[KEY][LAYER][PARAM])
                                else:
                                    key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                        else:
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                     
        with open('/CONSTANT_HYPERPARAMETERS.txt' , 'w') as file: 
            file.write(key_CONST)
        with open('/PARAMETERS_TO_TUNE.txt', 'w') as file: 
            file.write(key_VAR) 
 
            



    #SAVES HIST AND PRED PLOTS
    def SAVE_PLOTS(self):
        save_NAME, plot_header = self.CREATE_SAVE_NAME()
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(plot_header)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( self.save_DIR + '/' + save_NAME + '.png')
        
        fig2 = self.plotz()
        plt.savefig(self.save_DIR + '/PRD_' + save_NAME + '.png')
        plt.close('all')
        self.keyz.append(save_NAME)

    def CONV_DICT_TO_INT(self,DDD):
        for KEY in list(DDD.keys()):
            for LAYER in list(DDD[KEY].keys()):
                for PARAM in list(DDD[KEY][LAYER].keys()):    
                    if LAYER != 'flatten':
                        if PARAM not in ['dropout','batchnorm']:
                            DDD[KEY][LAYER][PARAM] = np.int(np.round(DDD[KEY][LAYER][PARAM]))
        return DDD


        
    def preprocess(self,split):
        data = pd.read_excel('clean.xlsx').dropna()
        windowlength = self.DICT['OTHERS']['1']['windowlength']
        outsize = self.DICT['OTHERS']['1']['out_size']
        arr = np.asarray(data['sales'])
        vv =pd.read_csv('vix.csv',sep=',')

        vix = np.array(vv['Şimdi'])
        for i in range(len(vix)):
            vix[i] = float(vix[i].replace(',','.'))

        dol =pd.read_csv('dollar.csv',sep=',')
        dollars = np.array(dol['Şimdi'])
        for i in range(len(dollars)):
            dollars[i] = float(dollars[i].replace(',','.'))
            
            
        res = STL(arr,period = self.DICT['OTHERS']['1']['period'] ,seasonal = 23 , trend = 25).fit()
        observed = res.observed
        a = np.concatenate([np.array(res.observed).reshape(res.observed.shape[0],1),np.array(res.seasonal).reshape(observed.shape[0],1),np.array(res.trend).reshape(observed.shape[0],1),np.array(res.resid).reshape(observed.shape[0],1).reshape(observed.shape[0],1),np.array(vix).reshape(observed.shape[0],1),np.array(dollars).reshape(observed.shape[0],1)],axis=1)
        dataz = np.swapaxes(np.array([res.observed,res.seasonal,res.trend,res.resid,vix,dollars]),0,1)
        train = dataz[:split]
        test = dataz[split:]
                
        MAX_window = self.DICT['OTHERS']['1']['windowlength']
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
        
        
        TR_INP = torch.Tensor(TR_INP)#.to(device = cuda)
        TST_INP = torch.Tensor(TST_INP)#.to(device = cuda)
        TR_OUT = torch.Tensor(TR_OUT)#.to(device = cuda)
        TST_OUT = torch.Tensor(TST_OUT)#.to(device = cuda)
        
        TRA_DSet = TensorDataset(TR_INP, TR_OUT)
        VAL_DSet = TensorDataset(TST_INP, TST_OUT)
        self.train_DL = DataLoader(TRA_DSet, batch_size=self.DICT['OTHERS']['1']['batchsize'])
        self.val_DL = DataLoader(VAL_DSet, batch_size=self.DICT['OTHERS']['1']['batchsize']*2)
        
        
    def GET_MODEL(self,DD):
        print(DD)
        print(DD)
        print(DD)
        print(self.DICT)

        call_data_again = False
        DD = self.CONV_DICT_TO_INT(DD)    
        for KEY in  list(DD.keys()):
            if KEY is 'OTHERS':
                key__ = list(DD['OTHERS']['1'].keys())
                for key_ in key__:
                    if key_ in ['windowlength','Period','batchsize','outsize']:
                        call_data_again = True
              
            for layernum in list(DD[KEY].keys()):
                for PARAM in list(DD[KEY][layernum].keys()):
                    PARAM_ = PARAM[:-1]
                    if PARAM_ == 'dropout' :
                        self.DICT[KEY][layernum][PARAM_] = [True,DD[KEY][layernum][PARAM]]
                    else:
                        self.DICT[KEY][layernum][PARAM_] = DD[KEY][layernum][PARAM]
        if call_data_again:
            self.Preprocess(split=220)

        print(self.DICT)
        self.DICT_TO_LIST()
        model = Model(self.DICT,self.LIST, self.DICT['OTHERS']['1'], self.scaler, self.train_DL, self.val_DL)
        #model#.to(device = cuda)
        model.optimizer = optim.Adam(model.parameters(),lr=self.DICT['OTHERS']['1']['lrate'])
        minloss = model.fit()
        #torch.cuda.empty_cache()
        return {
            'loss': minloss,
            'status': STATUS_OK,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
              }
  
print('PARAMETERS DEFINED !!!!')




class Model(nn.Module, PARAMETERS):
    def __init__(self, DICT,LIST, OTHERS, SCLR, TRAIN, VAL):
        super().__init__()

        self.scaler = StandardScaler()
        self.LIST = LIST
        self.DICT = DICT
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
        BEST_LOSS = 5
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

            is_best = val_loss < BEST_LOSS
            BEST_LOSS = min(val_loss,BEST_LOSS)
            self.save_checkpoint({
                              'epoch': epoch + 1,
                              'state_dict': self.state_dict(),
                              'BEST_LOSS': BEST_LOSS,
                              'optimizer' : self.optimizer.state_dict(),
                            }, is_best)

        return BEST_LOSS
      
    def save_checkpoint(self,state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')




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


def SET_EXPERIMENT(PARAMS_TO_CHANGE=None):
    P_OBJ = PARAMETERS()
    P_OBJ.EXPERIMENT_NUMBER = 1
    P_OBJ.GET_DICT()
    P_OBJ.GET_PARAMS_TO_CHANGE()
    P_OBJ.CREATE_SEARCH_SPACE()
    P_OBJ.WRITE_CONSTANTS()
    P_OBJ.preprocess(split=220)
    print(P_OBJ.DICT)
    best = fmin(fn=P_OBJ.GET_MODEL,
                space=P_OBJ.space,
                algo=tpe.suggest,
                max_evals=3)
