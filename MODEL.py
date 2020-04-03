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

