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
