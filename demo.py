import torch as t
import torchvision as tv
import os
import time
import numpy as np
from tqdm import tqdm


class DefaultConfigs(object):

    data_dir = "./traffic-sign/"
    data_list = ["train","test"]

    lr = 0.001
    epochs = 10
    num_classes = 62
    image_size = 224
    batch_size = 40
    channels = 3
    gpu = "0"
    train_len = 4572
    test_len = 2520
    use_gpu = t.cuda.is_available()

config = DefaultConfigs()

normalize = tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225]
                                    )

transform = {
    config.data_list[0]:tv.transforms.Compose(
        [tv.transforms.Resize([224,224]),tv.transforms.CenterCrop([224,224]),
        tv.transforms.ToTensor(),normalize]#tv.transforms.Resize 用于重设图片大小
    ) ,
    config.data_list[1]:tv.transforms.Compose(
        [tv.transforms.Resize([224,224]),tv.transforms.ToTensor(),normalize]
    ) 
}

datasets = {
    x:tv.datasets.ImageFolder(root = os.path.join(config.data_dir,x),transform=transform[x])
    for x in config.data_list
}

dataloader = {
    x:t.utils.data.DataLoader(dataset= datasets[x],
        batch_size=config.batch_size,
        shuffle=True
    ) 
    for x in config.data_list
}


def get_model(num_classes):
    
    model = tv.models.resnet18(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False
    model.fc = t.nn.Sequential(
        t.nn.Dropout(p=0.3),
        t.nn.Linear(512,num_classes)
    )
    return(model)


def train(epochs):
 
    model = get_model(config.num_classes)
    print(model)
    loss_f = t.nn.CrossEntropyLoss()
    if(config.use_gpu):
        model = model.cuda()
        loss_f = loss_f.cuda()
    
    opt = t.optim.Adam(model.fc.parameters(),lr = config.lr)
    time_start = time.time()
    
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []
        model.train(True)
        print("Epoch {}/{}".format(epoch+1,epochs))
        for batch, datas in tqdm(enumerate(iter(dataloader["train"]))):
            x,y = datas
            if (config.use_gpu):
                x,y = x.cuda(),y.cuda()
            y_ = model(x)
            #print(x.shape,y.shape,y_.shape)
            _, pre_y_ = t.max(y_,1)
            pre_y = y
            #print(y_.shape)
            loss = loss_f(y_,pre_y)
            #print(y_.shape)
            acc = t.sum(pre_y_ == pre_y)
 
            loss.backward()
            opt.step()
            opt.zero_grad()
            if(config.use_gpu):
                loss = loss.cpu()
                acc = acc.cpu()
            train_loss.append(loss.data)
            train_acc.append(acc)
            #if((batch+1)%5 ==0):
        time_end = time.time()
        print("Batch {}, Train loss:{:.4f}, Train acc:{:.4f}, Time: {}"\
            .format(batch+1,np.mean(train_loss)/config.batch_size,np.mean(train_acc)/config.batch_size,(time_end-time_start)))
        time_start = time.time()
        
        model.train(False)
        
        for batch, datas in tqdm(enumerate(iter(dataloader["test"]))):
            x,y = datas
            if (config.use_gpu):
                x,y = x.cuda(),y.cuda()
            y_ = model(x)
            #print(x.shape,y.shape,y_.shape)
            _, pre_y_ = t.max(y_,1)
            pre_y = y
            #print(y_.shape)
            loss = loss_f(y_,pre_y)
            acc = t.sum(pre_y_ == pre_y)
 
            if(config.use_gpu):
                loss = loss.cpu()
                acc = acc.cpu()
 
            test_loss.append(loss.data)
            test_acc.append(acc)
        time_end = time.time()
        print("Batch {}, Test loss:{:.4f}, Test acc:{:.4f}, Time :{}".format(batch+1,np.mean(test_loss)/config.batch_size,np.mean(test_acc)/config.batch_size,(time_end-time_start)))
        time_start = time.time()
        t.save(model,str(epoch+1)+"ttmodel.pkl")
 
 
 
if __name__ == "__main__":
    train(config.epochs)
