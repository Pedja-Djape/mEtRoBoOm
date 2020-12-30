import numpy as np 
import torch 
from torch.autograd import Variable 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image 
import torchvision
import matplotlib.pyplot as plt 
from torch.optim import SGD,Adam 
from model import CNN
import time 
from sklearn.metrics import confusion_matrix

def getStats(path):
    """
        Input:
        path --> Path to Image Folder

        Returns:
        avg --> mean of channels
        std --> stdDev of channels
        transform --> transform of images/tensors
    """
    # temporary ds to find mean and std
    tempDS = ImageFolder(path)

    # find avg and std in ds
    avg = [0,0,0]; std = [0,0,0]
    for pic in tempDS:
        # PIL needs pixel value b/w 0 and 1.
        img = np.asarray(pic[0])/255
        for i in range(3):
            avg[i] += (img[:,:,i].mean())/len(tempDS)
            std[i] += ((img[:,:,i]).std())/(len(tempDS))
    print(f'AVGS: {avg} \nSTDS: {std}')
    # transform each image to a tensor and normalize
    transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize(avg,std)]
    )
    return avg,std,transform

def unNormalizeShow(imgArr,mean, stdDev):
    # don't wanna modify the original image
    tmp = imgArr
    for i in range(3):
        pix = tmp[:,:i] * stdDev[i] + mean[i] 
    
    npimg = tmp.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def loadData(batchSize,transform,shuffle=True,seed=0,split = {'train': 0.7, 'valid': 0.2, 'test': 0.1}):
    train = ImageFolder("./Dataset",transform=transform)
    size = len(train)
    idxs = list(range(size))

    # print(f"train.classes: \n{train.classes}")

    country = idxs[:499]
    edm = idxs[499:499*2]
    hiphop = idxs[499*2:499*3]
    latin = idxs[499*3:499*4]
    pop = idxs[499*4:499*5] 
    rb = idxs[499*5:499*6] 
    rock = idxs[499*6:]

    # print(len(country),len(edm),len(hiphop),len(latin),len(pop),len(rb),len(rock))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(country)
        np.random.shuffle(edm)
        np.random.shuffle(hiphop)
        np.random.shuffle(latin)
        np.random.shuffle(pop)
        np.random.shuffle(rb)
        np.random.shuffle(rock)

    trainStop = int(np.floor(
        499 * (split['train'])
    ))

    validStop = int(np.floor(
        trainStop + (499 * (split['valid']))
    ))

    idxs = [country, edm, hiphop, latin, pop, rb, rock]

    train_idxs = []
    valid_idxs = []
    test_idxs = []
    for i in range(7):
        for j in range(trainStop):
            train_idxs.append(idxs[i][j])
        for j in range(trainStop, validStop):
            valid_idxs.append(idxs[i][j])
        for j in range(validStop,499):
            test_idxs.append(idxs[i][j])

    splitIdxs = {'train': train_idxs, 'valid': valid_idxs, 'test': test_idxs}
    loaders = {"trainLoader": None,"validLoader": None,"testLoader": None}

    for split in splitIdxs:
        for loader in loaders:
            if (split + "Loader" == loader) and loaders[loader] == None:
                sampler = SubsetRandomSampler(splitIdxs[split])
                loaders[loader] = DataLoader(train,batch_size=batchSize,sampler=sampler)

    return loaders


def loadModel(lr):
    torch.manual_seed(seed=0)
    model = CNN()
    optim = Adam(model.parameters(),lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model,optim,criterion



def evaluate(outputs, labels, loss_fnc):
  output = outputs.detach().numpy()
  label = labels.detach().numpy()
  count = 0
  for i in range(output.shape[0]):
    if np.argmax(output[i]) == label[i]:
      count += 1
  return count / output.shape[0]

def evaluate1(model, val_loader, batch_size, loss, val_loss, loss_fnc):
  eval, count, total_loss = 0, 0, 0
  for i, data in enumerate(val_loader):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()
    outputs = model(inputs)
    outputs = outputs.type(torch.float)
    labels = labels.type(torch.LongTensor)
    eval += evaluate(outputs, labels, loss_fnc)
    count += 1

    loss_in = loss(input=outputs, target=labels)
    total_loss += float(loss_in.item())
  val_loss.append(total_loss/count)
  return eval/count

def train(batchSize,transform,lr,epochs=10,evalEvery=77):
    sets = loadData(batchSize,transform)
    train,valid,test = sets['trainLoader'], sets['validLoader'], sets['testLoader']
    valid_acc, train_acc, valid_loss, train_loss, time_total = [], [], [], [], []
    first_time = time.time()

    classes = [0,0,0,0,0,0,0]
    for j, data in enumerate(test):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    # print(classes)

    classes = [0,0,0,0,0,0,0]
    for j, data in enumerate(train):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    # print(classes)

    classes = [0,0,0,0,0,0,0]
    for j, data in enumerate(valid):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    model,optim, criterion = loadModel(lr=lr)

    for epoch in range(epochs):
        total_loss, total_corr, train_eval = 0, 0, 0
        for i, data in enumerate(train, 0):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            optim.zero_grad()

            outputs = model(inputs)

            outputs = outputs.type(torch.float)
            labels = labels.type(torch.LongTensor)

            loss_in = criterion(input=outputs, target=labels)
            loss_in.backward()
            optim.step()

            total_loss = float(total_loss + loss_in.item())
            train_eval = train_eval + evaluate(outputs, labels, criterion)

            if i % evalEvery == evalEvery-1:

                acc = evaluate1(model, valid, batchSize, criterion, valid_loss, criterion)
                print('Validation Accuracy: ' + str(acc))
                valid_acc.append(acc)
                train_acc.append(train_eval / ((i%evalEvery)+1))
                print('Training Accuracy: ' + str(train_eval / ((i%evalEvery)+1)))
                time_diff = time.time() - first_time
                print('  Elapsed Time: ' + str(time_diff) + ' seconds')
                train_loss.append(total_loss / ((i%evalEvery)+1))
                train_eval = 0
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, total_loss / ((i%evalEvery)+1)))
                total_loss = 0

    
    test_loss = 0
    test_acc = 0
    true = []
    pred = []
    for j, data in enumerate(test):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          true.append(i)
      outputs = model(inputs)
      for i in outputs.detach().numpy():
          pred.append(np.argmax(i))
      outputs = outputs.type(torch.float)
      labels = labels.type(torch.LongTensor)
      loss_in = criterion(input=outputs, target=labels)
      loss_in.backward()
      test_loss += float(test_loss + loss_in.item())
      test_acc += evaluate(outputs, labels, criterion)
    print("Test Loss: " + str(test_loss/j))
    print("Test Accuracy: " + str(test_acc/j))

    print(confusion_matrix(true,pred))

    plt.plot(valid_acc, label = "Validation")
    plt.plot(train_acc, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Steps")
    plt.show()

    plt.plot(valid_loss, label = "Validation")
    plt.plot(train_loss, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Steps")
    plt.show()


if __name__ == "__main__":
    print("Obtaining Stats and transform")
    _, _, transform = getStats(path="./Dataset")
    print("loading model and training...")
    train(batchSize=32,transform=transform,lr=0.001)