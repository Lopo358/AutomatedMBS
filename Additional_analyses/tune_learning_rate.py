import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os, os.path
import copy
from torch.utils.data.sampler import Sampler
#from torchsampler import ImbalancedDatasetSampler #how to install this??

# Hyperparameters
num_epochs = 20
num_classes = 4
batch_size = 32
#learning_rate = 0.001
nc = 3 #number of channels
ngpu = 1 #number of GPUs to use
optimizer = "SGD" #or Adam or RMSprop
init = "normal" #initialization method (normal, xavier, kaiming)
imgSize = 224
nonlinearity = "relu" #selu, prelu, leaky, relu
dropout = 0.5 #probability of a dropout

data_dir = './MBS/pictures'


#Number of images per class
class_sample_count = np.array(
    [(len([name for name in os.listdir('./MBS/pictures/train/MBS_0')])), (len([name for name in os.listdir('./MBS/pictures/train/MBS_1')])), (len([name for name in os.listdir('./MBS/pictures/train/MBS_2')])), (len([name for name in os.listdir('./MBS/pictures/train/MBS_3')]))])
print(class_sample_count)

val_class_sample_count = np.array(
    [(len([name for name in os.listdir('./MBS/pictures/val/MBS_0')])), (len([name for name in os.listdir('./MBS/pictures/val/MBS_1')])), (len([name for name in os.listdir('./MBS/pictures/val/MBS_2')])), (len([name for name in os.listdir('./MBS/pictures/val/MBS_3')]))])
print(val_class_sample_count)
weight = 1. / class_sample_count
#weight_val = 1. / val_class_sample_count
print(weight)

sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, batch_size)
#sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weight_val, batch_size)
#sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(imgSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.6063, 0.5952, 0.6372], [0.0909, 0.0695, 0.0525]) #mean and sd of training images 
    ]),
    'val': transforms.Compose([
        transforms.Resize(imgSize+30),
        transforms.CenterCrop(imgSize),
        transforms.ToTensor(),
        transforms.Normalize([0.6063, 0.5952, 0.6372], [0.0909, 0.0695, 0.0525]) #mean and sd of training images
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
samplers = {
    'train':sampler,
    'val':None
}
shufflers = {
    'train':False,
    'val':True
}
print(samplers)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=shufflers[x], sampler = samplers[x], num_workers=4)
              for x in ['train', 'val']}
print(dataloaders)

#dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=False, sampler = sampler, num_workers=4)
#dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True, num_workers=4)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#for i, (inp, tar) in enumerate(dataloaders['train']):
#    print(inp.size())

#Schedule the learning rate, saving the best model
#parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler.

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#Load a pretrained model and reset final fully connected layer.

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2. nn.Linear(num_ftrs, 2)
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss() #cross entropy is a good choice for unbalanced classes

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Tune the learning rate
# Thanks to Maximillian Pfau
#Get optimal learning rate
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-8)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=2)

#Train model with exp_lr_scheduler
model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

#Get data and plot
train_history = model['train_history']
model = model['model']

train_history = pd.DataFrame.from_dict(train_history)
#print(train_history[['lr','train_loss','val_loss']])

plt.savefig('path/to/figure.pdf') 

torch.save(model_ft, './MBS/PyTorch/transfer_tune_LR_mod.pt')
torch.save(model_ft.state_dict(), './MBS/PyTorch/ransfer_tune_LR_dict.ckpt')
