import numpy as np
import torch
from torchvision import datasets, transforms, models
import json
from torch import nn
from torch import optim
from collections import OrderedDict
from PIL import Image

def preprocess_data(data_dir, preprocessor):
    train = data_dir + '/train'
    valid = data_dir + '/valid'
    test = data_dir + '/test'
    preprocessors = {}
    preprocessors['train'] = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.Resize(256),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                          ])
    preprocessors['valid'] = preprocessor
    preprocessors['test'] = preprocessor
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(train, transform=preprocessors['train'])
    image_datasets['valid'] = datasets.ImageFolder(valid, transform=preprocessors['valid'])
    image_datasets['test'] = datasets.ImageFolder(test, transform=preprocessors['test'])
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    return dataloaders, image_datasets

def get_classifier(hidden_units, output_size, input_size):
    return nn.Sequential(
        OrderedDict([
            ('fc1',     nn.Linear(input_size, hidden_units)),
            ('relu',    nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2, inplace=False)),
            ('fc2',  nn.Linear(hidden_units, output_size)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

def validate_model(model, criterion, dataloader, device):
    model.eval()
    model.to(device=device)

    accuracy, test_loss = 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss/len(dataloader), accuracy/len(dataloader)

def test_network(model, validloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()

            print("Accuracy is", correct/total*100)

def train_network(epochs, dataloaders, model, criterion, optimizer, device):
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if ii % print_every == 0:
                loss, accuracy = validate_model(model=model, criterion=criterion, dataloader=dataloaders['valid'], device=device)
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                    "Training Loss: {:.3f} ".format(running_loss/print_every),
                    "Validation Loss: {:.3f} ".format(loss),
                    "Validation Accuracy: {:.3f}".format(accuracy))
                running_loss = 0
                model.train()

def get_weights(arch):
    weights = models.get_model_weights(arch)
    weight = weights._member_map_['DEFAULT']
    return weight

def get_cat_to_name(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def trainNetwork(save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    weight = get_weights(arch)
    dataloaders, image_datasets = preprocess_data('flowers', weight.transforms())
    cat_to_name = get_cat_to_name('cat_to_name.json')
    outputs = len(cat_to_name)

    model = getattr(models, arch)(weights=weight)

    if not (arch.startswith("vgg") or arch.startswith("densenet")):
        print("Only vgg and densenet models are supported")
        return

    if arch.startswith("vgg"):
        input_size = model.classifier[0].in_features

    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }
    if arch.startswith("densenet"):
        input_size = densenet_input[arch]
    for param in model.parameters():
        param.requires_grad = False

    classifier = get_classifier(hidden_units, outputs, input_size)
    model.classifier = classifier

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = get_device(gpu)
    model.to(device)


    train_network(epochs, dataloaders, model, criterion, optimizer, device)
    test_network(model, dataloaders['test'], device)
    save_model(save_dir, arch, hidden_units, epochs, model, optimizer, image_datasets, input_size, learning_rate)

def get_device(gpu):
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

def save_model(save_dir, arch, hidden_units, epochs, model, optimizer, image_datasets, input_size, learning_rate):
    model.class_to_idx = image_datasets['train'].class_to_idx
    chkpt = save_dir + '/checkpoint.pth'
    torch.save({'structure': arch,
            'hidden_layer1': hidden_units,
            'input_size': input_size,
            'droupout': 0.2,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'model_state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer_dict': optimizer.state_dict()},
            chkpt)

def load_model(filepath, gpu):
    checkpoint = torch.load(filepath)
    weights = get_weights(checkpoint['structure'])
    model = getattr(models, checkpoint['structure'])(weights=weights)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = get_classifier(checkpoint['hidden_layer1'], len(model.class_to_idx), checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])

    model.eval()
    device = get_device(gpu)
    model.to(device)
    return model, optimizer

def process_image(image):
    width, height = image.size

    if  width > height:
        image.thumbnail((width, 256))
    else:
        image.thumbnail((256, height))

    width, height = image.size
    left = (width - 224)/2
    bottom = (height - 224)/2
    right = left + 224
    top = bottom + 224

    crop_image=image.crop((left, bottom, right, top))

    np_image=np.array(crop_image)
    np_image=np_image.astype(np.float32)
    np_image=np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_image=(np_image-mean)/std


    procesed_image=new_image.transpose((2,0,1))

    return procesed_image

def predict_image(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    image_as_np = process_image(img)

    image = torch.from_numpy(image_as_np).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model.forward(image)
    exp = torch.exp(output)
    top_p, top_class = exp.topk(topk, dim=1)
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    top_p = top_p.detach().numpy()
    top_class = top_class.detach().numpy()
    return top_p, top_class

def view_classify(img_path, model, cat_to_name, gpu, topk=5):
    device = get_device(gpu)
    top_p, top_class = predict_image(img_path, model, device, topk)
    top_p = top_p[0]
    top_class = top_class[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[lab] for lab in top_class]
    top_flowers = [cat_to_name[lab] for lab in top_class]
    for item in zip(top_flowers, (top_p*100)):
        print(item)


def predict(image_path, model_path, topk, category_names, gpu):
    model, _ = load_model(model_path, gpu)
    cat_to_name = get_cat_to_name(category_names)
    view_classify(image_path, model, cat_to_name, gpu, topk)
