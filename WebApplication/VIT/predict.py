import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from .vit_model import vit_base_patch16_224 as create_model
from .vit_model import vit_base_patch16_224_in21k as create_model2

def recognize():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img = Image.open('static/segment/cropped.jpg')
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'VIT/class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model
    model = create_model(num_classes=1000).to(device)
    # load model weights
    model_weight_path = "weight/vit_base_patch16_224.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
    print(print_res)
    res = []
    for i in range(len(predict)):
        # res[class_indict[str(i)]] = predict[i].numpy()
        # print(predict[i].numpy())
        res.append({'class':class_indict[str(i)], 'prob':round(predict[i].numpy().item(),4)})

    res = sorted(res, key=lambda x: x['prob'], reverse=True)
    # print(res[0:10])  
    return res[0:10]
    

def recognize2():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img = Image.open('static/segment/cropped.jpg')
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'VIT/class_indices2.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # create model
    model = create_model2().to(device)
    # load model weights
    model_weight_path = "weight/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
    print(print_res)
    res = []
    for i in range(len(predict)):
        # res[class_indict[str(i)]] = predict[i].numpy()
        # print(predict[i].numpy())
        res.append({'class':class_indict[str(i)], 'prob':round(predict[i].numpy().item(),4)})

    res = sorted(res, key=lambda x: x['prob'], reverse=True)
    # print(res[0:10])  
    return res[0:10]
    

