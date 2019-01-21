import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets

def main():
    args = get_arguments()
    model = load_checkpoint(args.checkpoint)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
    results = check_Model(args.input,model)
    print(results)
    
    
def Check_Image(img,m):
    filename = img.split('/')[-2]
    img = Image.open(img)
    flower = m[filename]
    return flower
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store",dest="input",default='flowers/test/1/image_06743.jpg')
    parser.add_argument("--checkpoint", action="store",dest="checkpoint",default='checkpoint.pth')
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--cuda", action="store_true", dest="cuda", default=False, help="Set Cuda True for using the GPU")
    return parser.parse_args()

def process_image(image):
    opened_image = images = Image.open(image)
    image_processor = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    final = image_processor(opened_image)
    return final
    
def predict(image_path, model, topk=5):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder("flowers/train", transform=train_transforms)
    model.class_to_idx = train_data.class_to_idx
    md = model.class_to_idx
    torchs = process_image(image_path)
    torchs = torchs.unsqueeze(0)
    torchs = torchs.float()
    with torch.no_grad():
        output = model.forward(torchs.cuda())
    plob = F.softmax(output.data,dim=1)
    return plob.topk(topk)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    return model

def check_Model(path,model):
    args = get_arguments()
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
    prob = predict(path, model)
    names = [cat_to_name[str(index + 1)] for index in np.array(prob[1][0])]
    count = 0
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
    name = Check_Image(args.input,cat_to_name)
    print("                                  {}\n\n".format(name))
    p = np.array(prob[0][0])
    for i in names:
        z = p[count]*100
        x = round(z,2)
        print("                             {}: {}%\n".format(i,x))
        count +=1

main()