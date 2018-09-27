from __future__ import print_function, division

#pytorch imports
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils

#image / graphics imports 
from skimage import io, transform
from PIL import Image
from pylab import *

#data
from copy import deepcopy

#data science
import numpy as np
import scipy as sp
import math

#import other modules
import CAM_CXRDataset as CXR
import Eval as E

from importlib import reload
reload(CXR)
reload(E)

def calc_cam(x,label,model):
    """
    function to generate a class activation map corresponding to a torch image tensor

    Args:
        x: the 1x3x224x224 pytorch tensor file that represents the NIH CXR 
        label:user-supplied label you wish to get class activation map for; must be in FINDINGS list
        model: densenet121 trained on NIH CXR data

    Returns:
        cam_torch: 224x224 torch tensor containing activation map
    """
    FINDINGS = [ 'iu','msh','nih']
    
    if not label in FINDINGS:
        raise ValueError(str(label)+"is an invalid finding - please use one of "+str(FINDINGS))

    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #print("target:"+label)
    #find index for label; this corresponds to index from output of net
    label_index = next((x for x in range(len(FINDINGS)) if FINDINGS[x]==label)) 
    #print("label_index="+str(label_index))
        
        
    #define densenet_last_layer class so we can get last 1024 x 7 x 7 output of densenet for class activation map
    class densenet_last_layer(torch.nn.Module):
        def __init__(self,model):
            super(densenet_last_layer, self).__init__()
            self.features = torch.nn.Sequential(
                *list(model.children())[:-1]
            )
        def forward(self, x):
            x = self.features(x)
            x = torch.nn.functional.relu(x, inplace=True)
            
            return x

    #print(densenet_last_layer)
    #instantiate cam model and get output
    model_cam = densenet_last_layer(model)
    #print(model_cam)
    x = torch.autograd.Variable(x)
    y=model_cam(x)
    y=y.cpu().data.numpy()
    y=np.squeeze(y)
    #print("y")
    #print(y)
    
    #pull weights corresponding to the 1024 layers from model
    #print(model.state_dict().keys())
    weights = model.state_dict()['classifier.0.weight'] 
    weights = weights.cpu().numpy()
    #print(weights)
    
    #could replicate bottleneck and probability calculation here from last_layer network and params from 
    #original network to ensure that your reconstruction is accurate; have previously confirmed this
    
    bias = model.state_dict()['classifier.0.bias']
    #print(bias)
    bias = bias.cpu().numpy()
    model_bn = deepcopy(model)
    new_classifier = torch.nn.Sequential(*list(model_bn.classifier.children())[:-2]) #-2
    model_bn.classifier = new_classifier
    #print(model_bn)
    bn=model_bn(x)
    #print(bn)
    recreate=0
    bottleneck = []
    for k in range(0,1024):
        avg_value = np.mean(y[k,:,:])# over the 7x7 grid        
        bottleneck.append(avg_value)
        recreate += avg_value*weights[label_index,k]
    recreate = recreate + bias[label_index]
    #recreate = 1/(1+math.exp(-recreate))
    #print("recalc:")
    #print(recreate)
    #print("original:")
    #print(model(x).data.numpy()[0][label_index])  
    if np.abs(recreate - model(x).data.numpy()[0][label_index])>0.01:
        raise ValueError ("recreate != original - investigate")

    #create 7x7 cam
    cam = np.zeros((7,7)) #np.zeros((7,7,1))
    for i in range(0,7):
        for j in range(0,7):
            for k in range(0,1024):
                cam[i,j] += y[k,i,j]*weights[label_index,k]
    cam[:,:]+=bias[label_index]

    #make sure it checks out
    if np.abs(np.mean(cam)-model(x).data.numpy()[0][label_index])>0.01:
        raise ValueError ("np.mean(cam) != original - investigate")
    #else:
    #    print("np.mean(cam)")
    #    print(str(np.mean(cam)))
    #    print("orig")
    #    print(str(model(x).data.numpy()[0][label_index]))
    #print("cam leaving calc_cam")    
    #print(cam)
    #resize and smooth it                
    #cam_net_resize=np.zeros((224,224))
    #for i in range(0,7):
    #    for j in range(0,7):
    #        cam_net_resize[(32*i):(32*(i+1)),(32*j):(32*(j+1))]=cam[i,j]
    #sigma=[7,7]
    
    #cam_net_resize_smooth = cam_net_resize#sp.ndimage.filters.gaussian_filter(cam_net_resize, sigma, mode='constant')
    #cam_net_resize_smooth=torch.from_numpy(cam_net_resize_smooth).float()
    
    return cam


def show_cxr(inp, title=None):
    """
    displays grid of images
    
    Args:
        inp: output from torchvision.output.makegrid; images to be displayed in grid format
    Returns:
        None (plots)
        
        
    """
    matplotlib.pyplot.close("all")
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(9,9))
    #print(inp)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def load_data(PATH_TO_IMAGES,LABEL,PATH_TO_MODEL,POSITIVE_FINDINGS_ONLY):
    """
    Loads dataloader and torchvision model
    
    Args:
        PATH_TO_IMAGES: path to NIH CXR images
        LABEL: finding of interest (must exactly match one of FINDINGS defined below or will get error)
        PATH_TO_MODEL: path to downloaded pretrained model or your own retrained model
        POSITIVE_FINDINGS_ONLY: dataloader will show only examples + for LABEL pathology if True, otherwise shows positive
                                and negative examples if false
    
    Returns:
        dataloader: dataloader with test examples to show
        model: fine tuned torchvision densenet-121
    """
    
    checkpoint = torch.load(PATH_TO_MODEL)
    model = checkpoint['model_ft']
    del checkpoint
    model.cpu()

    #build dataloader on test
    mean= [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    

    FINDINGS = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    data_transform = transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    #show positive findings only; 
    if not POSITIVE_FINDINGS_ONLY:
        finding="any"
    else:
        finding=LABEL
        
    dataset = CXR.CXRDataset(path_to_images=PATH_TO_IMAGES, fold='test', transform=data_transform,finding=finding)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    #find index for label; this corresponds to index from output of net
    print("length of dataset:")
    print(len(dataset))
    return dataloader, model

def get_next(dataloader,model, LABEL):
    """
    Plots CXR, class activation map of CXR, and shows model probabilities of findings
    
    Args:
        dataloader: dataloader of test CXRs
        model: fine-tuned torchvision densenet-121 
        LABEL: finding we're interested in seeing heatmap for
    Returns:
        None (plots output)
    """
    FINDINGS = ['iu','msh','nih']
    SHOW=False
    label_index = next((x for x in range(len(FINDINGS)) if FINDINGS[x]==LABEL))  
    
    
    #get next iter from dataloader
    inputs,labels = next(iter(dataloader))
    
    #get normalized cam map
    original = inputs.clone()
    cam_iu = calc_cam(inputs,"iu",model)
    cam_msh = calc_cam(inputs,"msh",model)
    cam_nih = calc_cam(inputs,"nih",model)
    raw_net_cam_net=(cam_nih-cam_msh-cam_iu)
    if(SHOW):
        print("RAW NET CAM NET")
        print(raw_net_cam_net)
    
    
    #normalize according to the one we want
    cam_prob = np.zeros((7,7))
    for i in range(0,7):
        for j in range(0,7):
            #print("ij"+str(i)+str("j"))
            #print(np.exp(cam_iu[i,j]))
            denom=(np.exp(cam_iu[i,j])+np.exp(cam_msh[i,j])+np.exp(cam_nih[i,j]))
            proba=np.nan
            if LABEL=="iu": 
                  cam_prob[i,j]= np.exp(cam_iu[i,j])/denom
            if LABEL=="msh": 
                  cam_prob[i,j]= np.exp(cam_msh[i,j])/denom
            if LABEL=="nih": 
                  cam_prob[i,j]= np.exp(cam_nih[i,j])/denom
                                                           
    if LABEL=="iu": 
          proba = np.exp(np.mean(cam_iu))/(np.exp(np.mean(cam_iu))+np.exp(np.mean(cam_msh))+np.exp(np.mean(cam_nih)))
    if LABEL=="msh": 
          proba = np.exp(np.mean(cam_msh))/(np.exp(np.mean(cam_iu))+np.exp(np.mean(cam_msh))+np.exp(np.mean(cam_nih)))    
    if LABEL=="nih": 
          proba = np.exp(np.mean(cam_nih))/(np.exp(np.mean(cam_iu))+np.exp(np.mean(cam_msh))+np.exp(np.mean(cam_nih)))         
    if(SHOW):
        print("cam prob")
        print(cam_prob)
        
    #normalize raw_net_cam_net
    raw_net_cam_net=raw_net_cam_net-np.mean(raw_net_cam_net)
    raw_net_cam_net[raw_net_cam_net<0]=0
    raw_net_cam_net=raw_net_cam_net/np.max(raw_net_cam_net)
    
    if(SHOW):
        print("raw_net_cam_net after proc")
        print(raw_net_cam_net)
        
    #print("clip cam prob")
    #cam_prob = np.minimum(cam_prob,(1-1e-10))

    #cam_odds = cam_prob / (1-cam_prob)
    #cam_odds = cam_odds / np.max(cam_odds)
    #if(SHOW):
    #    print("cam_odds")
    #    print(cam_odds)
    #clip cam_prob if <0.90
    #cam_prob[cam_prob<0.95]=-1
    #cam_prob=cam_prob*3
    #make displayable version of cam_odds
    cam_net_resize=np.zeros((224,224))
    for i in range(0,7):
        for j in range(0,7):
            cam_net_resize[(32*i):(32*(i+1)),(32*j):(32*(j+1))]=raw_net_cam_net[i,j]
    sigma=[7,7]
    cam_net_resize_smooth = sp.ndimage.filters.gaussian_filter(cam_net_resize, sigma, mode='constant')
    cam_net_resize_smooth=torch.from_numpy(cam_net_resize_smooth).float()

    
    #print("averages from each cam map")
    #print("iu")
    #print(np.mean(cam_iu))
    #print("msh")
    #print(np.mean(cam_msh))
    #print("nih")
    #print(np.mean(cam_nih))
    #print("cam_prob")
    #print("~~~")
    #print(cam_prob)
    #print(np.mean(cam_prob))
    
    active = cam_net_resize_smooth
    #active_mean = active.mean()
    #active = active-active_mean
    #active[active<0]=0
    #active_max = np.max(active.cpu().numpy())    
    #if active_max > 3: 
    #    active = active * (3/active_max)
    
    #place cam map into the red channel of the original image
    inputs[0,0,:,:]+=3*active
    #inputs[0,1,:,:]=-3#active
    #inputs[0,2,:,:]=-3#active

    #display original and original with overlaid cam map
    if(SHOW):
        #out = torchvision.utils.make_grid(torch.cat((original,inputs)))
        out = torchvision.utils.make_grid(torch.cat((inputs)))
        show_cxr(out)
        plt.axis('off')
        savefig('heatmap.png')
        plt.show()        
        #plt.imshow(cam_prob, cmap='hot', interpolation='nearest')
        #plt.show()
        #print(cam_prob)
    
    #print predictions for label of interest and all labels
    pred=model(torch.autograd.Variable(original.cpu())).data.numpy()[0]
    predx=['%.3f' % elem for elem in list(pred)]
    if(SHOW):

        #print("True labels:")
        #print(np.array(FINDINGS)[labels.numpy().astype(int)[0]==1])
        #print("\n")

        print("Print proba each class:")
        denom = math.exp(pred[0])+math.exp(pred[1])+math.exp(pred[2])
        for item in zip(FINDINGS,predx):
            print(item[0]+" "+str(math.exp(double(item[1]))/(denom)))
        print("\n")
        
        print("Raw XB for all class:")
        for item in zip(FINDINGS,predx):
            print(item)
        print("\n")
        
        #print("CAM:")
        #print(cam_prob)
    
    return(original,cam_prob,proba)

#def get_avg(dataloader,model, LABEL,iterations=1000):
#    original,cam=get_next(dataloader,model, LABEL)
#    for i in range(0,1000):
        
    