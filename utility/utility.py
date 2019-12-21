import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
from scipy import ndimage
import cv2

import matplotlib.pyplot as plt



# post processing for images
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                            # transforms.Normalize(mean=[0,0,0], #subtract imagenet mean
                            #                         std=[1/0.5,1/0.5,1/0.5]),
                            # transforms.Normalize(mean=[-0.5,-0.5,-0.5], #subtract imagenet mean
                            #                         std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        Fe = input.view(b, c, h*w)
        G = torch.bmm(Fe, Fe.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

def postp(tensor, image_size, invert): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    if invert:
        img = PIL.ImageOps.invert(img)
    img = transforms.functional.resize(img,[image_size, image_size])
    return img

# Function to load images
def load_image(img_dir, img_size, device, invert):
    prep = transforms.Compose([transforms.Resize((img_size,img_size)),
                            # transforms.RandomRotation(angle),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                    std=[1,1,1]),
                        #    transforms.Normalize(mean=[0.5, 0.5, 0.5], #add imagenet mean
                        #                         std=[0.5,0.5,0.5]),
                            transforms.Lambda(lambda x: x.mul_(255)),
                            ])
    # Load & invert image
    image = Image.open(img_dir)
    if invert:
        image = PIL.ImageOps.invert(image)
    image = image.convert('RGB')
    # Make torch variable
    img_torch = prep(image)
    # Use cuda if available
    # if torch.cuda.is_available():
    #     img_torch = Variable(img_torch.unsqueeze(0).cuda())
    # else:
    #     img_torch = Variable(img_torch.unsqueeze(0))
    img_torch = Variable(img_torch.unsqueeze(0).to(device))
    
    return img_torch

# Function to save images
def save_images(content_image, style_image, opt_img, image_size, output_path, n_iter, content_invert, style_invert, result_invert):

    style_image = postp(style_image, image_size, style_invert)
    style_image.save(output_path + 'style.bmp')

    content_image = postp(content_image, image_size, content_invert)
    content_image.save(output_path + 'content.bmp')

    # Save optimized images
    out_img = postp(opt_img, image_size, result_invert)
    # out_img.save(output_path + '/{}.bmp'.format(n_iter))
        
    # Save summary image as [content image, optimized image, style image]
    images = [content_image, style_image, out_img]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_path + '/all.bmp')
    
"""
Input tensor
Outputs distance transform
"""
def dist_cv2(input_tensor, device, image_size, content_invert):
    out_img = postp(input_tensor.data[0].cpu().squeeze(), image_size, content_invert)
    out_img = out_img.convert('L')

    img = np.asarray(out_img)
    
    img = ndimage.grey_erosion(img, size=(3,3))

    img_dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    cont_dist = torch.from_numpy(img_dist).float().to(device)
    f = cont_dist.unsqueeze(0)
    a = torch.cat((f,f,f),0)
    a = a.unsqueeze(0)    
    return a

def save_plot(loss_list, label, output_path):
    plt.plot(loss_list, label=label)
    plt.legend()
    plt.savefig(f'{output_path}{label}.jpg')
    plt.close()