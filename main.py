import time
import os
import argparse

import torch
from torch.autograd import Variable
from torch import optim

import PIL
import matplotlib.pyplot as plt

from utility.utility import *
from utility.vgg_network import *
from utility.loss_fns import *

from tqdm import tqdm

#############################################################################
# PARSER
parser = argparse.ArgumentParser(description='CNNMRF')
# parser for image
parser.add_argument('-image_size', type=int, default=256)
# parser for weights
parser.add_argument('-alpha', type=float, default=1e-1, help='Weight for content')
parser.add_argument('-beta',  type=float, default=1e-1, help='Weight for style')
parser.add_argument('-gamma', type=float, default=1e-6, help='Weight for mrf')
parser.add_argument('-delta', type=float, default=1e2,  help='Weight for distance loss')
# parser for patch size
parser.add_argument('-patch_size', '-patch_size', type=int, default=5, help='Patch size')
# parser for input images paths and names
parser.add_argument('-content_path',type=str, default='./inputs/contents/Swallow-Silhouette.jpg', help='Path to content image')
parser.add_argument('-style_path',    type=str, default='./inputs/styles/delicate.jpg', help='Path to content image')
# parser for output path
parser.add_argument('-output_dir', type=str, default='./sample_outputs/', help='Path to save output files')
# parser for cuda
parser.add_argument('-cuda', type=int, default=0, help='gpu # or -1 for cpu')
# parser for number of iterations
parser.add_argument('-epoch', type=int, default=5000, help='Number of iterations to run')

args = parser.parse_args()
#############################################################################

# Get image paths
# Content
content_path = args.content_path
content_dir = os.path.dirname(content_path)+'/'
content_name = os.path.basename(content_path)

# Style
style_path = args.style_path
style_dir = os.path.dirname(style_path)+'/'
style_name = os.path.basename(style_path)

# Parameters
alpha = args.alpha
beta = args.beta
gamma = args.gamma
delta = args.delta
image_size = args.image_size
patch_size = args.patch_size
content_invert = True
style_invert = True
result_invert = True

# Cuda device
if args.cuda >= 0  and torch.cuda.is_available:
    device = f'cuda:{args.cuda}'
else:
    device = 'cpu'
print("Using device: ", device)
    
# Get output path
output_dir = args.output_dir
try:
    os.mkdir(output_dir)
except:
    pass
output_dir = output_dir + content_name[:-4] + '_' + style_name[:-4] + '/'

# Create output directory
try:
    os.mkdir(output_dir)
except:
    pass

start = time.time()

# Get network
vgg = VGG()
vgg.load_state_dict(torch.load('vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.to(device)

content_img = load_image(content_path, image_size, device, content_invert)
style_img = load_image(style_path, image_size, device, style_invert)

# Random initialization
opt_img = Variable(torch.randn(content_img.size()).type_as(content_img.data).to(device), requires_grad=True).to(device)
# Content initialization
# opt_img = Variable(content_img.data.clone(), requires_grad=True)

save_images(content_img.data[0].cpu().squeeze(), style_img.data[0].cpu().squeeze(), opt_img.data[0].cpu().squeeze(), image_size, output_dir, 0, content_invert, style_invert, result_invert)

# Define style layers
style_layers = ['r11','r21','r31','r41','r51']
style_weights = [1e3/n**3 for n in [64,128,256,512,512]]
# Define mrf layers
mrf_layers = ['r31', 'r41'] 
# Defince content layers
content_layers = ['r42']
content_weights = [1e0]
# loss layers: layers to be used by opt_img ( style_layers & mrf_layers & content_layers)
loss_layers = mrf_layers + style_layers + content_layers

# Feature maps from style images
mrf_fms = [A.detach() for A in vgg(style_img, mrf_layers)]
# Extract style patches & create conv3d from those patches
style_patches_lists, weight_list = get_style_patch_weights(mrf_fms, device, k=patch_size)

# Compute style target
style_targets = [GramMatrix()(A).detach() for A in vgg(style_img, style_layers)]
# Computer content target
content_targets = [A.detach() for A in vgg(content_img, content_layers)]
# targets
targets = style_targets + content_targets
# layers weights
weights = style_weights + content_weights
# Opt layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

# Prepare distance transform loss template
cont_dist = dist_cv2(content_img.clone(), device, image_size, content_invert)
cont_dist = cont_dist**6
cont_dist[cont_dist>1e3] = 1e3
cont_dist[cont_dist==float("Inf")] = 1e3
dist_template = cont_dist*content_img


# Define optimizer
optimizer = optim.LBFGS([opt_img])

n_iter = [0]
naming_it = [0]
loss_list = []
content_loss_list = []
style_loss_list = []
mrf_loss_list = []
dist_loss_list = []
max_iter = args.epoch
show_iter = 100

start_res = time.time()

pbar = tqdm(total=max_iter)
while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        opt_fms = vgg(opt_img, loss_layers)
        
        # Content & style loss
        style_loss = 0
        content_loss = 0
        for a,A in enumerate(opt_fms[len(mrf_layers):]):
            one_layer_loss = weights[a] * loss_fns[a](A, targets[a])
            if a < len(style_layers):
                style_loss += one_layer_loss
            else:
                content_loss += one_layer_loss

        # MRF loss
        mrf_loss = mrf_loss_fn(opt_fms[:len(mrf_layers)], style_patches_lists, weight_list,patch_size)
        
        # Distance transform loss
        d_temp = cont_dist*opt_img.clone()
        dist_loss = nn.MSELoss().to(device)(d_temp, dist_template)
              
        # Regularzier
        regularizer = smoothnes_loss(opt_img)

        # Total loss
        total_loss = alpha * content_loss + beta * style_loss + gamma * mrf_loss + delta * dist_loss + 0.001*regularizer
 
        # log 
        content_loss_list.append(content_loss.item())
        style_loss_list.append(style_loss.item())
        mrf_loss_list.append(mrf_loss.item())
        dist_loss_list.append(dist_loss.item())
        loss_list.append(total_loss)

        # Calculate backward
        total_loss.backward()

        #print loss
        if (n_iter[0])%show_iter == 0:

            tqdm.write('Iteration: {}'.format(naming_it[0]))
            tqdm.write('Content loss : {}'.format(alpha*content_loss.item()))
            tqdm.write('Style loss   : {}'.format(beta *style_loss.item()))
            tqdm.write('MRF loss     : {}'.format(gamma*mrf_loss.item()))
            tqdm.write('Dist loss    : {}'.format(delta*dist_loss.item()))
            tqdm.write('Regulari loss: {}'.format(0.001*regularizer.item()))
            tqdm.write('Total loss   : {}'.format(total_loss.item()))

            # Save loss graph
            save_plot(loss_list, label='total loss',output_path=output_dir)
            save_plot(content_loss_list, label='content loss',output_path=output_dir)
            save_plot(style_loss_list, label='style loss',output_path=output_dir)
            save_plot(mrf_loss_list, label='mrf loss',output_path=output_dir)
            save_plot(dist_loss_list, label='dist loss', output_path=output_dir)

            # Save optimized image
            out_img = postp(opt_img.data[0].cpu().squeeze(), image_size, result_invert)
            out_img.save(output_dir + '{}.bmp'.format(naming_it[0]))

        n_iter[0] += 1
        naming_it[0] += 1
        pbar.update(1)
        return total_loss
    optimizer.step(closure)
pbar.close()
# Save summary image
save_images(content_img.data[0].cpu().squeeze(), style_img.data[0].cpu().squeeze(), opt_img.data[0].cpu().squeeze(), image_size, output_dir, naming_it[0], content_invert, style_invert, result_invert)

end = time.time()
print("Style transfer took {} seconds overall".format(end-start))