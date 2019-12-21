import torch
import torch.nn.functional as F

"""
Inputs feature maps from all given layers.
Outputs list of patches.
patch_size: [Channel,k,k]
"""
def get_patches(feature_map, k=5, stride=1):
    # list of all patches from a feature map
    patches = []
    for i in range(0,feature_map.shape[3] - k, stride):
        for j in range(0,feature_map.shape[3] - k, stride):
            patch = feature_map[:,:,i:i+k,j:j+k]
            patches.append(patch)
    return patches
def divide_patches(style_patches):
        style_patches_lists = [[],[],[],[]]
        for sp in style_patches:
            if sp.shape[1] == 64:
                style_patches_lists[0].append(sp)
            elif sp.shape[1] == 128:
                style_patches_lists[1].append(sp)
            elif sp.shape[1] == 256:
                style_patches_lists[2].append(sp)
            elif sp.shape[1] == 512:
                style_patches_lists[3].append(sp)
        # Remove empty lists
        style_patches_lists = [x for x in style_patches_lists if x !=[]]
        return style_patches_lists
def weight_maker(style_patches,k,device):
        weights = torch.zeros(size=(len(style_patches),style_patches[0].shape[1],k,k))
        weights = weights.to(device)
        for i in range(len(style_patches)):
            weights[i,:,:,:] = style_patches[i].clone()
            weights[i,:,:,:] = weights[i,:,:,:] / torch.sqrt((weights[i,:,:,:]**2).sum())
        return weights

"""
Inputs feature map of style image
Outputs channel-wise tensors, conv3d weights
"""
def get_style_patch_weights(style_fms, device, k=5):
    # Sort based on the channel size
    style_fms = sorted(style_fms, key=lambda x:x.shape[1])
    # Extract patches from feature maps
    style_patches = []
    for style_fm in style_fms:
        one_layer_patches = get_patches(style_fm,k)
        style_patches += one_layer_patches
    # Divide patches by channel sizes
    style_patches_lists = divide_patches(style_patches)
    # Create weights to use for convolution, in per channel size
    weight_list = [weight_maker(style_plist,k,device) for style_plist in style_patches_lists]
    return style_patches_lists, weight_list

"""
Style loss function from Eq.2.
Inputs optimizing images's feature maps & patches from style image.
Outputs style loss.
"""
def mrf_loss_fn(opt_fms,style_patches_lists,weight_list,k=5):
    # Loss variable
    E_s = 0
    # Sort based on the channel size
    opt_fms = sorted(opt_fms, key=lambda x:x.shape[1])
    # get opt patches
    opt_patches = []
    for opt_fm in opt_fms:
        one_layer_patches = get_patches(opt_fm,k)
        opt_patches += one_layer_patches
    # Divide patches by channel sizes
    opt_patches_lists = divide_patches(opt_patches)
    
    for i,opa_list in enumerate(opt_patches_lists):
        batch_size = 16
        for j in range(0,len(opa_list),batch_size):
            new = torch.cat(opt_patches_lists[i][j:j+batch_size])
            new = new.unsqueeze(2)
            out = F.conv3d(new,weight_list[i].unsqueeze(2),stride=1,padding=0,dilation=1,groups=1)
            out.squeeze_()
            try:
                _,argmaxes = F.max_pool1d(out.unsqueeze_(0),max(out.shape),return_indices=True)
            except:
                _,argmaxes = F.max_pool1d(out.unsqueeze(0),max(out.shape),return_indices=True)
            argmaxes.squeeze_()
            if not argmaxes.shape:
                argmaxes.unsqueeze_(0)
            for ind,arg in enumerate(argmaxes):
                E_s += torch.sum((opa_list[j+ind] - style_patches_lists[i][arg])**2)
    
    return E_s

"""
Inputs feature maps of optimizing image and content image.
Outputs content loss.
"""
def content_loss_fn(opt_fms, content_fms):
    c_loss = 0
    for i, opt_fm in enumerate(opt_fms):
        c_loss += ((opt_fm - content_fms[i])**2).sum()
    return c_loss

"""
Inputs an image as pytorch tensor[d;c;h;w]
Outputs gamma from equation 5.
"""
def smoothnes_loss(x):
    # make padded tensor
    p2d = (0, 1, 0, 1) # pad one row and column, below and after.
    padded = F.pad(x, p2d, 'constant', 0)
    # make (i-1) tensor
    temp = x[:,:,x.shape[2]-1,:].clone()
    a = padded[:,:,1:, :-1].clone()
    a[:,:,x.shape[2]-1,:] = temp.clone()
    # make  (j-1) tensor
    temp = x[:,:,:,x.shape[3]-1].clone()
    b = padded[:,:,:-1,1:].clone()
    b[:,:,:,x.shape[3]-1]= temp.clone()
    # calculate regularization loss
    reg_loss = ((x-a)**2 + (x-b)**2).sum()
    return reg_loss

