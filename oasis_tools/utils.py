 
import torch
import numpy as np
import random

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def label_to_one_hot(label_map, num_classes, device='cuda'):
    label_map = label_map.to(device)
    bs, _, h, w = label_map.size()
    if device != 'cpu':
        input_label = torch.cuda.FloatTensor(bs, num_classes, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return input_semantics
 

def recursive_check(instance, name = "instance", recursion_depth=-1, first_element = True, colored = True, enable_print = True):
    '''
    Recursively examines any object and prints what it finds.
    Useful function for debugging.
    '''
    if enable_print:
        green_on_black = '\x1b[1;32;40m' if colored else ''
        blue_on_black = '\x1b[1;34;40m' if colored else ''
        end_clr = '\x1b[0m' if colored else '' 
        empty_block = "    "
        bar_block = green_on_black + "|" + end_clr 
        green_minus = green_on_black + "-" + end_clr 

        # for printing
        colored_name = blue_on_black + name + end_clr  

        max_str = blue_on_black + 'max:' + end_clr  
        min_str = blue_on_black + 'min:' + end_clr  
        mean_str = blue_on_black + 'mean:' + end_clr  
        var_str = blue_on_black + 'var:' + end_clr  
        device_str = blue_on_black + 'device:' + end_clr  

        if not first_element or recursion_depth>-1:
            print(empty_block*(recursion_depth+1) + bar_block )
            
        if isinstance(instance, list):
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "= List of len",len(instance))
            if len(instance)<=10:
                for i,element in enumerate(instance):
                    recursive_check(element, name + "[" + str(i) + "]" , recursion_depth+1, first_element = i==0)
            else:
                print(empty_block*(recursion_depth+1) + bar_block, "only showing first 3 elements:")
                for i,element in enumerate(instance[:3]):
                    recursive_check(element, name + "[" + str(i) + "]" , recursion_depth+1, first_element = i==0)

        elif isinstance(instance, tuple):
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "= Tuple of len",len(instance))
            for i,element in enumerate(instance):
                recursive_check(element, name + "[" + str(i) + "]" , recursion_depth+1, first_element = i==0)

        elif isinstance(instance, dict):
            sorted_keys = sorted(list(instance.keys()))
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "= Dict with",len(sorted_keys), "keys")
            
            if len(sorted_keys)<=10:
                print(empty_block*(recursion_depth+1) + bar_block,"Keys:", sorted_keys)
                for i, key in enumerate(sorted_keys):
                    recursive_check(instance[key], name + "[" + str(key) + "]" , recursion_depth+1, first_element = i==0)
            else:
                print(empty_block*(recursion_depth+1) + bar_block, "Keys:", sorted_keys[:3], "...")
                print(empty_block*(recursion_depth+1) + bar_block, "only showing first 3 elements:")
                for i, key in enumerate(sorted_keys[:3]):
                    recursive_check(instance[key], name + "[" + str(key) + "]" , recursion_depth+1, first_element = i==0)

        elif torch.is_tensor(instance):
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name,  "= Torch tensor of size", tuple(instance.size()))
            if instance.numel()>0:
                print(empty_block*(recursion_depth+1) + bar_block, f"{min_str}", np.round(instance.min().item(),4), 
                                                                f"\t{max_str}", np.round(instance.max().item(),4), 
                                                                f"\t{mean_str}", np.round(instance.float().mean().item(),4), 
                                                                f"\t{var_str}", np.round(instance.float().var().item(),4), 
                                                                f"\t{device_str}", instance.device)

        elif type(instance).__module__=='numpy':
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "= Numpy tensor of size", instance.shape)
            print(empty_block*(recursion_depth+1) + bar_block, f"{min_str}", np.round(instance.min(),4), f"\t{max_str}", np.round(instance.max(),4))
        elif isinstance(instance,int) or isinstance(instance,str):
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "=", instance)
        elif isinstance(instance,float):
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "=", np.round(instance,4))
        else:
            instance_name = instance.__class__
            print(empty_block*(recursion_depth+1) + bar_block +green_minus+ colored_name, "=", instance_name)
 