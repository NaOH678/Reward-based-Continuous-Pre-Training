import torch                                                                                                                          
import torch.distributed as dist                                                                                                      
import torch.distributed.checkpoint as DCP                                                                                            
import os                                                                                                                             
                                                                                                                                        
if "RANK" in os.environ:                                                                                                              
      dist.init_process_group(backend="gloo")                                                                                           
                                                                                                                                        
state_dict = {}                                                                                                                       
DCP.load(state_dict, checkpoint_id="../OLMo-1B/checkpoint_final/checkpoint/step-0")                                                              
                                                                                                                                    
print(f"✅ Loaded {len(state_dict)} keys")                                                                                            
print(f"✅ Total parameters: {sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)):,}")                         
                                                                                                                                    
if dist.is_initialized():                                                                                                             
    dist.destroy_process_group() 