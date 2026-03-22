import torch                                                                                                                          
from transformers import AutoModelForCausalLM, AutoTokenizer                                                                          
                                                                                                                                        
  # 1. 加载原始 HF 模型                                                                                                                 
model = AutoModelForCausalLM.from_pretrained(                                                                                         
    "../OLMo-1B",                                                                                                                     
    trust_remote_code=True,                                                                                                           
    torch_dtype=torch.bfloat16,                                                                                                       
    device_map="cuda"                                                                                                                 
)                                                                                                                                     
                                                                                                                                    
tokenizer = AutoTokenizer.from_pretrained("../OLMo-1B", trust_remote_code=True)                                                       
                                                                                                                                    
# 2. 从训练数据加载第一个 batch（需要根据您的数据加载代码调整）                                                                       
# 或者使用简单的测试数据                                                                                                              
test_texts = [                                                                                                                        
    "The quick brown fox jumps over the lazy dog. " * 50,                                                                             
    "In the beginning, there was nothing but darkness. " * 50,                                                                        
]                                                                                                                                     
                                                                                                                                    
losses = []                                                                                                                           
for text in test_texts:                                                                                                               
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to("cuda")                                        
                                                                                                                                    
    with torch.no_grad():                                                                                                             
        outputs = model(**inputs, labels=inputs["input_ids"])                                                                         
        losses.append(outputs.loss.item())                                                                                            
        print(f"Text: {text[:50]}...")                                                                                                
        print(f"Loss: {outputs.loss.item():.4f}\n")                                                                                   
                                                                                                                                    
print(f"\n平均 loss: {sum(losses)/len(losses):.4f}")                                                                                  
print(f"训练起始 loss: 4.2891")                                                                                                       
print(f"差异: {abs(sum(losses)/len(losses) - 4.2891):.4f}")