import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self , d_k):
    super().__init__()
    self.d_k = d_k # 输入特征的维度
    
  def forward(self , Q , K , V , mask = None):
      # 计算注意力分数
      # K.transpose(-2 , -1)转置操作,交换最后两个维度的位置
      scorces = torch.matmul(Q , K.transpose(-2 , -1)) / (self.d_k ** 0.5)
      
      if mask is not None:
        scorces = scorces.masked_fill(mask == 0 , -1e9)
        
      attn = F.softmax(scorces , dim = -1)
      
      output = torch.matmul(attn , V)
      
      return output , attn

class MutliAttention(nn.Module):
  def __init__(self , d_model , n_head):
    super().__init__()
    self.d_model = d_model
    self.n_head  = n_head
    self.d_k = d_model // n_head
    
    # 线性变化层生成Q , K , V
    # 线性变换层的维度变化：[batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
    self.W_Q = nn.Linear(d_model , d_model)
    self.W_K = nn.Linear(d_model , d_model)
    self.W_V = nn.Linear(d_model , d_model)
    
    # 初始化最后的线性层
    self.W_O = nn.Linear(d_model , d_model)
    
    self.attention = Attention(self.d_k)
    
  def forward(self , Q , K , V , mask = None):
      # 线性变化Q , K , V
    Q = self.W_Q(Q)
    K = self.W_K(K)
    V = self.W_V(V)
    
    # 分头操作，需要变化维度，这是Mutli的核心
    # [batch_size , seq_len , d_model] -> [batch_size , n_head , seq_len , d_k]     
    bach_size = Q.size(0)
    Q = Q.view(bach_size , -1 , self.n_head , self.d_k).transpose(1 , 2)
    K = K.view(bach_size , -1 , self.n_head , self.d_k).transpose(1 , 2)
    V = V.view(bach_size , -1 , self.n_head , self.d_k).transpose(1 , 2)
    
    # 计算注意力
    # [batch_size , n_head , seq_len , d_k]
    output , attn = self.attention(Q , K , V , mask = mask)
    
    # 拼接多头结果并变换回原维度
    # [batch_size , n_head , seq_len , d_k] -> [batch_size , seq_len , d_model]
    # output = self.W_O(output.permute(0, 2, 1, 3).reshape(bach_size, -1, self.d_model))
    
    # 标准写法
    output = output.transpose(1 , 2).contiguous().view(bach_size, - 1 , self.d_model)
    output = self.W_O(output)
    
    return output , attn


#! 关于掩码结合我再自动驾驶有关掩码的应用中可以学习

# 创建填充掩码      
def create_padding_mask(seq, pad_idx=0):
    # seq: [batch_size, seq_len], pad_idx 是填充符号的索引
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask  # 有效位置为 True，填充位置为 False    

# 创建未来掩码  
def create_future_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 1  # 上三角为 True，未来位置被屏蔽
      
if __name__ == "__main__":
  # 测试代码
  d_model = 512
  n_head = 8
  batch_size = 2
  seq_len = 10
  
  Q = torch.rand(batch_size, seq_len, d_model)
  K = torch.rand(batch_size, seq_len, d_model)
  V = torch.rand(batch_size, seq_len, d_model)
  
  attention_layer = MutliAttention(d_model, n_head)
  
  output, attn = attention_layer(Q, K, V)
  
  print("Output shape:", output.shape)  # 应该是 [batch_size, seq_len, d_model]
  print("Attention shape:", attn.shape)  # 应该是 [batch_size, n_head, seq_len, seq_len]
     
    