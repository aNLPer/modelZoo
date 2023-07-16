import torch


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad) # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

model = 0 # model
data = 0 # dataset

# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
  # 正常训练
  loss = model(batch_input, batch_label)
  loss.backward() # 反向传播，得到正常的grad
  # 对抗训练
  fgm.attack() # embedding被修改了
  # optimizer.zero_grad() # 如果不想累加梯度，就把这里的注释取消
  loss_sum = model(batch_input, batch_label)
  loss_sum.backward() # 反向传播，在正常的grad基础上，累加对抗训练的梯度
  fgm.restore() # 恢复Embedding的参数
  # 梯度下降，更新参数
  optimizer.step()
  optimizer.zero_grad()