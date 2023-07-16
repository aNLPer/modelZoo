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



x = torch.LongTensor([[0,1,2,3]])

class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.emb = torch.nn.Embedding(5, 10)
        self.linear = torch.nn.Linear(10, 2)
        self.l = torch.nn.CrossEntropyLoss()
    def forward(self, x):
        emb_x = self.emb(x)
        emb_x = torch.sum(emb_x, dim=-2)
        out = self.linear(emb_x)
        loss = self.l(out, torch.tensor([0]))
        return loss

m = mymodel()
fgm = FGM(m)


loss = m(x)
loss.backward()
fgm.attack()

