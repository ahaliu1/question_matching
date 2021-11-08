import paddle


class FGM_paddle():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        optimizer.clear_grad()
    '''

    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and self.emb_name in name:
                self.backup[name] = param.clone()
                norm = paddle.norm(param.grad)
                if norm != 0 and not paddle.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param += r_at

    def restore(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and self.emb_name in name:
                assert name in self.backup
                param = self.backup[name]
        self.backup = {}
