import time
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from models import LNet, DNet, ENet  # 假设LNet, DNet, ENet分别为Lip-Sync网络、表达式编辑网络、身份增强网络
from dataset import  # 在此留空，假设这里是用于数据加载的类
import util
import torch.nn.functional as F
from torch import nn

def to_np(x):
    """将Tensor转换为Numpy数组，便于后续的结果分析"""
    return x.data.cpu().numpy()

# 设定参数和路径
opt = Config().parse()  # 从配置文件加载训练参数
writer = SummaryWriter(comment=opt.name)  # 使用TensorBoard记录训练日志，便于后期监控
train_data_path = os.path.join(opt.main_PATH, 'train')  # 训练数据集路径
val_data_path = os.path.join(opt.main_PATH, 'val')  # 验证数据集路径

# 加载数据集
train_dataset =  # 此处加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers, drop_last=True)
val_dataset =  # 此处加载验证数据集
val_dataloader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, drop_last=False)

# 输出数据集大小
dataset_size = len(train_dataset)
print(f'#training images = {dataset_size}')  # 输出训练集的大小

# 初始化模型
model_L = LNet()  # 初始化Lip-Sync网络模型
model_D = DNet()  # 初始化表达式编辑网络模型，用于去除信息泄漏
model_E = ENet()  # 初始化身份增强网络模型

# 加载之前训练的模型检查点
if opt.resume:
    model_L, start_step, start_epoch = util.load_checkpoint(opt.resume_path, model_L)  # 从指定路径加载Lip-Sync模型检查点
else:
    model_L = util.load_separately(opt, model_L)  # 如果没有检查点，直接初始化Lip-Sync模型

# 初始化优化器
optimizer_L = Adam(model_L.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 为Lip-Sync网络初始化Adam优化器
optimizer_D = Adam(model_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 为DNet网络初始化Adam优化器
optimizer_E = Adam(model_E.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # 为ENet网络初始化Adam优化器

# 定义损失函数
# 根据论文中D-Net的描述，使用感知损失和Gram矩阵损失，具体实现如下：
def perceptual_loss(pred, target):
    """使用感知损失（Perceptual Loss）衡量生成图像与真实图像之间的差异"""
    loss = nn.MSELoss()(pred, target)
    return loss

def gram_matrix_loss(pred, target):
    """使用Gram矩阵损失（Gram Matrix Loss）来捕捉图像的风格信息"""
    pred_gram = gram_matrix(pred)
    target_gram = gram_matrix(target)
    return nn.MSELoss()(pred_gram, target_gram)

def gram_matrix(x):
    """计算Gram矩阵"""
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram / (c * h * w)
    return gram

def l1_loss(pred, target):
    """使用像素级L1损失（L1 Loss）来计算预测与真实图像之间的绝对差异"""
    return nn.L1Loss()(pred, target)

def adversarial_loss(pred, target, model):
    """对抗性损失（Adversarial Loss），用于训练生成网络"""
    return -torch.mean(pred)  # 基本的对抗性损失计算

def lipsync_discriminator_loss(pred, target, model):
    """唇动同步判别器损失（Lip-Sync Discriminator Loss），用于评估图像的同步质量"""
    return nn.MSELoss()(pred, target)

def identity_loss(pred, target):
    """身份损失（Identity Loss），计算生成图像与目标图像的身份差异"""
    return nn.MSELoss()(pred, target)  # 假设使用MSE作为身份损失的度量

# 训练循环
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()  # 记录每个epoch开始的时间
    epoch_iter = 0  # 记录每个epoch的迭代次数

    # 训练循环，逐步处理每个batch数据
    for i, (data, label) in enumerate(train_dataloader):
        iter_start_time = time.time()  # 记录每个迭代开始的时间
        total_steps += opt.batchSize  # 更新总步数
        epoch_iter += opt.batchSize  # 更新当前epoch的步数

        input_video, audio_data = data  # 提取输入视频和音频数据
        model_L.set_input(input_video, audio_data, label)  # 将输入数据传入Lip-Sync网络
        model_L.optimize_parameters()  # 训练Lip-Sync网络

        model_D.set_input(input_video, audio_data, label)  # 将输入数据传入DNet网络
        model_D.optimize_parameters()  # 训练DNet网络，去除表达式泄漏

        model_E.set_input(input_video, audio_data, label)  # 将输入数据传入ENet网络
        model_E.optimize_parameters()  # 训练ENet网络，进行身份增强

        # 计算损失
        loss_D = perceptual_loss(model_D.prediction, model_D.target) + gram_matrix_loss(model_D.prediction, model_D.target)  # D-Net损失
        loss_L = perceptual_loss(model_L.prediction, model_L.target) + adversarial_loss(model_L.prediction, model_L.target, model_L)  # Lip-Sync网络的感知损失和对抗性损失
        loss_E = perceptual_loss(model_E.prediction, model_E.target) + l1_loss(model_E.prediction, model_E.target) + identity_loss(model_E.prediction, model_E.target)  # ENet的感知损失、L1损失和身份损失

        # 记录训练日志
        model_L.TfWriter(writer, total_steps)  # 记录Lip-Sync网络的训练日志
        model_D.TfWriter(writer, total_steps)  # 记录DNet网络的训练日志
        model_E.TfWriter(writer, total_steps)  # 记录ENet网络的训练日志

        # 每隔一定步数显示当前的训练结果
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model_L.get_current_visuals(), epoch)  # 显示Lip-Sync网络的当前结果
            visualizer.display_current_results(model_D.get_current_visuals(), epoch)  # 显示DNet网络的当前结果
            visualizer.display_current_results(model_E.get_current_visuals(), epoch)  # 显示ENet网络的当前结果

        # 每隔一定步数打印当前的训练误差
        if total_steps % opt.print_freq == 0:
            errors_L = model_L.get_current_errors()  # 获取Lip-Sync网络的当前误差
            errors_D = model_D.get_current_errors()  # 获取DNet网络的当前误差
            errors_E = model_E.get_current_errors()  # 获取ENet网络的当前误差
            t = (time.time() - iter_start_time) / opt.batchSize  # 计算每个batch的处理时间
            visualizer.print_current_errors(epoch, epoch_iter, errors_L, t)  # 打印Lip-Sync网络的训练误差
            visualizer.print_current_errors(epoch, epoch_iter, errors_D, t)  # 打印DNet网络的训练误差
            visualizer.print_current_errors(epoch, epoch_iter, errors_E, t)  # 打印ENet网络的训练误差

        # 每隔一定步数进行验证
        if total_steps % opt.eval_freq == 0:
            evaluation(val_dataloader, model_L, total_steps, writer=writer)  # 验证Lip-Sync网络
            evaluation(val_dataloader, model_D, total_steps, writer=writer)  # 验证DNet网络
            evaluation(val_dataloader, model_E, total_steps, writer=writer)  # 验证ENet网络

        # 每隔一定步数保存当前模型
        if total_steps % opt.save_latest_freq == 0:
            print(f'{opt.name} saving the latest model (epoch {epoch}, total_steps {total_steps})')
            util.save_checkpoint({
                'step': total_steps,
                'epoch': epoch,
                'model_L': model_L.state_dict(),
                'model_D': model_D.state_dict(),
                'model_E': model_E.state_dict(),
                'optimizer_L': optimizer_L.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'optimizer_E': optimizer_E.state_dict()
            }, epoch)

    # 结束一个epoch后的输出
    print(f'{opt.name} End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec')

    # 更新学习率
    if epoch > opt.niter:
        model_L.update_learning_rate()  # 更新Lip-Sync网络的学习率
        model_D.update_learning_rate()  # 更新DNet网络的学习率
        model_E.update_learning_rate()  # 更新ENet网络的学习率

    # 在测试集上评估
    test_data_path = os.path.join(opt.main_PATH, 'test')
    test_dataset =  # 在此加载测试数据集
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, drop_last=False)
    evaluation(test_dataloader, model_L, total_steps, writer=writer)  # 测试Lip-Sync网络
    evaluation(test_dataloader, model_D, total_steps, writer=writer)  # 测试DNet网络
    evaluation(test_dataloader, model_E, total_steps, writer=writer)  # 测试ENet网络
