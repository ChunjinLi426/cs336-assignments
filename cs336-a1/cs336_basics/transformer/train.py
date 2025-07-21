from cs336_basics.transformer.train_utils import *


def set_seed(seed: int = 20060321):
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 的种子
    np.random.seed(seed)  # numpy 随机种子
    # random.seed(seed)  # Python 随机种子

    # CUDA 算法稳定性保证（有时训练速度会变慢）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def learning_rate_tuning_experiment(): 
    set_seed()
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr = 1e3)
    for t in range(10): 
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

def main(): 
    learning_rate_tuning_experiment()


if __name__ == "__main__": 
    main()
    