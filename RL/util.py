import os
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
# USE_CUDA = torch.cuda.is_available()
# FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def to_numpy(var):
    return var.cpu().data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.float32, device='cpu'):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).to(dtype=dtype, device=device)
    # ).type(dtype)

def to_tensor2(ndarray, dtype=torch.float32, device='cpu'):
    return torch.tensor(ndarray).to(dtype=dtype, device=device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        if len(self.buffer) > self.max_size: self.buffer.pop(0)
        transition = [np.array(x) for x in transition]
        if len(transition[0].shape) < 1: return
        self.buffer.append(transition)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            # print(s.shape, a.shape, r.shape, s_.shape, d.shape)
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def sample_tensor(self, batch_size):
        s,a,r,s2,d = self.sample(batch_size)
        r = r.reshape([batch_size, 1])
        d = d.reshape([batch_size, 1])
        return to_tensor2(s), to_tensor2(a), to_tensor2(r), to_tensor2(s2), to_tensor2(d)

class ReplayBuffer_Tensor:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        if len(self.buffer) > self.max_size: self.buffer.pop(0)
        transition = [torch.tensor(x) for x in transition]
        if len(transition[0].shape) < 1: return
        self.buffer.append(transition)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            # print(s.shape, a.shape, r.shape, s_.shape, d.shape)
            state.append(s)
            action.append(a)
            reward.append(r)
            next_state.append(s_)
            done.append(d)
        state = torch.stack(state, dim=0).to(torch.float32)
        action = torch.stack(action, dim=0).to(torch.float32)
        reward = torch.stack(reward, dim=0).to(torch.float32).reshape([batch_size, 1])
        next_state = torch.stack(next_state, dim=0).to(torch.float32)
        done = torch.stack(done, dim=0).to(torch.float32).reshape([batch_size, 1])
        return state, action, reward, next_state, done
        
class ReplayBuffer_dataset(Dataset):
    def __init__(self, max_size=5e5) -> None:
        super().__init__()
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def __len__(self):
        return self.max_size
    
    def __getitem__(self, index):
        index = index % self.size
        s, a, r, s_, d = self.buffer[index]
        return s, a, r, s_, d

        
    def add(self, transition):
        if len(self.buffer) > self.max_size: self.buffer.pop(0)
        transition = [np.array(x) for x in transition]
        if len(transition[0].shape) < 1: return
        self.buffer.append(transition)
        self.size = len(self.buffer)

    

class EpReplayBuffer:
    def __init__(self, max_size=5e3, window_size=10):
        self.epbuffer = []
        self.stepbuffer = []
        self.max_size = int(max_size)
        self.size_step = 0
        self.size = 0
        self.size_index = []
        self.window_size = window_size
    
    def add_step(self, transition):
        self.size_step +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        transition = [np.array(x) for x in transition]
        self.stepbuffer.append(transition)
    
    def add_ep(self, trunc=False):
        if trunc:
            if self.size_step < self.window_size:
                return
        self.size_step = 0
        self.size += 1
        # s,a,r,s2,d = zip(*self.stepbuffer)
        ep = []
        for s in zip(*self.stepbuffer):
            s = np.array(s)
            # print(s.shape)
            ep.append(s)
            size_index = s.shape[0]
        self.epbuffer.append(ep)
        self.size_index.append(size_index)
        self.stepbuffer = []
    
    def sample_step(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.epbuffer[0:int(self.size/5)]
            del self.size_index[0:int(self.size/5)]
            self.size = len(self.epbuffer)

        state, action, reward, next_state, done = [], [], [], [], []
        
        # for i in range(batch_size):
        while len(state) < batch_size:
            _ep = np.random.randint(0,len(self.epbuffer))
            _step = np.random.randint(0, len(self.epbuffer[_ep]))
            # print(_ep, _step, len(self.epbuffer[_ep]))s
            s, a, r, s_, d = self.epbuffer[_ep]
            state.append(np.array(s[_step], copy=False))
            action.append(np.array(a[_step], copy=False))
            reward.append(np.array(r[_step], copy=False))
            next_state.append(np.array(s_[_step], copy=False))
            done.append(np.array(d[_step], copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def sample_window(self, batch_size, window_size=None):
        if window_size is None: window_size = self.window_size
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.epbuffer[0:int(self.size/5)]
            del self.size_index[0:int(self.size/5)]
            self.size = len(self.epbuffer)

        state, action, reward, next_state, done = [], [], [], [], []
        
        # for i in range(batch_size):
        while len(state) < batch_size:
            _ep = np.random.randint(0,len(self.epbuffer))
            _len = len(self.epbuffer[_ep][0]) - window_size
            if _len <= 0: continue
            _step = np.random.randint(0, _len)
            # print(_ep, _step, len(self.epbuffer[_ep]))s
            s, a, r, s_, d = self.epbuffer[_ep]
            state.append(np.array(s[_step:_step+window_size], copy=False))
            action.append(np.array(a[_step:_step+window_size], copy=False))
            reward.append(np.array(r[_step:_step+window_size], copy=False))
            next_state.append(np.array(s_[_step:_step+window_size], copy=False))
            done.append(np.array(d[_step:_step+window_size], copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
        
    def sample_step_tensor(self, batch_size):
        s,a,r,s2,d = self.sample_step(batch_size)
        return to_tensor2(s), to_tensor2(a), to_tensor2(r), to_tensor2(s2), to_tensor2(d)

    def sample_window_tensor(self, batch_size):
        s,a,r,s2,d = self.sample_window(batch_size)
        return to_tensor2(s), to_tensor2(a), to_tensor2(r), to_tensor2(s2), to_tensor2(d)

class PrioritizedReplayBuffer(EpReplayBuffer):
    def __init__(self, max_size=5000, window_size=10, alpha=0.99):
        super().__init__(max_size, window_size)
        self.alpha = alpha
        self.weighted_buffer = []

    def add_ep_with_weight(self, trunc=False):
        s,a,r,s2,d = zip(*self.stepbuffer)
        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        s2 = np.array(s2)
        d = np.array(d)

        weights = []
        w = 0
        for i in range(r.shape[0],):
            w = r[-(i+1)] + (1. - d[-(i+1)]) * self.alpha * w
            weights.append(w)
        weights = np.array(weights[::-1])

        self.epbuffer.append([s,a,r,s2,d,weights])
    
    def sample_step(self, batch_size):
        return 

    def update_R(self):
        pass


class local_buffer():
    def __init__(self, window_size=15, zero_fill=False) -> None:
        self.window_size = window_size
        self.buffer = []
        pass

    def add(self, transition):
        # transiton is tuple of (state, action, reward, next_state, done)
        transition = [np.array(x) for x in transition]
        self.buffer.append(transition)

    def out(self):
        s,a,r,s2,d = zip(*self.buffer)
        




if __name__ == '__main__':
    import time
    ee = EpReplayBuffer(window_size=13)
    b_iter = 50
    for b in range(b_iter):
        for i in range(np.random.randint(100, 200)):
            s = np.random.rand(3)
            a = np.random.rand(2)
            r = np.random.rand()
            d = np.random.rand()

            ee.add_step((s,a,r,s,d))
        ee.add_ep(trunc=True)
        print(f'  stacking steps...{b:7d}/{b_iter:7d}', end='\r')
    print()

    s,a,r,s2,d = ee.sample_window_tensor(3)
    print(s.size())

    # time_stack = []
    # n_iter = int(1e5)
    # for i in range(n_iter):
    #     _st = time.time()
    #     s,a,r,s2,d = ee.sample_step(128)
    #     _tt = time.time() - _st
    #     time_stack.append(_tt)
    #     if i % 1000 == 0:
    #         print(f'  sampling...{i:7d}/{n_iter:7d}', end='\r')
    # # print(s.shape)
    # print()
    # print(f' mean time to ep sampling:{np.mean(time_stack):0.6f}, max:{np.max(time_stack):0.6f}')
