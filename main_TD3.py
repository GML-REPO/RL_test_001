import os
import torch
import numpy as np
import time
import argparse

import gym
from Envs.baseline import BaselineEnv

from RL.TD3 import TD3

from RL.util import prBlack, prCyan, prGreen, prLightGray, prLightPurple, prPurple, prRed, prYellow, ReplayBuffer, ReplayBuffer_Tensor
# from utils import visualizer_visdom
from torch.utils.tensorboard import SummaryWriter

CWD = os.path.dirname(os.path.abspath(__file__))
def to_numpy(x:torch.Tensor):
    return x.detach().cpu().numpy()

def parsing():
    description_text = '''
    test run file
    '''
    parser = argparse.ArgumentParser(description=description_text, conflict_handler='resolve')
    parser.add_argument("--name", help="name",
                    type=str, default='NP')
    
    parser.add_argument("--gpu", help="train or test",
                    type=str, default='0')
    parser.add_argument("--task", help="train or test",
                    type=str, default='train')
    
    parser.add_argument("--print_ep", help="train or test",
                    type=int, default=100)
    parser.add_argument("--ckpt", help="train or test",
                    type=str, default='')
    
    parser.add_argument("--log_dir", help="targetPos, dPos, dIK, torque",
                    type=str, default='')
    parser.add_argument("--GUI", help="targetPos, dPos, dIK, torque",
                    type=int, default=0)
                    
    parser.add_argument("--EP_MAX", help="train or test",
                    type=int, default=1e4)
    parser.add_argument("--STEP_MAX", help="train or test",
                    type=int, default=2e3)
    parser.add_argument("--STEP_SAVE", help="train or test",
                    type=int, default=1e4)
    parser.add_argument("--BUFFER_SIZE", help="train or test",
                    type=float, default=5e5)
                    
    parser.add_argument("--batch_size", help="train or test",
                    type=int, default=128)

    parser.add_argument("--gamma", help="discount for future rewards",
                    type=float, default=0.995)
    parser.add_argument("--pNoise", help="target policy smoothing noise",
                    type=float, default=0.1)
    parser.add_argument("--eNoise", help="exploration noise",
                    type=float, default=0.2)
    parser.add_argument("--pDelay", help="delayed policy updates parameter",
                    type=int, default=2)
    parser.add_argument("--polyak", help="target policy update parameter (1-tau)",
                    type=float, default=0.995 )
                    

    parser.add_argument('--update_after', type=int, default=1000)
    parser.add_argument('--update_every', type=int, default=100)
    
    args = parser.parse_args()
    args.BUFFER_SIZE = int(args.BUFFER_SIZE)
    return args


def train(args, agent:TD3, env:gym.Env, debug=True, writer=None):
    log_dir = args.log_dir

    random_seed = 0
    gamma = args.gamma                # discount for future rewards
    batch_size = args.batch_size            # num of transitions sampled from replay buffer
    exploration_noise = args.eNoise 
    polyak = args.polyak              # target policy update parameter (1-tau)
    policy_noise = args.pNoise          # target policy smoothing noise
    noise_clip = 0.1
    policy_delay = args.pDelay            # delayed policy updates parameter
    max_episodes = args.EP_MAX         # max num of episodes
    max_timesteps = args.STEP_MAX        # max timesteps in one episode

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    replay_buffer = ReplayBuffer_Tensor(args.BUFFER_SIZE)


    test_env = env

    done_cnt = 0

    ref_val = -np.inf

    save_path = log_dir#CWD + '/ckpt/'+args.name
    os.makedirs(save_path, exist_ok=True)

    wamp = 10000

    ep_reward = 0
    int_ep_reward = 0
    ema_r, ema_s = 0,0
    ema_init = True
    ep_cnt = 1
    eval_flag = False

    _step = 0
    
    state = env.reset()
    print(f' start training...')

    for total_step in range(1, int(max_episodes*max_timesteps)):
        start_time = time.time()
        total_step += 1
        _step += 1

        if total_step > wamp:
            action = agent.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(-agent.max_action, agent.max_action)
        else:
            action = env.action_space.sample()


        # env response with next_observation, reward, terminate_info
        next_state, reward, done, _ = env.step(action)
        
        ep_reward += reward
        done = False if _step == max_timesteps else done


        replay_buffer.add((state, action, reward, next_state, float(done)))

        state = next_state

        
        end_time = time.time()

        if done or (_step == max_timesteps):
            if debug and ep_cnt % args.print_ep == 0:
                prGreen(f'#{ep_cnt:7d}: episode_reward:{ep_reward:>10.4f} steps:{total_step:7d} |'+
                    f'c_step:{_step:7d} |'+
                        f'TPS: {(end_time-start_time)*1000:.1f} ms')

            writer.add_scalar('train_spec/reward', ep_reward, total_step)
            writer.add_scalar('train_spec/steps', _step, total_step)
            writer.add_scalar('train_spec/reward per steps', ep_reward/_step, total_step)
            
            if ema_init:
                ema_r, ema_s = ep_reward, _step
                ema_init = False
            else:
                ema_r = ema_r * 0.9 + 0.1 * ep_reward
                ema_s = ema_s * 0.5 + 0.5 * _step
            _step = 0
            ep_cnt += 1
            ep_reward = 0
            state = env.reset()
            eval_flag = True
            

        if total_step >= args.update_after and total_step % args.update_every == 0:
            loss_q1, loss_q2, loss_pi = agent.update(replay_buffer, args.update_every, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)

            writer.add_scalar('PG_spec/Loss_Q1', loss_q1, total_step)
            writer.add_scalar('PG_spec/Loss_Q2', loss_q2, total_step)
            writer.add_scalar('PG_spec/Loss_PI', loss_pi, total_step)

        if total_step % args.STEP_SAVE == 0:
            agent.save(save_path, 'regular')

        if ep_cnt % 50 == 0 and eval_flag:
            eval_flag = False
            with torch.no_grad():
                r_eval = []
                for i in range(10):
                    rr = evaluate(args, agent, test_env, ESSP=None)
                    r_eval.append(rr)
            r_mean = np.mean(r_eval)
            writer.add_scalar('eval_spec/eval_rewards', r_mean, ep_cnt)
            
            if r_mean > ref_val:
                ref_val = r_mean
                prLightPurple(f'##{ep_cnt:>7d}: episode_eval_mean_reward(10): {r_mean:>10.4f}')
                ll = f'{(ref_val*1000)//1000:0.0f}' + '_' +str(done_cnt)
                writer.add_scalar('eval_spec/best_eval_rewards', r_mean, ep_cnt)
                agent.save(save_path, ll)
                agent.save(save_path, 'best')


def evaluate(args, agent:TD3, env:gym.Env, is_test=False):
    observation = env.reset()
    reward_episode = 0
    done = False
    cnt = 0
    while True:
        policy = agent.select_action(observation)
        action = policy
        observation, reward, done, _ = env.step(action)
        reward_episode += reward
        cnt += 1
        
        if cnt > (args.STEP_MAX):
            done = True
        if done: break
        if is_test: time.sleep(0.05)

    try: env.step_counter = 0
    except: pass
    env.reset()
    return reward_episode

def test(args, agent:TD3, env:gym.Env):

    save_path = args.log_dir
    agent.load(save_path, 'best')

    rr_stack = []
    for i in range(100):
        rr = evaluate(args, agent, env, True)
        prLightPurple('##episode_eval_reward:{}'.format(rr))
        rr_stack.append(rr)
    prLightPurple('##episode_eval_mean_reward:{}'.format(np.mean(rr_stack)))


if __name__ == '__main__':
    torch.manual_seed(221109)
    torch.cuda.manual_seed(221109)
    args = parsing()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cpu' if args.gpu == '-1' else 'cuda'
    args.device = device

    name_component = []
    name_component += [args.name]
    name_component += [args.model]

    # env = gym.make(args.name)
    env = BaselineEnv()
    ############################## make Env first
        
    nb_action = env.action_space.shape[0]
    
    print(env.__class__, device)
    print(type(env.observation_space))
    
    nb_states = env.observation_space.shape[0]
    agent = TD3(lr=1e-3, 
                state_dim=nb_states,
                action_dim=nb_action,
                device=device)
        

    print(nb_states, nb_action)

    agent.max_action = env.action_space.high[0]

    args.nb_states = nb_states
    args.nb_action = nb_action
        
    args.NAME = '-'.join(name_component)

    if args.task == 'train':
        cc = len(os.listdir('local/'))
        log_dir = f'local/{cc}-{args.NAME}'
        while os.path.exists(log_dir):
            cc += 1
            log_dir = f'local/{cc}-{args.NAME}'
        print(f' tensorboard log saved at :', log_dir)
        writer = SummaryWriter(log_dir)
        args.log_dir = log_dir
        train(args, agent, env, writer=writer)

    elif args.task == 'test':
        test(args, agent, env)