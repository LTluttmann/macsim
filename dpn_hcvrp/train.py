import os
import time
import torch
import math
import torch

from tqdm import tqdm
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils import move_to
from utils.log_utils import log_values
from utils.problem_augment import augment
from nets.attention_model import set_decode_type


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost_file = open(os.path.join(opts.save_dir, 'gap.txt'), mode='a+')
    # for i in [1, 2, 3, 5, 7, 10, 20]:
    for i in opts.agent_list:
        print('Validating...,with' + str(i) + 'agents\n')
        model.agent_num = i
        model.decay = 0
        model.depot_num = opts.depot_eval
        model.embedder.agent_num = i
        cost = rollout(model, dataset, i, opts)
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}\n'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))
        cost_file.write(str(avg_cost.item()) + ' ')
    cost_file.write('\n')

    return avg_cost


def validate_greedy(model, dataset, opts):
    # Validate
    print('Validating greedily...')
    cost_file = open(os.path.join(opts.save_dir, 'gap2.txt'), mode='a+')
    for i in opts.agent_list:
        print('Validating greedily with' + str(i) + 'agents\n')
        model.agent_num = i
        model.embedder.agent_num = i
        model.decay = 0
        model.depot_num = 1
        cost = rollout(model, dataset, i, opts, aug=8, pomo=1, batch_size=1000)
        # print(cost.shape)
        avg_cost = cost.mean()
        print('Greedy validation overall avg_cost: {} +- {}\n'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))
        cost_file.write(str(avg_cost.item()) + ' ')
    cost_file.write('\n')

    return avg_cost


def rollout(model, dataset, i, opts, aug=None, pomo=None, batch_size=None):
    aug = aug or opts.aug_eval
    pomo = pomo or opts.r_eval
    batch_size = batch_size or opts.eval_batch_size
    # Put in greedy evaluation mode!
    if max(1, aug) * max(1, pomo) == 1:
        set_decode_type(model, "greedy")
    else:
        set_decode_type(model, "sampling")
    model.eval()

    def eval_model_bat(bat, batch_size, agt, aug=8):
        with torch.no_grad():
            agent_per = torch.arange(agt).cuda()[None, :].expand(pomo, -1)
            if (pomo > 1):
                for i in range(100):
                    a = torch.randint(0, agt, (pomo,)).cuda()
                    b = torch.randint(0, agt, (pomo,)).cuda()
                    p = agent_per[torch.arange(pomo), a].clone()
                    q = agent_per[torch.arange(pomo), b].clone()
                    agent_per = agent_per.scatter(dim=1, index=b[:, None], src=p[:, None])
                    agent_per = agent_per.scatter(dim=1, index=a[:, None], src=q[:, None])
                agent_per[0] = torch.arange(agt).cuda()
            model.agent_per = agent_per
            cost, _, route = model(move_to(bat, opts.device), return_pi=True)
            cost, _ = cost.min(-1)
            # route = route.view(aug * batch_size, pomo, -1).gather(1, _[:, None, None].expand(-1, -1, route.size(-1)))
            cost, _ = cost.view(aug, -1).min(0, keepdim=False)
            # code related to printing a solution out

        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(augment(bat, aug), batch_size=batch_size, agt=i, aug=aug)
        for bat
        in tqdm(DataLoader(dataset, batch_size=batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, lr_scheduler, epoch, val_dataset, problem, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    graph_size = opts.graph_size

    training_dataset = problem.make_dataset(
        size=graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    # model = get_inner_model(model)
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        agent_num = 3
        depot_num = 1
        model.agent_num = agent_num
        model.embedder.agent_num = agent_num
        if epoch < 0:
            model.decay = 0.2
        else:
            model.decay = 0

        cost = train_batch(
            model,
            optimizer,
            agent_num,
            batch,
            opts
        )

        if batch_id % 10 == 0:
            print('current cost:' + str(cost.item()) + ' ' + str(depot_num) + ' ' + str(agent_num))
            cost_file = open(os.path.join(opts.save_dir, 'curve.txt'), mode='a+')
            cost_file.write(str(cost.item()) + ' ' + str(depot_num) + ' ' + str(agent_num) + '\n')


    if epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    epoch_duration = time.time() - start_time

    step += 1

    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    lr_scheduler.step()

    sampled_cost = validate(model, val_dataset, opts)
    greedy_cost = validate_greedy(model, val_dataset, opts)
    return sampled_cost, greedy_cost

def train_batch(
        model,
        optimizer,
        agent_num,
        batch,
        opts
):
    x = move_to(batch, opts.device)
    agent_per = torch.arange(agent_num).cuda()[None, :].expand(opts.pomo_size, -1)
    if (opts.pomo_size > 1):
        for i in range(100):
            a = torch.randint(0, agent_num, (opts.pomo_size,)).cuda()
            b = torch.randint(0, agent_num, (opts.pomo_size,)).cuda()
            p = agent_per[torch.arange(opts.pomo_size), a].clone()
            q = agent_per[torch.arange(opts.pomo_size), b].clone()
            agent_per = agent_per.scatter(dim=1, index=b[:, None], src=p[:, None])
            agent_per = agent_per.scatter(dim=1, index=a[:, None], src=q[:, None])
    # Evaluate model, get costs and log probabilities
    x_aug = augment(x, opts.N_aug)
    model.agent_per = agent_per
    makespan, partspan, cost_route, log_likelihood = model(x_aug)
    log_likelihood = log_likelihood.view(opts.N_aug, -1, opts.pomo_size, log_likelihood.size(-1)).permute(1, 0, 2, 3)
    makespan = makespan.view(opts.N_aug, -1, opts.pomo_size).permute(1, 0, 2)
    ll = log_likelihood.sum(-1)
    advantage_makespan = (makespan - makespan.mean(dim=1).mean(dim=-1)[:, None, None])
    loss = ((advantage_makespan) * ll).mean()
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cost_mean = makespan.mean().view(-1, 1)

    return cost_mean
