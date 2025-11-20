import torch
from copy import deepcopy


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


def augment(input, N_aug=8):
    if isinstance(input, dict):
        if N_aug == 8:
            _input = deepcopy(input)
            input = {k: x.repeat(N_aug, *(1,)*(x.dim()-1)) for k, x in input.items() if isinstance(x, torch.Tensor)}
            if _input['loc'].size() == _input['depot'].size():
                _input['loc'], _input['depot'] = augment_xy_data_by_8_fold(_input['loc']), augment_xy_data_by_8_fold(_input['depot'])
            else:
                _input['loc'], _input['depot'] = augment_xy_data_by_8_fold(_input['loc']), augment_xy_data_by_8_fold(_input['depot'].view(-1, 1, 2))
            input["loc"] = _input['loc']
            input["depot"] = _input['depot']
        else:
            if input['loc'].size() == input['depot'].size():
                input['loc'], input['depot'] = input['loc'], input['depot']
            else:
                input['loc'], input['depot'] = input['loc'], input['depot'].view(-1, 1, 2)
    else:
        if N_aug == 8:
            input = augment_xy_data_by_8_fold(input)
        else:
            raise NotImplementedError
    return input
