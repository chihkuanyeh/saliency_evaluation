import numpy as np
import torch
import loader
from infid_sen_utils import evaluate_infid_sen
import config


class Args:
    def __init__(self, args):
        self.source = args
        for key, val in args.items():
            setattr(self, key, val)

    def __repr__(self):
        return repr(self.source)


class OutputLog(object):
    '''Create output log'''
    def __init__(self):
        self.infid_dict = dict()
        self.max_sen_dict = dict()

    def write(self, method, infid, max_sen):
        self.infid_dict[method] = infid
        self.max_sen_dict[method] = max_sen

    def __str__(self):
        log = 'Infidelity:'
        log += '{}\n'.format(self.infid_dict)
        log += 'Max-Sensitivity:'
        log += '{}\n'.format(self.max_sen_dict)
        return log


if __name__ == "__main__":
    args = Args(config.args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = loader.mnist_loaders(batch_size=1)
    model = loader.mnist_load_model(args.model, state_dict=True, tf=True)
    model = model.cuda()

    output_log = OutputLog()

    for pert in args.perts:
        for exp in args.exps:
            print('Perturbation =', pert, '/ Explanation =', exp)
            infid, max_sen = evaluate_infid_sen(test_loader, model, exp, pert,
                                                args.sen_r, args.sen_N)
            output_log.write(exp, infid, max_sen)
            print(output_log)

        for sg in args.sgs:
            print('Perturbation =', pert, '/ Smooth-Grad =', sg)
            infid, max_sen = evaluate_infid_sen(test_loader, model, 'Smooth_Grad', pert,
                                                args.sen_r, args.sen_N, args.sg_r, args.sg_N,
                                                given_expl=sg)
            output_log.write(sg+'-SG', infid, max_sen)
            print(output_log)

