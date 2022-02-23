import torch
from torch.autograd import Variable
import copy
import numpy as np
from explanations import GuidedBackpropReLUModel
import math

FORWARD_BZ = 5000


def forward_batch(model, input, batchsize):
    inputsize = input.shape[0]
    for count in range((inputsize - 1) // batchsize + 1):
        end = min(inputsize, (count + 1) * batchsize)
        if count == 0:
            tempinput = input[count * batchsize:end]
            out = model(tempinput.cuda())
            out = out.data.cpu().numpy()
        else:
            tempinput = input[count * batchsize:end]
            temp = model(tempinput.cuda()).data
            out = np.concatenate([out, temp.cpu().numpy()], axis=0)
    return out


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def sample_eps_Inf(image, epsilon, N):
    images = np.tile(image, (N, 1, 1, 1))
    dim = images.shape
    return np.random.uniform(-1 * epsilon, epsilon, size=dim)


def get_explanation_pdt(image, model, label, exp, sg_r=None, sg_N=None, given_expl=None, binary_I=False):
    image_v = Variable(image, requires_grad=True)
    model.zero_grad()
    out = model(image_v)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])

    if exp == 'Grad':
        pdt.backward()
        grad = image_v.grad
        expl = grad.data.cpu().numpy()
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'GBP':
        gb_model = GuidedBackpropReLUModel(model=copy.deepcopy(model), use_cuda=True)
        gb = gb_model(image_v, index=label)
        expl = gb
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'SHAP':
        expl = shap(image.cpu(), label, pdt, model, 20000)
    elif exp == 'Square':
        expl = optimal_square(image.cpu(), label, pdt, model, 20000)
    elif exp == 'NB':
        expl = optimal_nb(image.cpu(), label, pdt, model, 20000)
        if binary_I:
            expl = expl * image.cpu().numpy().flatten()
    elif exp == 'Int_Grad':
        for i in range(10):
            image_v = Variable(image * i/10, requires_grad=True)
            model.zero_grad()
            out = model(image_v)
            pdt = torch.sum(out[:, label])
            pdt.backward()
            grad = image_v.grad
            if i == 0:
                expl = grad.data.cpu().numpy() / 10
            else:
                expl += grad.data.cpu().numpy() / 10
        if binary_I:
            expl = expl * image.cpu().numpy()
    elif exp == 'Smooth_Grad':
        avg_points = 50
        for count in range(int(sg_N/avg_points)):
            sample = torch.FloatTensor(sample_eps_Inf(image.cpu().numpy(), sg_r, avg_points)).cuda()
            X_noisy = image.repeat(avg_points, 1, 1, 1) + sample
            expl_eps, _ = get_explanation_pdt(X_noisy, model, label, given_expl, binary_I=binary_I)
            if count == 0:
                expl = expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])
            else:
                expl = np.concatenate([expl,
                                       expl_eps.reshape(avg_points, int(expl_eps.shape[0]/avg_points),
                                                        expl_eps.shape[1], expl_eps.shape[2], expl_eps.shape[3])],
                                      axis=0)
        expl = np.mean(expl, 0)
    else:
        raise NotImplementedError('Explanation method not supported.')

    return expl, pdtr


def kernel_regression(Is, ks, ys):
    """
    *Inputs:
        I: sample of perturbation of interest, shape = (n_sample, n_feature)
        K: kernel weight
    *Return:
        expl: explanation minimizing the weighted least square
    """
    n_sample, n_feature = Is.shape
    IIk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), Is)
    Iyk = np.matmul(np.matmul(Is.transpose(), np.diag(ks)), ys)
    expl = np.matmul(np.linalg.pinv(IIk), Iyk)
    return expl


def set_zero_infid(array, size, point, pert):
    if pert == "Gaussian":
        ind = np.random.choice(size, point, replace=False)
        randd = np.random.normal(size=point) * 0.2 + array[ind]
        randd = np.minimum(array[ind], randd)
        randd = np.maximum(array[ind] - 1, randd)
        array[ind] -= randd
        return np.concatenate([array, ind, randd])
    elif pert == "SHAP":
        nz_ind = np.nonzero(array)[0]
        nz_ind = np.arange(array.shape[0])
        num_nz = len(nz_ind)
        bb = 0
        while bb == 0 or bb == num_nz:
            aa = np.random.rand(num_nz)
            bb = np.sum(aa < 0.5)
        sample_ind = np.where(aa < 0.5)
        array[nz_ind[sample_ind]] = 0
        ind = np.zeros(array.shape)
        ind[nz_ind[sample_ind]] = 1
        return np.concatenate([array, ind])


def sample_nb_Z(X, size, point):
    """
    *Inputs:
        X: flatten X vector of shape = (n_feature, )
    *Return:
        Z: perturbation of sample point
    """
    ind = np.arange(784)
    randd = np.random.normal(size=point) * 0.2 + X[ind]
    randd = np.minimum(X[ind], randd)
    randd = np.maximum(X[ind] - 1, randd)

    return randd


def sample_shap_Z(X):
    nz_ind = np.nonzero(X)[0]
    nz_ind = np.arange(X.shape[0])
    num_nz = len(nz_ind)
    bb = 0
    while bb == 0 or bb == num_nz:
        aa = np.random.rand(num_nz)
        bb = np.sum(aa > 0.5)
    sample_ind = np.where(aa > 0.5)
    Z = np.zeros(len(X))
    Z[nz_ind[sample_ind]] = 1

    return Z


def shap_kernel(Z, X):
    M = X.shape[0]
    z_ = np.count_nonzero(Z)
    return (M-1) * 1.0 / (z_ * (M - 1 - z_) * nCr(M - 1, z_))


def shap(X, label, pdt, model, n_sample):
    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)
    Xs_img = Xs.reshape(n_sample, 1, 28, 28)

    Zs = np.apply_along_axis(sample_shap_Z, 1, Xs)
    Zs_real = np.copy(Zs)
    Zs_real[Zs == 1] = Xs[Zs == 1]
    Zs_real_img = Zs_real.reshape(n_sample, 1, 28, 28)
    Zs_img = Variable(torch.tensor(Xs_img - Zs_real_img), requires_grad=False).float()
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]

    ys = pdt.data.cpu().numpy() - ys
    ks = np.apply_along_axis(shap_kernel, 1, Zs, X=X.reshape(-1))

    expl = kernel_regression(Zs, ks, ys)

    return expl


def optimal_nb(X, label, pdt, model, n_sample):
    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)
    Xs_img = Xs.reshape(n_sample, 1, 28, 28)

    Zs = np.apply_along_axis(sample_nb_Z, 1, Xs, 784, 784)
    Zs_img = Zs.reshape(n_sample, 1, 28, 28)
    Zs_img = Variable(torch.tensor(Xs_img - Zs_img), requires_grad=False).float().cuda()
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]
    ys = pdt.data.cpu().numpy() - ys

    ks = np.ones(n_sample)
    expl = kernel_regression(Zs, ks, ys)
    return expl


def optimal_square(X, label, pdt, model, n_sample):
    im_size = X.shape
    width = im_size[2]
    height = im_size[3]
    rads = np.arange(10) + 1
    n_sample = 0
    for rad in rads:
        n_sample += (width - rad + 1) * (height - rad + 1)

    X = X.numpy()
    Xs = np.repeat(X.reshape(1, -1), n_sample, axis=0)

    Zs_img, Zs = get_imageset(Xs, im_size[1:], rads=rads)
    Zs_img = Zs_img.reshape(n_sample, 1, 28, 28)
    ks = np.ones(n_sample)

    Zs_img = Variable(torch.tensor(Zs_img), requires_grad=False).float().cuda()
    out = forward_batch(model, Zs_img, FORWARD_BZ)
    ys = out[:, label]
    ys = pdt.data.cpu().numpy() - ys

    expl = kernel_regression(Zs, ks, ys)
    return expl


def get_exp(ind, exp):
    return (exp[ind.astype(int)])


def get_imageset(image_copy, im_size, rads=[2, 3, 4, 5, 6]):
    rangelist = np.arange(np.prod(im_size)).reshape(im_size)
    width = im_size[1]
    height = im_size[2]
    ind = np.zeros(image_copy.shape)
    count = 0
    for rad in rads:
        for i in range(width - rad + 1):
            for j in range(height - rad + 1):
                ind[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 1
                image_copy[count, rangelist[:, i:i+rad, j:j+rad].flatten()] = 0
                count += 1
    return image_copy, ind


def get_exp_infid(image, model, exp, label, pdt, binary_I, pert):
    point = 784
    total = (np.prod(exp.shape))
    num = 10000
    if pert == 'Square':
        im_size = image.shape
        width = im_size[2]
        height = im_size[3]
        rads = np.arange(10) + 1
        num = 0
        for rad in rads:
            num += (width - rad + 1) * (height - rad + 1)
    exp = np.squeeze(exp)
    exp_copy = np.reshape(np.copy(exp), -1)
    image_copy = np.tile(np.reshape(np.copy(image.cpu()), -1), [num, 1])

    if pert == 'Gaussian':
        image_copy_ind = np.apply_along_axis(set_zero_infid, 1, image_copy, total, point, pert)
    elif pert == 'Square':
        image_copy, ind = get_imageset(image_copy, im_size[1:], rads=rads)

    if pert == 'Gaussian' and not binary_I:
        image_copy = image_copy_ind[:, :total]
        ind = image_copy_ind[:, total:total+point]
        rand = image_copy_ind[:, total+point:total+2*point]
        exp_sum = np.sum(rand*np.apply_along_axis(get_exp, 1, ind, exp_copy), axis=1)
        ks = np.ones(num)
    elif pert == 'Square' and binary_I:
        exp_sum = np.sum(ind * np.expand_dims(exp_copy, 0), axis=1)
        ks = np.apply_along_axis(shap_kernel, 1, ind, X=image.reshape(-1))
        ks = np.ones(num)
    else:
        raise ValueError("Perturbation type and binary_I do not match.")

    image_copy = np.reshape(image_copy, (num, 1, 28, 28))
    image_v = Variable(torch.from_numpy(image_copy.astype(np.float32)).cuda(), requires_grad=False)
    out = forward_batch(model, image_v, FORWARD_BZ)
    pdt_rm = (out[:, label])
    pdt_diff = pdt - pdt_rm

    # performs optimal scaling for each explanation before calculating the infidelity score
    beta = np.mean(ks*pdt_diff*exp_sum) / np.mean(ks*exp_sum*exp_sum)
    exp_sum *= beta
    infid = np.mean(ks*np.square(pdt_diff-exp_sum)) / np.mean(ks)
    return infid

def get_exp_sens(X, model, expl,exp, yy, pdt, sg_r,sg_N,sen_r,sen_N,norm,binary_I,given_expl):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        expl_eps, _ = get_explanation_pdt(X_noisy, model, yy, exp, sg_r, sg_N,
                                     given_expl=given_expl, binary_I=binary_I)
        max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)/norm)
    return max_diff

def evaluate_infid_sen(loader, model, exp, pert, sen_r, sen_N, sg_r=None, sg_N=None, given_expl=None):
    if pert == 'Square':
        binary_I = True
    elif pert == 'Gaussian':
        binary_I = False
    else:
        raise NotImplementedError('Only support Square and Gaussian perturbation.')

    model.eval()
    infids = []
    max_sens = []

    for i, (X, y) in enumerate(loader):
        if i >= 5:  # i >= 50 for the experiments used in the paper
            break

        X, y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)

        expl, pdt = get_explanation_pdt(X, model, y[0], exp, sg_r, sg_N,
                                        given_expl=given_expl, binary_I=binary_I)
        pdt = pdt.data.cpu().numpy()
        norm = np.linalg.norm(expl)

        infid = get_exp_infid(X, model, expl, y[0], pdt, binary_I=binary_I, pert=pert)
        infids.append(infid)

        max_diff = -math.inf
        sens = get_exp_sens(X, model, expl,exp, y[0], pdt, sg_r, sg_N,sen_r,sen_N,norm,binary_I,given_expl)
        #for _ in range(sen_N):   
            #sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
            #X_noisy = X + sample
            #expl_eps, _ = get_explanation_pdt(X_noisy, model, y[0], exp, sg_r, sg_N,
            #                                  given_expl=given_expl, binary_I=binary_I)
            #max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)) / norm
        max_sens.append(sens)

    infid = np.mean(infids)
    max_sen = np.mean(max_sens)

    return infid, max_sen
