import torch.distributions as dist


def calc_kld(qz):
    kld = -0.5*(1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
    return(kld)


def calc_poisson_loss(ld,obs):
    p_z = dist.Poisson(ld)
    l = -p_z.log_prob(obs)
    return(l)


def calc_gaussian_loss(ld,obs): 
    p_z = dist.Normal(ld,1)
    l = -p_z.log_prob(obs)
    return(l)


#loss function for tVAE
def t_elbo_loss(qz,xld,x):
    t_elbo_loss = 0
    t_elbo_loss += calc_kld(qz).sum()
    t_elbo_loss += calc_poisson_loss(xld, x).sum()
    return(t_elbo_loss)


#loss function for eVAE
def e_elbo_loss(qz,ld_img,xcell_id,maintrain=False):
    e_elbo_loss = 0
    e_elbo_loss += calc_kld(qz).sum()
    e_elbo_loss += calc_gaussian_loss(ld_img, xcell_id).sum()
    return(e_elbo_loss)
