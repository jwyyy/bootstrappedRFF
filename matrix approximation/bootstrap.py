import numpy as np
import matplotlib.pyplot as plt

from genTestMatrix import *
from tqdm import tqdm
import timeit
import argparse

import timeit
from scipy.sparse.linalg import svds 



def rfapprox(X, R, D, gamma, metric):
    N    = X.shape[0]    
    W    = np.random.normal(loc=0, scale=1, size=(R, D)) * (10*gamma)**-1
    b    = np.random.uniform(0, 2*np.pi, size=R)
    B    = np.repeat(b[:, np.newaxis], N, axis=1)
    norm = 1./ np.sqrt(R)
    Z    = norm * np.sqrt(2) * np.cos(W @ X.T + B)
    
    if metric=='inf':
        ZZ   = Z.T@Z
        return ZZ, Z
    elif metric=='op':
        return Z, Z



def boot(e, Z, metric):
    
    l = Z.shape[0]
    idxBoot = np.random.choice(range(l), l, replace=True, p=None)    
    Zs = Z[idxBoot, :]
    
    if metric=='inf':
        ZZs = Zs.T @ Zs    
        w = np.linalg.norm(ZZs-e, ord=np.inf) 
    
    elif metric=='op':
        u, s, _ = svds(Zs, k=1)
        w = np.abs(e**2 - s**2)

    return w
        

    

def bootstrap(K, X, sample, runs, boots, alpha=0.1, metric = 'inf', gamma=1.0):

    print('Start Bootstrap with sample: ', sample)
    
    #======================
    # Stage 1
    #======================
    error = []
    error_boot = []
    print(sample)


    if metric=='op':
        u, s, _ = svds(K, k=1)

    for i in tqdm(range(runs)):
        
        Kapprox, Z = rfapprox(X, sample, D=X.shape[1], gamma=gamma, metric=metric)

        if metric=='inf':
            dist = np.linalg.norm(K-Kapprox, ord=np.inf)
        elif metric=='op':
            uapprox, sapprox, _ = svds(Z, k=1)
            dist = np.abs(sapprox**2 - s)
            
        error.append(dist)
        
        #======================
        # Stage 2
        #======================           
        if metric=='inf':
            out = [boot(Kapprox, Z, metric) for i in range(boots)]
        elif metric=='op':    
            out = [boot(sapprox, Z, metric) for i in range(boots)]  

        error_boot.append(out)
    
    return error, error_boot




def plot_results(indx):

    plt.figure(figsize=(9,7))
    plt.plot(samples[indx::], qunatile_e[indx::], 'k--', lw=4, label=r'$q$')    
    plt.plot(samples[indx::], quantiles_star_mean[indx::], lw=4,
             color='#3182bd', label=r'Mean of $\hat{q}_i$')

    samples_inter = np.arange(samples[indx], samples[-1], 1)
    scaling = np.sqrt(np.asarray(samples)[indx] / samples_inter)
    error_extrapol = scaling * quantiles_star_mean[indx]
    sd_extrapol = scaling * quantiles_star_sd[indx]
    upperlimits = error_extrapol + 1*sd_extrapol
    lowerlimits = np.maximum(0, error_extrapol - 1*sd_extrapol)
    
    plt.plot(samples_inter, error_extrapol, lw=4,
             color='#de2d26', label='Extrapolated')

    plt.fill_between(samples_inter, lowerlimits, upperlimits, 
                     facecolor='b', color='#de2d26', alpha=0.1)

    plt.plot(samples_inter, lowerlimits, lw=2, color='#de2d26', alpha=0.3)
    plt.plot(samples_inter, upperlimits, lw=2, color='#de2d26', alpha=0.3)

    plt.ylabel('error', fontsize=22)
    plt.xlabel('Random Features', fontsize=22)
    plt.legend(loc="best", fontsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.tick_params(axis='x', labelsize=22)     

    #plt.yscale('log')   
    #plt.xscale('symlog')     
    plt.tight_layout()
    plt.savefig('bootstrap_' + str(indx) + '_' + args.name +  '.pdf')
    #plt.close()    



if __name__ == "__main__":
    print('Start.........')
    t0 = timeit.default_timer()

    np.random.seed(1234)  
    
    parser = argparse.ArgumentParser(description='Bootstrap Simulation')
    #
    parser.add_argument('--name', type=str, default='example1_gamma1')
    #
    parser.add_argument('--samples', type=int, nargs='+', default=[50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 4000, 6000])
    #
    parser.add_argument('--runs', type=int, default=300)
    #
    parser.add_argument('--boots', type=int, default=30)
    #
    parser.add_argument('--problem', type=str, default='example2') 
    #
    parser.add_argument('--n', type=int, default=25000)     
    #
    parser.add_argument('--alpha', type=float, default=0.1)
    #
    parser.add_argument('--metric', type=str, default='op')
    #
    parser.add_argument('--gamma', type=float, default=4)    
    #    
    args = parser.parse_args()    
    
    #=========================================================================
    # Number of oversamples
    #=========================================================================
    runs = args.runs
    boots = args.boots
    samples = args.samples
    alpha = args.alpha
    metric = args.metric
    
    #=========================================================================
    # Generate random input matrix
    #=========================================================================
    if args.problem == 'example1':
        K, X = example1(gamma=args.gamma, n_samples=args.n)
    elif args.problem == 'example2':
        K, X = example2(gamma=args.gamma, n_samples=args.n)      
    elif args.problem == 'mnist':
        K, X = mnist(gamma=args.gamma, n_samples=args.n)        

    #=========================================================================
    #=========================================================================
    qunatile_e = []
    qunatile_e_mean = []

    quantiles_star_mean = []
    quantiles_star_sd = []
    quantiles_star_lower = []    
    quantiles_star_upper = []    
    fail_prob = []
    traps = []
    
    for sample in samples:
        e, e_star = bootstrap(K, X, sample, runs, boots, alpha = alpha, metric = metric, gamma=args.gamma)
        
        qunatile_e.append((np.percentile(e, (1-alpha)*100)))
        qunatile_e_mean.append(np.percentile(e, 50))

        quantiles_star = [(np.percentile(temp, (1-alpha)*100)) for temp in e_star ]
        quantiles_star_mean.append(np.mean(quantiles_star))
        quantiles_star_sd.append(np.std(quantiles_star))   
        quantiles_star_lower.append(np.percentile(quantiles_star, (alpha)*100))   
        quantiles_star_upper.append(np.percentile(quantiles_star, (1-alpha)*100))
        

        store = [qunatile_e, quantiles_star_mean, quantiles_star_sd, quantiles_star_lower, quantiles_star_upper, samples]
        np.save('results/bootstrap_' + args.name +  '.npy', store)


    store = [qunatile_e, quantiles_star_mean, quantiles_star_sd, quantiles_star_lower, quantiles_star_upper, samples]
    np.save('results/bootstrap_' + args.name +  '.npy', store)

    plot_results(0)

    
    print('Total time:', timeit.default_timer()  - t0 )
