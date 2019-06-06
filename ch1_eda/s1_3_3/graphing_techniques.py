import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def autocorrelation_plot(x: np.ndarray) -> np.ndarray:
    """
    Computes and graphs the auto correlation of the series `x`
    That is, it checks if there is an auto correlation between `x` 
    and its time lagged version for different values of the time lag

    This can help check for the randomness of the data

    reference: [NIST Engineering Stats Handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm)
    """
    N = x.shape[0]
    x_bar = x.mean()
    C_0 = ((x-x_bar)**2).sum() / N # variance
    
    def autocovariance(x, h):
        x_head = x[:N-h]
        x_tail = x[h:]

        C_h = ((x_head-x_bar) * (x_tail-x_bar)).sum() / N # could also be divided by `(N-h)`

        return C_h

    corr = np.array([autocovariance(x, h) for h in range(N)]) / C_0

    # thresholds for confidence limits 95% and 99%
    # correlation values that are outside `[-th_95, th_95]` are significant at 95% confidence level
    th_95 = scipy.stats.norm.ppf(.975) / N**.5
    th_99 = scipy.stats.norm.ppf(.995) / N**.5
    


    _, ax = plt.subplots()
    
    ax.axvline(x=0, color='k', linestyle='-')
    ax.axhline(y=0, color='k', linestyle='-')

    ax.axhline(y=th_95, color='r', linestyle=':')
    ax.axhline(y=-th_95, color='r', linestyle=':')
    ax.axhline(y=th_99, color='r', linestyle=':')
    ax.axhline(y=-th_99, color='r', linestyle=':')

    ax.plot(corr, linestyle='-')

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'AUTOCORRELATION PLOT (N = {N})')

    plt.show()

    return corr



if __name__ == '__main__':
    X = np.random.standard_normal(50)#np.arange(50)
    # print(X)
    autocorrelation_plot(X)