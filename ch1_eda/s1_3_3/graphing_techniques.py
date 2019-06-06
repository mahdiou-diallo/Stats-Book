import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def autocorrelation_plot(x: np.ndarray) -> np.ndarray:
    """
    Computes and graphs the auto correlation of the series `x`
    That is, it checks if there is an auto correlation between `x` 
    and its time lagged version for different values of the time lag

    This can help check for the randomness of the data.
    It can also help decide if we can model the data using a linear regression
    or a Box-Jenkins autoregressive model

    reference: [NIST Engineering Stats Handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm)
    """
    N = x.shape[0]
    x_bar = x.mean()
    C_0 = ((x-x_bar)**2).sum() / N # variance
    
    def autocovariance(x, h):
        x_head = x[:N-h]
        x_tail = x[h:]

        C_h = ((x_head-x_bar) * (x_tail-x_bar)).sum() / (N-h) # could either divide by `N` or `(N-h)`. the latter has less bias

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

    ax.plot(corr, linestyle='-', marker='o')

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'AUTOCORRELATION PLOT (N = {N})')

    plt.show()

    return corr


def bihistogram_plot(x1: np.ndarray, x2: np.ndarray) -> None:
    """
    Plots a bihistogram of the data
    The data can be from a single data set where we want to check the significance of a factor
    It gives graphical answers to the following questions:
        - Is a 2-level factor significant?
        - Does a (2-level) factor have an effect?
        - Does the location change between the 2 subgroups?
        - Does the variation change between the 2 subgroups?
        - Does the distributional shape change between subgroups?
        - Are there any outliers?
    It is a graphical equivalent of the following quantitative tests:
        - 2 sample t-test (shift in location)
        - Fisher test (shift in variation)
        - Kolgomorov-Smirnov test (shift in distribution)
        also
        - quantile-quantile plot (shift in location and distribution)

        reference: [NIST Engineering Stats Handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/eda332.htm)
    """
    _, bins, _ = plt.hist(x1, bins=50, density=True, label='x1')
    _ = plt.hist(x2, bins=bins, alpha=0.5, density=True, label='x2')
    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    # X = np.random.standard_normal(50)
    # X = np.arange(50)
    X = np.sin(np.linspace(0, 4*np.pi, 50))


    autocorrelation_plot(X)