from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, len(samples) + 1, 10)
    estimated_mean = []
    for sample_size in sample_sizes:
        ug = UnivariateGaussian()
        ug.fit(samples[:sample_size])
        estimated_mean.append(np.abs(ug.mu_ - 10))

    go.Figure(
        [go.Scatter(x=sample_sizes, y=estimated_mean, mode='markers+lines',
                    name=r'$\widehat\mu$')],
        layout=go.Layout(
            title="Estimation of Expectation As Function Of Number Of Samples",
            xaxis_title="$m\\text{ - number of samples}$",
            yaxis_title="r$\hat\mu$",
            height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(
        [go.Scatter(x=samples, y=univariate_gaussian.pdf(samples),
                    mode='markers', name=r'$\widehat\mu$')],
        layout=go.Layout(
            title="PDF values of samples",
            xaxis_title="sample",
            yaxis_title="PDF",
            height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    cov_matrix = np.array([
        [1, 0.2, 0, 0.5],
        [0.2, 2, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(np.array([0, 0, 4, 0]), cov_matrix,
                                            1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihood_lst = []
    max_log_likelihood = MultivariateGaussian.log_likelihood(np.array([0, 0,
                                                                       0, 0]),
                                                             cov_matrix,
                                                             samples)
    max_x = 0
    max_y = 0
    for x in f1:
        for y in f3:
            mu = np.array([x, 0, y, 0])
            current_log_likelihood = MultivariateGaussian.log_likelihood(mu,
                                     cov_matrix, samples)
            log_likelihood_lst.append(current_log_likelihood)
            if current_log_likelihood > max_log_likelihood:
                max_log_likelihood = current_log_likelihood
                max_x = x
                max_y = y

    go.Figure(go.Heatmap(x=f1, y=f3,
                         z=np.array(log_likelihood_lst).reshape(
                             len(f1), len(f3)))).update_layout(
        title="Log Likelihood as Function of Expectation").update_xaxes(
        title="f1").update_yaxes(title="f3").show()

    # Question 6 - Maximum likelihood
    print(f"The model that achieved the maximum log-likelihood value"
          f" is: f1={round(max_x, 3)}, f3={round(max_y, 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
