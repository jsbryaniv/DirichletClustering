
# Import libraries
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace


# Define the model class
class DirichletClustering:
    """A model for predicting features based on a Dirichlet process clustering."""

    # Define model variables
    VARIABLES = {
        # Constants
        'n_data': None,             # The number of data entries
        'n_features': None,         # The number of features
        'max_classes': 10,          # The maximum number of classes considered
        # Variables
        'P': None,                  # The probability of the variables
        's': None,                  # The class assignments
        'x': None,                  # The class outcome probabilities
        'mu': None,                 # The feature means
        'sigma': None,              # The feature standard deviations
        'beta': None,               # The Hierarchical Dirichlet Process parameter
        'pi': None,                 # The class probabilities
        # Priors
        'x_prior_alpha': None,      # The p prior alpha
        'x_prior_beta': None,       # The p prior beta
        'mu_prior_mean': None,      # The mu prior mean
        'mu_prior_std': None,       # The mu prior standard deviation
        'sigma_prior_shape': None,  # The sigma prior shape
        'sigma_prior_scale': None,  # The sigma prior scale
        'beta_prior_conc': None,    # The beta prior concentration
    }

    def initialize_variables(self, data):
        """Initialize the variables for the model.

        Args:
            data (np.ndarray):
                The data to initialize the variables with.
                The shape should be (n_data, n_features+1)
                    where the last column is binary outcomes.
        """

        # Initialize variables
        variables = SimpleNamespace(self.VARIABLES.copy())

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        max_classes = variables.max_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        beta = variables.beta
        pi = variables.pi
        x_prior_alpha = variables.x_prior_alpha
        x_prior_beta = variables.x_prior_beta
        mu_prior_mean = variables.mu_prior_mean
        mu_prior_std = variables.mu_prior_std
        sigma_prior_shape = variables.sigma_prior_shape
        sigma_prior_scale = variables.sigma_prior_scale
        beta_prior_conc = variables.beta_prior_conc

        # Set up data constants
        n_data = data.shape[0]
        n_features = data.shape[1] - 1
        variables.n_data = n_data
        variables.n_features = n_features

        # Set up the probability
        P = -np.inf
        variables.P = P

        # Set up the class assignments
        s = np.random.choice(max_classes, n_data)
        variables.s = s

        # Set up the outcome probabilities
        x_mean = np.mean(data[:, -1])
        x_var = np.var(data[:, -1])
        if x_prior_alpha is None:
            x_prior_alpha = x_mean * (1-x_mean) / x_var - 1
        if x_prior_beta is None:
            x_prior_beta = x_prior_alpha * (1-x_mean) / x_mean
        x = np.random.beta(x_prior_alpha, x_prior_beta, max_classes)
        variables.x_prior_alpha = x_prior_alpha
        variables.x_prior_beta = x_prior_beta
        variables.x = x

        # Set up the feature means
        if mu_prior_mean is None:
            mu_prior_mean = np.ones(n_features) * np.mean(data[:, :-1])
        if mu_prior_std is None:
            mu_prior_std = np.ones(n_features) * np.std(data[:, :-1])
        mu = mu_prior_mean + mu_prior_std * np.random.randn(n_features)
        variables.mu_prior_mean = mu_prior_mean
        variables.mu_prior_std = mu_prior_std
        variables.mu = mu

        # Set up the feature standard deviations
        if sigma_prior_shape is None:
            sigma_prior_shape = 2 * np.ones(n_features)
        if sigma_prior_scale is None:
            sigma_prior_scale = 1 / np.var(data[:, :-1]) / sigma_prior_shape
        sigma = 1 / np.sqrt(np.random.gamma(sigma_prior_shape, sigma_prior_scale, n_features))
        variables.sigma_prior_shape = sigma_prior_shape
        variables.sigma_prior_scale = sigma_prior_scale
        variables.sigma = sigma

        # Set up the Dirichlet process parameters
        if beta_prior_conc is None:
            beta_prior_conc = np.ones(max_classes) / max_classes
        beta = beta_prior_conc.copy()
        pi = beta.copy()
        variables.beta_prior_conc = beta_prior_conc
        variables.beta = beta
        variables.pi = pi

        # Return variables
        return variables
    
    def likelihood(self, data, variables, **kwargs):

        # Merge variables and kwargs
        if len(kwargs) > 0:
            variables = copy.deepcopy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        max_classes = variables.max_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma

        # Calculate the likelihood
        lhood = 0
        for i in range(n_data):

            # Feature likelihood
            for j in range(n_features):
                if ~np.isnan(data[i, j]):
                    lhood += stats.norm.logpdf(data[i, j], mu[s[i], j], sigma[s[i], j])

            # Outcome likelihood
            if ~np.isnan(data[i, -1]):
                lhood += stats.bernoulli.logpmf(data[i, -1], x[s[i]])

        # Return the likelihood
        return lhood 
    
    def probability(self, data, variables, **kwargs):

        # Merge variables and kwargs
        if len(kwargs) > 0:
            variables = copy.deepcopy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        max_classes = variables.max_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        beta = variables.beta
        pi = variables.pi
        x_prior_alpha = variables.x_prior_alpha
        x_prior_beta = variables.x_prior_beta
        mu_prior_mean = variables.mu_prior_mean
        mu_prior_std = variables.mu_prior_std
        sigma_prior_shape = variables.sigma_prior_shape
        sigma_prior_scale = variables.sigma_prior_scale
        beta_prior_conc = variables.beta_prior_conc

        # Calculate the prior
        P_prior = (
            stats.dirichlet.logpdf(beta, beta_prior_conc)
            + np.sum(stats.beta.logpdf(x, x_prior_alpha, x_prior_beta))
            + np.sum(stats.norm.logpdf(mu, mu_prior_mean, mu_prior_std))
            + np.sum(stats.gamma.logpdf(1/sigma**2, sigma_prior_shape, scale=sigma_prior_scale))
        )

        # Calculate Dirichlet process probability
        P_dirichlet = stats.dirichlet.logpdf(pi+1e-100, beta+1e-100)

        # Calculate label probability
        P_labels = 0
        for i in range(n_data):
            P_labels += np.log(pi[s[i]])

        # Calculate likelihood
        P_lhood = self.likelihood(data, variables)

        # Return the probability
        P = P_prior + P_dirichlet + P_labels + P_lhood
        return P
    
    def update_s(self, data, variables):
        pass

    def update_x(self, data, variables):
        pass

    def update_mu(self, data, variables):
        pass

    def update_sigma(self, data, variables):
        pass

    def update_beta(self, data, variables):
        pass

    def update_pi(self, data, variables):
        pass

    def gibbs_sampler(self, data, n_iterations=1000):
        """Perform Gibbs sampling for parameter inference."""

        # Initialize variables from data
        print("Initializing variables.")
        variables = self.initialize_variables(data)

        # Initialize MAP and samples
        map_variables = copy.deepcopy(variables)
        samples = {key: [] for key in ['mu', 'sigma', 'x', 'beta', 'pi', 'P']}

        # Run the Gibbs sampler
        print("Running Gibbs sampler.")
        for i in range(n_iterations):
            t = time.time()

            # Update parameters
            variables = self.update_s(data, variables)
            variables = self.update_x(data, variables)
            variables = self.update_mu(data, variables)
            variables = self.update_sigma(data, variables)
            variables = self.update_beta(data, variables)
            P = self.probability(data, variables)

            # Store the sample
            samples['P'] = P
            for key in samples.keys():
                if key != 'P':
                    samples[key].append(variables[key])

            # Check for MAP
            if P >= map_variables['P']:
                map_variables = copy.deepcopy(variables)
                map_variables['P'] = P

            # Print output
            print(f"Iteration {i + 1}/{n_iterations} ({time.time() - t:.2f}s): P = {P:.2f}")

        # Return the MAP and samples
        print("Done.")
        return map_variables, samples