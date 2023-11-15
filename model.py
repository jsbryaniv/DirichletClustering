
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
        'n_classes': 2,            # The maximum number of classes considered
        # Variables
        'P': None,                  # The probability of the variables
        's': None,                  # The class assignments
        'x': None,                  # The class outcome probabilities
        'mu': None,                 # The feature means
        'sigma': None,              # The feature standard deviations
        'pi': None,                 # The class probabilities
        # Priors
        'x_prior_alpha': .1,        # The x prior alpha
        'x_prior_beta': .1,         # The x prior beta
        'mu_prior_mean': None,      # The mu prior mean
        'mu_prior_std': None,       # The mu prior standard deviation
        'sigma_prior_shape': 2,     # The sigma prior shape
        'sigma_prior_scale': None,  # The sigma prior scale
        'pi_prior_conc': None,    # The beta prior concentration
    }

    def __init__(self, n_classes=None):
        """Initialize the model."""

        # Set the number of classes
        if n_classes is not None:
            self.VARIABLES['n_classes'] = n_classes

    def initialize_variables(self, data):
        """Initialize the variables for the model.

        Args:
            data (np.ndarray):
                The data to initialize the variables with.
                The shape should be (n_data, n_features+1)
                    where the last column is binary outcomes.
        """

        # Initialize variables
        variables = SimpleNamespace(**self.VARIABLES.copy())

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        pi = variables.pi
        x_prior_alpha = variables.x_prior_alpha
        x_prior_beta = variables.x_prior_beta
        mu_prior_mean = variables.mu_prior_mean
        mu_prior_std = variables.mu_prior_std
        sigma_prior_shape = variables.sigma_prior_shape
        sigma_prior_scale = variables.sigma_prior_scale
        pi_prior_conc = variables.pi_prior_conc

        # Set up data constants
        n_data = data.shape[0]
        n_features = data.shape[1] - 1
        variables.n_data = n_data
        variables.n_features = n_features

        # Set up the probability
        P = -np.inf
        variables.P = P

        # Set up the class assignments
        s = np.random.choice(n_classes, n_data)
        variables.s = s

        # Set up the outcome probabilities
        x = np.random.beta(x_prior_alpha, x_prior_beta, n_classes)
        variables.x_prior_alpha = x_prior_alpha
        variables.x_prior_beta = x_prior_beta
        variables.x = x

        # Set up the feature means
        if mu_prior_mean is None:
            mu_prior_mean = np.ones((n_classes, n_features)) * np.mean(data[:, :-1], axis=0)
        if mu_prior_std is None:
            mu_prior_std = np.ones((n_classes, n_features)) * np.std(data[:, :-1], axis=0)
        mu = mu_prior_mean + mu_prior_std * np.random.randn(n_classes, n_features)
        variables.mu_prior_mean = mu_prior_mean
        variables.mu_prior_std = mu_prior_std
        variables.mu = mu

        # Set up the feature standard deviations
        sigma_prior_shape *= np.ones((n_classes, n_features))
        if sigma_prior_scale is None:
            sigma_prior_scale = 1 / np.var(data[:, :-1], axis=0) / sigma_prior_shape
        sigma = 1 / np.sqrt(np.random.gamma(sigma_prior_shape, sigma_prior_scale, size=(n_classes, n_features)))
        variables.sigma_prior_shape = sigma_prior_shape
        variables.sigma_prior_scale = sigma_prior_scale
        variables.sigma = sigma

        # Set up the Dirichlet process parameters
        if pi_prior_conc is None:
            pi_prior_conc = .05 * n_data * np.ones(n_classes)
        pi = pi_prior_conc / np.sum(pi_prior_conc)
        variables.pi_prior_conc = pi_prior_conc
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
        n_classes = variables.n_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        # Calculate the likelihood
        lhood = 0
        for i in range(n_data):

            # Feature likelihood
            for j in range(n_features):
                lhood += stats.norm.logpdf(data[i, j], mu[s[i], j], sigma[s[i], j])

            # Outcome likelihood
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
        n_classes = variables.n_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        pi = variables.pi
        x_prior_alpha = variables.x_prior_alpha
        x_prior_beta = variables.x_prior_beta
        mu_prior_mean = variables.mu_prior_mean
        mu_prior_std = variables.mu_prior_std
        sigma_prior_shape = variables.sigma_prior_shape
        sigma_prior_scale = variables.sigma_prior_scale
        pi_prior_conc = variables.pi_prior_conc

        # Calculate the prior
        P_prior = (
            np.sum(stats.dirichlet.logpdf(pi, pi_prior_conc))
            + np.sum(stats.beta.logpdf(x, x_prior_alpha, x_prior_beta))
            + np.sum(stats.norm.logpdf(mu, mu_prior_mean, mu_prior_std))
            + np.sum(stats.gamma.logpdf(1/sigma**2, sigma_prior_shape, scale=sigma_prior_scale))
        )

        # Calculate label probability
        P_labels = 0
        for i in range(n_data):
            P_labels += np.log(pi[s[i]])

        # Calculate likelihood
        P_lhood = self.likelihood(data, variables)

        # Return the probability
        P = P_prior + P_labels + P_lhood
        if ~np.isfinite(P):
            print("Warning: probability is not finite.")
        return P
    
    def update_s(self, data, variables):
        """Update the class assignments."""

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        pi = variables.pi

        # Loop over data
        for i in range(n_data):

            # Calculate the probability of each class
            prob = np.zeros(n_classes)
            for k in range(n_classes):
                prob[k] = np.log(pi[k])
                for j in range(n_features):
                    prob[k] += stats.norm.logpdf(data[i, j], mu[k, j], sigma[k, j])
                prob[k] += stats.bernoulli.logpmf(data[i, -1], x[k])
            
            # Softmax
            prob -= np.max(prob)
            prob = np.exp(prob)
            prob /= np.sum(prob)

            # Sample a new class
            s[i] = np.random.choice(n_classes, p=prob)
        
        # Update the variables
        variables.s = s

        # Return the variables
        return variables

    def update_x(self, data, variables):
        """Update the outcome probabilities."""

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        x = variables.x
        x_prior_alpha = variables.x_prior_alpha
        x_prior_beta = variables.x_prior_beta

        # Loop over classes
        for k in range(n_classes):

            # Calculate the number of successes and failures
            ids = np.where(s == k)[0]
            n_success = np.sum(data[ids, -1])
            n_fail = len(ids) - n_success

            # Update the outcome probability
            x[k] = np.random.beta(x_prior_alpha + n_success, x_prior_beta + n_fail)

            # Ensure that the outcome probability is not 0 or 1
            if x[k] == 0:
                x[k] = 1e-10
            elif x[k] == 1:
                x[k] = 1 - 1e-10

        # Update the variables
        variables.x = x

        # Return the variables
        return variables

    def update_mu(self, data, variables):
        """Update the feature means."""

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        mu = variables.mu
        sigma = variables.sigma
        mu_prior_mean = variables.mu_prior_mean
        mu_prior_std = variables.mu_prior_std

        # Loop over classes
        for k in range(n_classes):

            # Calculate the number of data points
            ids = np.where(s == k)[0]
            n_data_k = len(ids)
            
            # Loop over features
            for f in range(n_features):
                
                # Calculate the posterior mean and standard deviation
                mu_post_std = 1 / np.sqrt(
                    1 / mu_prior_std[k, f]**2 + n_data_k / sigma[k, f]**2
                )
                mu_post_mean = mu_post_std**2 * (
                    mu_prior_mean[k, f] / mu_prior_std[k, f]**2
                    + np.sum(data[ids, f]) / sigma[k, f]**2
                )

                # Update the feature mean
                mu[k, f] = np.random.normal(mu_post_mean, mu_post_std)

        # Update the variables
        variables.mu = mu

        # Return the variables
        return variables

    def update_sigma(self, data, variables):
        """Update the feature standard deviations."""

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        mu = variables.mu
        sigma = variables.sigma
        sigma_prior_shape = variables.sigma_prior_shape
        sigma_prior_scale = variables.sigma_prior_scale

        # Loop over classes
        for k in range(n_classes):

            # Calculate the number of data points
            ids = np.where(s == k)[0]
            n_data_k = len(ids)
            
            # Loop over features
            for f in range(n_features):
                    
                # Calculate the shape and scale
                shape = sigma_prior_shape[k, f] + n_data_k / 2
                scale = (
                    sigma_prior_scale[k, f]**-1 + np.sum((data[ids, f] - mu[k, f])**2) / 2
                ) ** -1
                
                # Update the feature standard deviation
                sigma[k, f] = 1 / np.sqrt(np.random.gamma(shape, scale))

        # Update the variables
        variables.sigma = sigma

        # Return the variables
        return variables

    def update_pi(self, data, variables):
        """Update the class probabilities."""

        # Extract variables
        n_data = variables.n_data
        n_features = variables.n_features
        n_classes = variables.n_classes
        s = variables.s
        pi = variables.pi
        pi_prior_conc = variables.pi_prior_conc

        # Calculate the number of data points in each class
        counts = np.zeros(n_classes)
        for k in range(n_classes):
            counts[k] = np.sum(s == k)

        # Update the class probabilities
        pi = np.random.dirichlet(counts + pi_prior_conc + 1e-100)

        # Update the variables
        variables.pi = pi

        # Return the variables
        return variables

    def train(self, data, n_iterations=100):
        """Perform Gibbs sampling for parameter inference."""

        # Initialize variables from data
        print("Initializing variables.")
        variables = self.initialize_variables(data)

        # Initialize MAP and samples
        map_variables = copy.deepcopy(variables)
        samples = {key: [] for key in ['P', 's', 'x', 'mu', 'sigma', 'pi']}

        # Run the Gibbs sampler
        print("Running Gibbs sampler.")
        for i in range(n_iterations):
            t = time.time()

            # Update parameters
            variables = self.update_s(data, variables)
            variables = self.update_x(data, variables)
            variables = self.update_mu(data, variables)
            variables = self.update_sigma(data, variables)
            variables = self.update_pi(data, variables)
            P = self.probability(data, variables)

            # Check for MAP
            if P >= map_variables.P:
                map_variables = copy.deepcopy(variables)
                map_variables.P = P

            # Save samples
            for key in samples.keys():
                samples[key].append(copy.deepcopy(getattr(variables, key)))

            # Print output
            print(f"Iteration {i + 1}/{n_iterations} ({time.time() - t:.2f}s): P = {P:.2f}")

        # Reformat samples
        for key in samples.keys():
            samples[key] = np.array(samples[key])

        # Return the MAP
        print("Done.")
        return map_variables, samples

    def predict(self, data, variables, **kwargs):
        """Predict the class of a data point."""

        # Format input
        if len(data.shape) == 1:
            data = np.array([data])
        if len(kwargs) > 0:
            variables = copy.deepcopy(variables)
            for key, value in kwargs.items():
                setattr(variables, key, value)

        # Extract variables
        n_data = data.shape[0]
        n_features = variables.n_features
        n_classes = variables.n_classes
        x = variables.x
        mu = variables.mu
        sigma = variables.sigma
        pi = variables.pi

        # Initialize s
        s = np.zeros(data.shape[0], dtype=int)

        # Loop over data
        for i in range(n_data):

            # Calculate the probability of each class
            prob = np.log(pi)
            if ~np.isnan(data[i, -1]):
                prob += stats.bernoulli.logpmf(data[i, -1], x)
            for f in range(n_features):
                if ~np.isnan(data[i, f]):
                    prob += stats.norm.logpdf(data[i, f], mu[:, f], sigma[:, f])
            
            # Softmax
            prob -= np.max(prob)
            prob = np.exp(prob)
            prob /= np.sum(prob)
            
            # Choose the class with the highest probability
            s[i] = np.argmax(prob)
        
        # Return the classes
        return s
