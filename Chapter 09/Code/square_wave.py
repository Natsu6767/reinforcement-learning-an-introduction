import numpy as np
import matplotlib.pyplot as plt
import tqdm


class Interval:
    def __init__(self, left, right):
        self.left = left
        self.right = right

        self.size = right - left

    def contains(self, x):
        return self.left <= x <= self.right

DOMAIN = Interval(0, 10)

class ValueFunction:
    def __init__(self, num_feats, feat_width, alpha, domain=DOMAIN):
        self.num_feats = num_feats
        self.feat_width = feat_width
        self.alpha = alpha

        self.weights = np.zeros(num_feats, dtype=np.float)

        step = (domain.size - feat_width) / (num_feats - 1)
        left = domain.left

        self.features = list()

        for i in range(num_feats):
            self.features.append(Interval(left, left + feat_width))
            left += step
        
    def get_active(self, x):
        activations = list()

        for i in range(self.num_feats):
            if self.features[i].contains(x):
                activations.append(i)

        return np.asarray(activations)

    def get_value(self, x):
        active_features = self.get_active(x)
        return np.sum(self.weights[active_features])

    def update(self, x, y):
        active_features = self.get_active(x)
        delta = (self.get_value(x) - y)
        delta *= self.alpha / len(active_features)

        self.weights[active_features] -= delta

def generate_square_wave(x):
    if 2.5 < x < 7.5:
        return 1
    return 0

def sample(n):
    samples = list()

    for i in range(n):
        x = np.random.uniform(DOMAIN.left, DOMAIN.right)
        y = generate_square_wave(x)
        
        samples.append((x, y))
    return samples

if __name__ == "__main__":
    num_samples = [10, 40, 160, 640, 2560, 10240]
    feat_widths = [1, 2, 5]
    num_feats = 50
    alpha = 0.2

    plt.figure(figsize=(25, 20))
    
    x_values = np.arange(DOMAIN.left, DOMAIN.right, 0.01)
    true_values = [generate_square_wave(x) for x in x_values]

    num_plots = len(num_samples) * len(feat_widths)

    index = 0
    pbar = tqdm.tqdm(num_samples)
    for n_samples in pbar:
        training_data = sample(n_samples)
        for feat_width in feat_widths:
            pbar.set_description("#Samples: {}, Feat Width: {}".format(n_samples, feat_width))
            index += 1
            plt.subplot(len(num_samples), len(feat_widths), index)

            value_function = ValueFunction(num_feats, feat_width, alpha)
            for x, y in training_data:
                value_function.update(x, y)
            
            values = [value_function.get_value(x) for x in x_values]
            plt.plot(x_values, values)
            if index in [1, 2, 3, num_plots-2, num_plots-1, num_plots]:
                plt.plot(x_values, true_values, ls="--", label="Desired Function")
                plt.legend(loc="best")
            plt.title("#Samples: {}, Feature Width: {}".format(n_samples, feat_width), fontsize=15)

    plt.suptitle("Effect of Feature Width on Initial Generalization (first row)\n "+\
                 "and asymptotic accuracy (last row)", fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("square_wave.png")
    plt.close()
