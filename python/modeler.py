import numpy as np
from scipy.spatial import KDTree

# When imported as a script-local module, allow absolute imports
try:
    from render import render_image
    from utils import image_to_cloud
except ImportError:  # pragma: no cover - fallback for package imports
    from .render import render_image
    from .utils import image_to_cloud


class Modeler:
    """Simplified PMF modeler for Python testing."""

    def __init__(self, data_img, lam=2.0, iteration_tol=50):
        self.data_cloud = image_to_cloud(data_img)
        self.tree = KDTree(self.data_cloud)
        self.lambda_ = lam
        self.iteration_tol = iteration_tol

    def compute_score(self, theta, model):
        rendered = render_image(model)
        cloud = image_to_cloud(rendered)
        dists, _ = self.tree.query(cloud)
        emd = np.sum(dists) / len(dists)
        return len(cloud) / (emd ** self.lambda_)

    def cuckoo_search(self, model):
        # placeholder search that evaluates the initial parameters only
        theta = np.zeros(1)
        best_score = self.compute_score(theta, model)
        return best_score
