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

    def __init__(self, data_img, lam=2.0, iteration_tol=50, search_range=5):
        self.data_cloud = image_to_cloud(data_img)
        self.tree = KDTree(self.data_cloud)
        self.lambda_ = lam
        self.iteration_tol = iteration_tol
        self.search_range = search_range

    def compute_score(self, theta, model):
        # theta: (dx, dy) translation of all strokes
        from motorprogram import translate_motorprogram
        rendered = render_image(translate_motorprogram(model, theta[0], theta[1]))
        cloud = image_to_cloud(rendered)
        dists, _ = self.tree.query(cloud)
        emd = np.sum(dists) / len(dists)
        return len(cloud) / (emd ** self.lambda_)

    def cuckoo_search(self, model):
        """Simple cuckoo search over translation parameters."""
        dim = 2
        lb = np.array([-self.search_range] * dim)
        ub = np.array([self.search_range] * dim)
        n = 15
        pa = 0.25
        nests = np.random.uniform(lb, ub, (n, dim))
        fitness = np.array([-self.compute_score(nest, model) for nest in nests])
        best = nests[fitness.argmin()].copy()
        best_score = -fitness.min()

        for _ in range(self.iteration_tol):
            new_nests = nests + 0.5 * np.random.randn(n, dim)
            new_nests = np.clip(new_nests, lb, ub)
            for i in range(n):
                f = -self.compute_score(new_nests[i], model)
                if f <= fitness[i]:
                    fitness[i] = f
                    nests[i] = new_nests[i]
            idx = fitness.argmin()
            if fitness[idx] < -best_score:
                best_score = -fitness[idx]
                best = nests[idx].copy()

            rand_nests = np.random.uniform(lb, ub, (n, dim))
            mask = np.random.rand(n, dim) < pa
            new_nests = np.where(mask, rand_nests, nests)
            for i in range(n):
                f = -self.compute_score(new_nests[i], model)
                if f <= fitness[i]:
                    fitness[i] = f
                    nests[i] = new_nests[i]
            idx = fitness.argmin()
            if fitness[idx] < -best_score:
                best_score = -fitness[idx]
                best = nests[idx].copy()

        return best_score
