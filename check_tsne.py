import sklearn
from sklearn.manifold import TSNE
import inspect

print(f"Scikit-learn version: {sklearn.__version__}")
sig = inspect.signature(TSNE.__init__)
print("TSNE parameters:", list(sig.parameters.keys()))
