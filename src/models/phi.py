from flax import linen as nn
import jax.numpy as jnp
import jax

# Source : https://github.com/tensorflow/tensorflow/issues/56575 converted to jax
@jax.jit
def cdist(x, y):
  # Calculate distance for a single row of x.
  per_x_dist = lambda i : jnp.linalg.norm(x[:,i:(i+1),:] - y, axis=2)
  # Compute and stack distances for all rows of x.
  dist = jax.lax.map(per_x_dist,jnp.arange(0,jnp.shape(x)[1]))
  # Re-arrange stacks of distances.
  return jnp.transpose(dist, perm=[1, 0, 2])

class PHI(nn.Module):
	'''
    Implementation of feature transformation layer with RBF layer.
    We assume here that alpha is constant for all basis.
    '''
    in_features: int
    alpha: jnp.float32 = 1.0
    n_centres: int = 10
    hidden_dim1: int = 20
    hidden_dim2: int = 20
    out_dims: int = 100

    def setup(self):
    	self.centers = self.param('centers', nn.initializers.xavier_uniform(), (self.n_centres, self.in_features))
        self.linear1 = nn.Dense(self.hidden_dim1,self.n_centers)
        self.linear2 = nn.Dense(self.hidden_dim2,self.hidden_dim1)
        self.out = nn.Dense(self.out_dims,self.hidden_dim2)

    def __call__(self, x):

        rbf = jnp.exp(-1*jnp.square(cdist(x,self.centers)))
        hidden1 = jnp.tanh(self.linear1(rbf))
        hidden2 = jnp.tanh(self.linear2(hidden1))
        out = self.out(hidden2)
        
        return out
