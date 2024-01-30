import sys
sys.path.append("..")
import numpyro
import numpyro.distributions as npdist
import random as rnd
from numpyro.infer import Predictive, init_to_median, MCMC, NUTS
import jax
from jax import random
import jax.numpy as jnp
import torch
from models.phi_pytorch import PHI
from models.vae_pytorch import VAE

class pivae_numpyro_inference:

	def __init__(self,phi,vae):

		self.phi = phi
		vae_weights = vae.state_dict()
		self.w1 = jnp.asarray(weights_vae['linear1.weight'].T.numpy())
		self.w2 = jnp.asarray(weights_vae['linear2.weight'].T.numpy())
		self.w3 = jnp.asarray(weights_vae['out.weight'].T.numpy())
		self.b1 = jnp.asarray(weights_vae['linear1.bias'].T.numpy())
		self.b2 = jnp.asarray(weights_vae['linear2.bias'].T.numpy())
		self.b3 = jnp.asarray(weights_vae['out.bias'].T.numpy())

	def decoder(self,x,W1,W2,W3,B1,B2,B3):

		def layer(x,W,B):
			return jnp.matmul(x,W) + B

		hidden_1 = layer(x,W1,B1)
		hidden_2 = layer(jnp.tanh(hidden_1),W2,B2)
		output = layer(jnp.tanh(hidden_2),W3,B3)

		return output

	def numpyro_model(x, z_dim ,y ,obs_dim):

		z = numpyro.sample("z", npdist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))

		f = numpyro.deterministic("f", self.decoder(z,self.w1,self.w2,self.w3,self.b1,self.b2,self.b3))
		phi_x = jnp.asarray(phi(torch.tensor(x).float()).detach().numpy())

		y_hat = numpyro.deterministic("y_hat",jnp.matmul(phi_x,f))

		sigma = numpyro.sample("sigma", npdist.HalfNormal(0.025))

		y = numpyro.sample("y", npdist.Normal(y_hat[obs_idx], sigma), obs=y)

	def run_mcmc_vae(rng_key, numpyro_model, args, verbose=True):
	    start = time.time()

	    init_strategy = init_to_median(num_samples=10)
	    kernel = NUTS(numpyro_model, init_strategy=init_strategy)
	    mcmc = MCMC(
	        kernel,
	        num_warmup=args["num_warmup"],
	        num_samples=args["num_samples"],
	        num_chains=args["num_chains"],
	        thinning=args["thinning"],
	        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
	    )
	    start = time.time()
	    mcmc.run(rng_key,args["x"], args["z_dim"], args["y_obs"], args["obs_idx"])
	    t_elapsed = time.time() - start
	    if verbose:
	        mcmc.print_summary(exclude_deterministic=False)
	    
	    print("\nMCMC elapsed time:", round(t_elapsed), "s")
	    
	    return (mcmc, mcmc.get_samples(), t_elapsed)










