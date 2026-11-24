import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import *

# Import JAX things
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax, config
config.update("jax_enable_x64", True)








@jit
def fit_linear_model_analytically_jax(data, templates, sigma=None):
    """
    This solves a weighted least squares problem. We assume that:
    
    Data = a1 * template1 + a2 * template2 + ...

    Solves the weighted least squares problem analytically using JAX.
    This function is fully JIT-compilable.
    
    INPUTS
        data:      (N,) JAX array
        templates: (K, N) JAX array (or list/array that JAX can convert)
        sigma:     (N,) JAX array of uncertainties (optional)
        
    OUTPUTS
        amps:       (K,) array of best-fit amplitudes
        sigma_amps: (K,) array of uncertainties on amplitudes
        cov:        (K, K) covariance matrix

    Where N is the number of datapoints (e.g., in Rubin we have 6 bands) and K is the number of templates.


    The code below is written to quickly solve the weight least squares problem using matrices.
    
    y: observed data that we are fitting
    M: input data, in this case the different templates
    x: the parameters we are solving for, in this case the amplitudes
    W: the weights, in this case the uncertainties. I.e., W_ii = 1/sigma_i**2

    The equation above can be rewritten in matrix form:

    (M^T W M) x = M^T W y
    => x = (M^T W M)^-1 M^T W y

    Where the covariance matrix is equal to: (M^T W M)^-1, so that:

    => x = Cov M^T W y
    
    Some links: 
    https://en.wikipedia.org/wiki/Weighted_least_squares
    

    As soon as the problem becomes:
    data=a⋅f(b⋅x+c) + etc
    this won't work, and better to use scipy.optimize.least_squares, I think.
    """
    # Convert inputs to JAX arrays. 
    # If templates is passed as (K, N), we transpose to (N, K) to match your 'M'
    y = jnp.asarray(data)
    M = jnp.asarray(templates).T  # Shape: (N, K)
    
    # 1. Handle weights without dynamic Python branching.
    # Instead of an 'if sigma is not None' Python branch which breaks JIT 
    # if the presence of sigma changes, we use JAX-compatible logic.
    if sigma is not None:
        W_diag_sq = 1.0 / jnp.asarray(sigma)
        y = y * W_diag_sq
        M = M * W_diag_sq[:, jnp.newaxis]
        
    # 2. Matrix Multiplication
    MTM = M.T @ M
    
    # 3. Safe Inversion
    # We use pseudo-inverse (pinv) instead of inv. 
    # This prevents the solver from crashing or raising LinAlgError if templates 
    # are linear duplicates (which can happen in binary fits before perturbation).
    cov = jnp.linalg.pinv(MTM)
    MTY = M.T @ y
    
    # 4. Compute amplitudes
    amps = cov @ MTY
    
    # 5. Extract uncertainties
    diagonals = jnp.diag(cov)
    sigma_amps = jnp.sqrt(jnp.maximum(diagonals, 0.0))
    
    return amps, sigma_amps, cov




@jit
def fit_linear_model_analytically_batch_jax(data, template_batch, sigma=None):
    """
    Solves a weighted least squares problem simultaneously for a massive batch 
    of templates using a single, unified, JIT-compiled JAX function.
    
    INPUTS:
        data:           (N,) array (observed data fluxes)
        template_batch: (M, K, N) array (M combinations, K templates, N bands)
        sigma:          (N,) array of uncertainties (optional)
    """
    # Part 0: Initialize and align shapes y -> (N, 1), M -> (M, N, K)
    y = jnp.asarray(data, dtype=jnp.float64)[:, jnp.newaxis]
    M = jnp.asarray(template_batch, dtype=jnp.float64)
    
    # Part 1: Add uncertainty if needed
    if sigma is not None:
        W_diag_sq = 1.0 / jnp.asarray(sigma, dtype=jnp.float64)#[:, jnp.newaxis]
        y = y * W_diag_sq[:, jnp.newaxis]
        M = M * W_diag_sq[:, jnp.newaxis]

    # Part 2: Define matrix multiplication
    def solve_single(single_M):
        MTM = single_M.T @ single_M
        cov = jnp.linalg.pinv(MTM, rtol=0.0)
        MTY = single_M.T @ y
        
        amps = (cov @ MTY).squeeze(-1)
        diagonals = jnp.diag(cov)
        sigma_amps = jnp.sqrt(jnp.maximum(diagonals, 0.0))
        
        return amps, sigma_amps, cov

    
    # Part 3: Vectorize and execute over the template batch axis (dimension 0 of M)
    amps, sigma_amps, cov = vmap(solve_single, in_axes=0)(M)
    
    return amps, sigma_amps, cov




@jit#(static_argnames=('max_iter',))
def fit_linear_model_numerically_jax(data, templates, sigma=None, learning_rate=None, max_iter=10_000, tol = 1e-20):
    """
    Solves a weighted linear least squares problem:
    
    Fits data = a1 * template1 + a2 * template2 + ... 

    using differentiable programming (Autodiff + Gradient Descent). Amplitudes can be 
    any real number from -inf to +inf.
    
    INPUTS:
        data:       (N,) JAX array
        templates:  (K, N) list/array or (N, K) array of template fluxes
        sigma:      (N,) JAX array of uncertainties (optional)
        
    OUTPUTS:
        amps:       (K,) array of best-fit amplitudes
        sigma_amps: (K,) array of uncertainties on amplitudes
        cov:        (K, K) covariance matrix

    If equation to solve becomes more complicated than a linear equation, I think I need to adjust loss_fn accordingly.


    TODO: I can probably use JAX's vmap to make it a batch solver if needed.
    """
    
    # Align shapes to N rows, K columns
    y = jnp.asarray(data, dtype=jnp.float64)
    

    M = jnp.asarray(templates, dtype=jnp.float64)
    if M.ndim == 1:
        M = M[:, jnp.newaxis]
    elif M.shape[0] != y.shape[0]:
        M = M.T  # Shape: (N, K)



            

        
    # Apply weights if needed
    if sigma is not None:
        W_diag = 1.0 / jnp.asarray(sigma, dtype=jnp.float64)
        y = y * W_diag
        M = M * W_diag[:, jnp.newaxis]

    # Calculate MTM
    MTM = M.T @ M

    # Determine maximum safe learning rate with Lipschitz constant
    if learning_rate is None: 
        max_eigenvalue = jnp.max(jnp.abs(jnp.linalg.eigvalsh(MTM)))
        learning_rate = jnp.where(max_eigenvalue > 0.0, 0.5 / max_eigenvalue, 0.01) # Default to 0.01 if the matrix is completely zero, otherwise auto-scale step
        learning_rate = learning_rate/4 # Take a quarter of it, just to be safe


    # Define the Raw Physical Loss Function and differentiate with JAX
    def loss_fn(raw_amps):
        prediction = M @ raw_amps
        return jnp.sum((y - prediction) ** 2)

    loss_and_grad_fn = jax.value_and_grad(loss_fn)



    # Initialize amplitudes at zero and define the init_state
    amps_initial = jnp.zeros(M.shape[1])
    init_loss = loss_fn(amps_initial)
    init_state = (0, amps_initial, init_loss, jnp.inf) #(step_idx, current_amps, current_loss, relative_delta)

    # Condition function: Keep running if we haven't hit max steps AND change is above tolerance
    def condition_fun(state):
        step_idx, _, _, rel_delta = state
        return (step_idx < max_iter) & (rel_delta > tol)

    # Define the optimization loop step
    def optimization_step(state):
        step_idx, current_amps, old_loss, _ = state
        
        loss_val, gradients = loss_and_grad_fn(current_amps)
        next_amps = current_amps - learning_rate * gradients
        
        new_loss = loss_fn(next_amps)
        

        rel_loss_change = jnp.abs(new_loss - old_loss) / (old_loss)# + 1e-15) # Calculate fractional change in loss
        
        return (step_idx + 1, next_amps, new_loss, rel_loss_change)

    # Run the optimization loop
    final_step_idx, final_amps, final_new_loss, final_rel_loss_change = lax.while_loop(condition_fun, optimization_step, init_state)

    # jax.debug.print("Converged early at step: {x}", x=final_step_idx)
    # jax.debug.print("Final new loss: {x}", x=final_new_loss)
    # jax.debug.print("Final rel loss change: {x}", x=final_rel_loss_change)


    #### OLD CODE

    # Initialize amplitudes at zero
    # amps_initial = jnp.zeros(M.shape[1])

    # # Define the optimization loop step
    # def optimization_step(step_idx, current_amps):
    #     loss_val, gradients = loss_and_grad_fn(current_amps)
    #     return current_amps - learning_rate * gradients

    # # Run the optimization loop
    # final_amps = jax.lax.fori_loop(0, max_iter, optimization_step, amps_initial)

    ####
    

    # Error Propagation
    cov = jnp.linalg.pinv(MTM, rtol=0.0)
    diagonals = jnp.diag(cov)
    sigma_amps = jnp.sqrt(jnp.maximum(diagonals, 0.0))

    return final_amps, sigma_amps, cov