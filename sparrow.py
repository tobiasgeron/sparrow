import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.optimize import lsq_linear
import multiprocessing
        

from utils import *





### ============== ###
### Main functions ###
### ============== ###


def fit_linear_model_analytically(data, templates, sigma=None):
    """
    This solves a weighted least squares problem. We assume that:
    
    Data = a1 * template1 + a2 * template2 + ...
    
    INPUTS
        data:     (N,) array
        templates: list of (N,) arrays
        sigma:    (N,) array of errors on data (optional)

    OUTPUTS
        amps:     (K,) array of best-fit amplitudes
        sigma_amps: (K,) array of uncertainties
        cov:      (K, K) covariance matrix

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

    y = np.asarray(data)
    M = np.asarray(templates).T

    if sigma is not None:
        #We apply the Cholesky Shortcut to transform Weighted least squares into ordinary least squares.
        # In essence this means we don't actually compute M^T @ W, instead we just multiply M and y with the diagonal
        # of the weight matrix. This assumes weight matrix is a diagonal (i.e., no off-diagonal elements, i.e., errors 
        # are independent) and all positive, which in our case is true. Doing this speeds things up as we don't have to 
        # do the full matrix multiplication.
        # We also do only 1 / sigma instead of 1/sigma**2, as later you need to do M * sqrt(W_diag), so the **2 and sqrt cancel out

        W_diag_sq = 1.0 / np.asarray(sigma)
        y = y * W_diag_sq
        M = M * W_diag_sq[:, np.newaxis]
        

    MTM = M.T @ M # Matrix multiplication
    cov = np.linalg.inv(MTM)
    MTY = M.T @ y

    amps = cov @ MTY
    
    diagonals = np.diagonal(cov)
    sigma_amps = np.sqrt(np.maximum(diagonals, 0))
    #sigma_amps = np.sqrt(np.diag(cov))
        
    return amps, sigma_amps, cov



def fit_linear_model_analytically_batch(data, template_batch, sigma = None):
    """
    This solves a weighted least squares problem for a batch of templates. We assume that:
    
    Data = a1 * template1 + a2 * template2 + ...

    But this code does it simultaneously for any M templates. Essentially the same as doing:

    for t in template:
        fit_linear_model_analytically()


    INPUTS:
    data: (N,) array 
    template_batch: (M, N, K) array 
        M = number of independent fits to perform (e.g., 1,000,000)
        N = number of data points (e.g., 6 bands)
        K = number of templates in each fit (e.g., 1 for single star and 2 for binary)
    sigma: (N,) array

    OUTPUTS:
    amplitudes, sigma_amplitudes, covariance matrices


    Additional notes:
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

    ### Part 0: Initialize
    y = data[np.newaxis, :, np.newaxis]# Shape (1, N, 1) for broadcasting, as template_batch is (M, N, K). Now N axis aligns
    M = template_batch

    ### Part 1: Add uncertainty if needed
    if sigma is not None:
        W_diag_sq = (1.0 / np.asarray(sigma))[np.newaxis, :, np.newaxis] # Shape (1, N, 1) for broadcasting
        y = y * W_diag_sq
        M = template_batch * W_diag_sq
    
    ### Part 2 Matrix Multiplication
    # M.swapaxes(1, 2) is (M, K, N)
    MTM = M.swapaxes(1, 2) @ M # Result: (M, K, K)
    MTY = M.swapaxes(1, 2) @ y # Result: (M, K, 1)
    cov = np.linalg.inv(MTM) # Shape (M, K, K)


    ### Part 3: Calculate amplitudes and uncertainties
    amps = (cov @ MTY).squeeze(-1) # (M, K, K) @ (M, K, 1) => (M, K, 1)

    diagonals = np.diagonal(cov, axis1=1, axis2=2)
    sigma_amps = np.sqrt(np.maximum(diagonals, 0))

    #sigma_amps = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))

    return amps, sigma_amps, cov



def fit_linear_model_numerically(data, templates, sigma=None):
    """
    Solves a weighted linear least squares problem with positivity constraint on amplitudes.
    
    Fits data = a1 * template1 + a2 * template2 + ... with a_i >= 0

    INPUTS
        data:     (N,) array
        templates: list of (N,) arrays
        sigma:    (N,) array of errors on data (optional)

    OUTPUTS
        amps:       (K,) array of best-fit amplitudes
        sigma_amps: (K,) array of uncertainties (approximate)
        cov:        (K, K) covariance matrix (approximate)


    If equation to solve becomes more complicated than a linear equation, then I need to use a different solver. 
    """
    y = np.asarray(data, dtype=np.float64)
    
    if isinstance(templates, list):
        M = np.vstack([np.asarray(t) for t in templates]).T  # shape (N, K)
    else:
        M = np.asarray(templates, dtype=np.float64)  # already shape (N, K)
 
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        W = 1.0 / sigma
        y = y * W
        M = M * W[:, np.newaxis]

    # Solve with non-negative constraint
    res = lsq_linear(M, y, bounds=(0, np.inf), method='trf')

    amps = res.x

    # Approximate covariance and uncertainties (based on linear approximation)
    MTM = M.T @ M
    try:
        cov = np.linalg.inv(MTM)
        sigma_amps = np.sqrt(np.diag(cov))

    except np.linalg.LinAlgError:
        cov = np.full((M.shape[1], M.shape[1]), np.nan)
        sigma_amps = np.full(M.shape[1], np.nan)


    """# Approximate covariance matrix from Jacobian
    if res.success:
        _, s, VT = np.linalg.svd(M, full_matrices=False)
        threshold = np.finfo(float).eps * max(M.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:len(s)]
        cov = VT.T @ np.diag(1 / s**2) @ VT
        sigma_amps = np.sqrt(np.diag(cov))
    else:
        cov = np.full((M.shape[1], M.shape[1]), np.nan)
        sigma_amps = np.full(M.shape[1], np.nan)"""

    return amps, sigma_amps, cov







def fit_single_star(template_data, templates, throughput_curves = {}, rescale_data = False, rescale_templates = False, method = 'analytical-batch', 
                    dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), Av_range = (0,0), Av_stepsize = 0.1, extinction_function = None,
                    full_output = False, only_main_sequence = False, only_main_sequence_and_giants = False, delta_MS = 2, cache_new_extinction_values = True, 
                    timeit = False, use_mass_constraint = True, main_sequence_model = None):

    '''
    Description
    -----------

    This function tries to fit the data in `template_data' using the templates provided in `templates'. We're essentially solving:

    data = A1 * template 1 

    where data and template contain the fluxes for N bands, and A = (R/D)**2.



    Parameters
    ----------

    template_data : `utils.sed_template'
                A `utils.sed_template' that contains the flux (in template_data.flux) and flux uncertainties (in template_data.e_flux) that we want to fit

    templates : list of `utils.sed_template'
                We use the templates in this list to fit the data.

    throughput_curves : dict or None
                A dict of the throughput curve for each band. This has to have a particular format, please see the additional documentation on throughput curves. This is needed in case we need to calculate fluxes for Av values that were not precomputed.

    rescale_data : bool, default = False
                Whether to rescale the data to 1 by dividing it by its maximum value. This can be turned on to avoid numerical instabilities.

    rescale_templates: bool, default = False
                Whether to rescale all the templates to 1 by dividing each template by its own maximum value. This can be turned on to avoid numerical instabilities.

    method: string, default = 'analytical-batch'
                Which method to use to do the fit. Must be one of the following: 'analytical-batch-jax','analytical-batch','analytical-jax','analytical','numerical','numerical-jax'. We STRONGLY suggest to keep this at the default value of 'analytical-batch' or 'analytical-batch-jax', as they are much faster. More information on the different fitting methods is given below.

    dist_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper distance limits that we will fit. The units must be in kpc. The lower limit must be specified first, then the upper limit.

    radius_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper radius limits that we will fit. The units must be in solar radii. The lower limit must be specified first, then the upper limit.

    mass_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper mass limits that we will fit. The units must be in solar masses. The lower limit must be specified first, then the upper limit. These mass limits will be used, together with the log(g) associated with each template, to calculate appropriate limits on the radius of the star. This will only work if templates have associated log(g) values accessible by doing: `template.g' and if use_mass_constraint = True (see below).

    Av_range : tuple, default = (0,0)
                A tuple to define the lower and upper Av limits that we will fit. The units must be in mag. The lower limit must be specified first, then the upper limit. If equal to (0,0), no extinction will be applied. See `extinction_function' on more details on how this is implemented. 

    Av_stepsize : float, default = 0.1
                The stepsize with which we will sample Av. The different Av values that are tested are calculated as follows: ''np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)''. Lower values means higher precision, at the cost of compute time. 

    extinction_function : dust_extinction model or None, default = None
                The user can specify a dust extinction model from the "dust_extinction" Python package. More information here: https://dust-extinction.readthedocs.io/en/latest/index.html. If no law is specified, we will use the Fitzpatrick 1999 law with Rv = 3.1. For example, you can specify as follows: 
                ```
                from dust_extinction.parameter_averages import F99
                extinction_function = F99(Rv = 3.1)
                ```

    full_output : bool, default = False
                Whether to return a dictionary with only the best-fit parameters of the best-fit combination of templates (when `full_output' == False) or a pandas.DataFrame with the best-fit parameters for all combinations of templates.

    only_main_sequence : bool, default = False
                Whether to only fit stars around the main sequence. The main sequence is defined with `delta_MS' (see below).

    only_main_sequence_or_giants : bool, default = False
                Whether to only fit stars around the main sequence and above (i.e., all types of giants). The main sequence is defined with `delta_MS' (see below).

    delta_MS : float, default = 2
                This is used to specify what counts as main sequence. We first fit a line on the HR diagram through common main sequence stars. All stars that are `delta_MS' values in log-luminosity space away from this best-fit line are still considered main sequence. 

    cache_new_extinction_values : bool, default = True
                Whether to cache the fluxes for Av values that were not precomputed so that subsequent lookup will be faster. The downside of this is that the RAM will blow up if you're testing a lot of new Av values.

    timeit : bool, default = False
                Whether to perform an analysis of how long each step takes. Added for debugging purposes, keep on default value of False. 
    
    use_mass_constraint : bool, default = True
                Whether to limit the sample based on mass_range. Added for debugging purposes.
                
    Returns
    -------

    df_output : dict or pandas.DataFrame
                If `full_output' == False, the output will be a dictionary with the best-fit Amplitude value for the best-fit template. If `full_output' == True, the output will be a pandas DataFrame with the best-fit amplitude values for every combination of templates.


    Example
    -------

    df_output = fit_single_star(data, templates)


    
    Notes
    -----

    --- A note on measurement uncertainties ---
    If your data templates contain uncertainties in self.e_flux, this function automatically extracts them and uses them in the least-squares fitting process.


    --- A note on the different fitting methods ---
    We implemented three different methods to fit the data, described in more detail below. We STRONGLY suggest to keep this at the default value of 'analytical-batch' as it is much faster. 
    -'analytical': As we're essentially solving a linear problem (data = A1 * template1 + A2 * template2), we can easily solve for the best-fit A1 and A2 values analytically using a weighted least squares approach. When method == 'analytical', that's what we do for every possible combination of templates.
    -'analytical-jax': same as analytical, but we use JAX. Need to have JAX installed for this to work. 
    -'analytical-batch': The maths and approach is the same as the above, but we stack all the template vectors so we can solve all the equations in one go. This avoids using nested for-loops and is *much* faster.
    -'analytical-batch-jax': Same as analytical-batch, but we use JAX. Need to have JAX installed for this to work. 
    -'numerical': Instead of solving for A1 and A2 analytically, we use scipy's lsq_linear() function to do this numerically. This approach was incorporated in the beginning of development, but we strongly advise against using it. The other two approaches are more accurate and faster.
    -'numerical-jax': Similar to numerical, but implemented with JAX. Need to have JAX installed for this to work. 
    
    --- A note on the mass constraints ---
    We can put thresholds on the mass that we're fitting (as provided by mass_limits). This is because we know that: g = G*M/R^2 and A = R^2/D^2. So combined:
    g = G * M / (A * D^2)
    <=> M = g / G * (A * D^2)
    We can then input the minimum and maximum possible distance, and you then get a minimum and maximum mass range for that specific log_g. In practise, in the code we solve for A instead and limit that, but the idea is the same. 


    --- Additional notes ---
    If the uncertainties in the data (in template_data.e_flux) are not well characterized, then the resultant sigma_amp values will also not be accurate.

    '''
    

    if timeit:
        sw=stopwatch()
        sw.register_processes(['Preprocess','Fit','Rescale','Amp_limits','Find_best_fit','Output'])
        sw.start()



    ### ========== ###
    ### Preprocess ###
    ### ========== ###


    if method in ['analytical-jax', 'analytical-batch-jax','numerical-jax']:
        try:
            import template_fitting_jax
        except ImportError as e:
            raise ImportError(
                "The experimental solver requires JAX to be installed. "
            ) from e

     
    data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
    sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
    if len(sigma) == 0:
        sigma = None

        
    templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 
    
    radius_range = (radius_range[0]*u.Rsun.to(u.kpc),radius_range[1]*u.Rsun.to(u.kpc)) #Convert Rsun to kpc


    # Setup Av_lst
    #Av_lst = np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)
    Av_lst = safe_arange(Av_range[0], Av_range[1], Av_stepsize)
    if len(Av_lst) == 0:
        Av_lst = [0]


    # Normalize data to itself
    if rescale_data: 
        scale = np.max(data)
        data = data/scale
        if sigma is not None:
            sigma = np.array(sigma)/scale
    else:
        scale = 1

    
    # Normalize templates to themselves
    if rescale_templates:
        template_scales = np.array([np.max(t.flux) for t in templates]) 
        for i, t in enumerate(templates):
            t.flux = t.flux / template_scales[i]
            t.full_flux = t.full_flux / template_scales[i]

            # Adjust extinction_cache fluxes too (by the same scale for simplicity)
            keys = t.extinction_cache.keys()
            for key in keys:
               t.extinction_cache[key]['flux'] = t.extinction_cache[key]['flux'] / template_scales[i]

    
    # Calculate limits on amplitude
    # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
    if only_main_sequence:
        amp_min = []
        amp_max = []

        for i,t in enumerate(templates):
            radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS, main_sequence_model = main_sequence_model)
            
            radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
            amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
            amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )

        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)


    elif only_main_sequence_and_giants:
        amp_min = []
        amp_max = []

        for i,t in enumerate(templates):
            radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS, main_sequence_model = main_sequence_model)

            radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
            amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
            amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )

        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)

    else:
        amp_min = [(np.min(radius_range)/np.max(dist_range))**2]*len(templates) 
        amp_max = [(np.max(radius_range)/np.min(dist_range))**2]*len(templates)




    # Calculate limits on amp based on mass_range and log_g
    # Instead of limits on mass, perhaps recalculate this as limits on amp? Is that possible?
    if use_mass_constraint:
        amp_min_mass = []
        amp_max_mass = []
        for i,t in enumerate(templates):
            log_g = t.g
            # G = 6.6743*10**-11 #m3 kg-1 s-2
            # g = 10**(log_g) / 100 #cm s-2 to m s-2

            # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
            # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

            # CONST contains all scaling params above

            CONST = 1.3938323614347172e-17
            amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
            amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)





    if timeit:
        sw.log_process('Preprocess')
        sw.start()
    

    ### ======= ###
    ### Fitting ###
    ### ======= ###

    
    Avs = []
    indices = []

    ### Step 1: Calculated fluxes to be tested
    ### Alternatively, keep track of only indices, and then in the step 2, actually save the fluxes with the different Avs. Might work.
    ### Also, if memory is an issue, you can put the linear bit back below actually and it should work. 
    flux_container = [] #NEW

    for i, template in enumerate(templates):
        for i_av, Av in enumerate(Av_lst):
            # if Av < Av_range[0]: #Sometimes np.arange causes weird floating point errors...
            #     continue
            # if Av > Av_range[1]:
            #     continue


            ### Get the flux and do the fit
            if Av == 0.0:
                flux = template.flux
            else:
                flux = template.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
                flux = flux['flux']
                 
            flux_container.append([flux]) #NEW
            indices.append([i])
            Avs.append(Av)


    
    ### Step 2: Solve 
    if method == 'analytical-batch':
        """
        template_batch: (M, N, K) array 
        M = number of independent fits to perform (e.g., 1,000,000)
        N = number of data points (e.g., 6 bands)
        K = number of templates in each fit (e.g., 1 for single star and 2 for binary)
        """
        # TODO: Can also add batches() here to allow for MP?

        #flux_container = np.swapaxes(flux_container, 1, 2) #Make sure flux_container has the right shape
        amps, sigma_amps, _ = fit_linear_model_analytically_batch(np.array(data), np.swapaxes(flux_container, 1, 2), sigma = sigma)
        

    elif method == 'analytical-batch-jax':
        """
        template_batch: (M, N, K) array 
        M = number of independent fits to perform (e.g., 1,000,000)
        N = number of data points (e.g., 6 bands)
        K = number of templates in each fit (e.g., 1 for single star and 2 for binary)
        """
        # TODO: Can also add batches() here to allow for MP?

        #flux_container = np.swapaxes(flux_container, 1, 2) #Make sure flux_container has the right shape
        amps, sigma_amps, _ = template_fitting_jax.fit_linear_model_analytically_batch_jax(np.array(data), np.swapaxes(flux_container, 1, 2), sigma = sigma)
        amps = np.asarray(amps).copy() #JAX arrays are immutable, so need to convert back to numpy here
        sigma_amps = np.asarray(sigma_amps).copy()
        



    else: 
        amps = []
        sigma_amps = []
        for i, flux in enumerate(flux_container): 
            if method == 'analytical':
                amp, sigma_amp, _ = fit_linear_model_analytically(data, flux, sigma = sigma)
            elif method == 'analytical-jax':
                amp, sigma_amp, _ = template_fitting_jax.fit_linear_model_analytically_jax(data, flux, sigma = sigma)
            elif method == 'numerical':
                amp, sigma_amp, _ = fit_linear_model_numerically(data, flux, sigma = sigma)
            elif method == 'numerical-jax':
                amp, sigma_amp, _ = template_fitting_jax.fit_linear_model_numerically_jax(data, flux, sigma = sigma)

            amps.append(amp)
            sigma_amps.append(sigma_amp)


    if timeit:
        sw.log_process('Fit')
        sw.start()
        
    

    ### Step 3: rescale amps and adjust amplitudes constraints

    for i in range(len(amps)):
        amp = amps[i]
        sigma_amp = sigma_amps[i]
        idx_template = indices[i][0]
            
        # Rescale the fit parameters back wrt data scale
        if rescale_data:
            amp = amp*scale
            if sigma is not None: #Wait, I need to rescale back regardless, no? -> No I double checked, it is okay here.
                sigma_amp = sigma_amp * scale

        # Rescale the fit parameters back wrt template scale
        if rescale_templates: 
            template_scale = template_scales[idx_template]
            amp = amp/template_scale
            sigma_amp = sigma_amp/template_scale



        
        # Add amp limits
        if amp < amp_min[idx_template]:
            amp = np.array([amp_min[idx_template]])
        elif amp > amp_max[idx_template]:
            amp = np.array([amp_max[idx_template]])


        # OLD MASS CONSTRAINT BELOW
        # if use_mass_constraint:
        #     kpc2Rsun = 44353565926.28097
        #     log_g = templates[idx_template].g
        #     pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(amp*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(amp*np.max(dist_range)**2)*kpc2Rsun, log_g)

        #     # G = 6.6743*10**-11 #m3 kg-1 s-2
        #     # g = 10**(log_g) / 100 #cm s-2 to m s-2
            
        #     if pred_mass_min > np.max(mass_range):
        #         # amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
        #         # amp = np.array([amp_new])
        #         amp = np.array([amp_max_mass[idx_template]])

        #     if pred_mass_max < np.min(mass_range):
        #         # amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
        #         # amp = np.array([amp_new])
        #         amp = np.array([amp_min_mass[idx_template]])

        #NEW MASS CONSTRAINT
        if use_mass_constraint:
            if amp < amp_min_mass[idx_template]:
                amp = np.array([amp_min_mass[idx_template]])
            elif amp > amp_max_mass[idx_template]:
                amp = np.array([amp_max_mass[idx_template]])
                
                


        # Overwrite
        amps[i] = amp
        sigma_amps[i] = sigma_amp
    
            





    if timeit:
        sw.log_process('Amp_limits')
        sw.start()

    




    ### ============ ###
    ### Post process ###
    ### ============ ###
    


    flux_container = np.array(flux_container)   # Shape: (N, 1, 6)
    indices = np.array(indices)           # Shape: (N, 1)
    amps = np.array(amps) #Shape: (N, 1)

    # Scale data flux back wrt scale
    data = data*scale
    if sigma is not None:
        sigma = np.array(sigma)*scale
    
    

    # Scale template flux back wrt template_scales
    if rescale_templates: 
        for i, t in enumerate(templates):
            t.flux = t.flux * template_scales[i]


            # Adjust extinction_cache fluxes too (by the same scale for simplicity)
            keys = t.extinction_cache.keys()
            for key in keys:
               t.extinction_cache[key]['flux'] = np.array(t.extinction_cache[key]['flux']) * template_scales[i]



        # Rescale flux container too
        temp_template_scales = template_scales[indices] 
        flux_container = flux_container * temp_template_scales[:, :, np.newaxis]

        



    if timeit:
        sw.log_process('Rescale')
        sw.start()

    
    # Calculate error on the fit and find best fit
    results = (amps[:, np.newaxis, :] @ flux_container).squeeze()
    errors = calculate_error(data, results, sigma)

    
    # errors = []
    # for i in range(len(amps)):
    #     amp = [amps[i]]
    #     idx = indices[i][0]
    #     template = flux_container[i]


    #     result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
    #     error = calculate_error(data, result, sigma = sigma)
    #     errors.append(error)







    if timeit:
        sw.log_process('Find_best_fit')
        sw.start()




    
    ### Return results
    df_output = pd.DataFrame()
    df_output['error'] = errors
    df_output['amp'] = [i[0] for i in amps]
    df_output['sigma_amp'] = [i[0] for i in sigma_amps]
    df_output['Av'] = Avs
    df_output['idx'] = [int(i[0]) for i in indices]
    df_output['idx'] = df_output['idx'].astype(int)

    all_templates_in_order = templates[indices]
    df_output['temperature'] = [t[0].T for t in all_templates_in_order]
    df_output['g'] = [t[0].g for t in all_templates_in_order]
    df_output['Z'] = [t[0].Z for t in all_templates_in_order]

    df_output = df_output.loc[df_output.groupby('idx')['error'].idxmin()].reset_index(drop=True)



    if timeit:
        sw.log_process('Output')
        sw.output_processes()
        sw.plot_processes()
    
    
    if full_output: #Output full table
        return df_output


    else:
        idx_best = np.argmin(df_output['error'])
        row = df_output.iloc[idx_best][['idx','amp','sigma_amp','Av','error']].to_dict()
        row['idx'] = int(row['idx'])
        return row
        





def fit_binary(template_data, templates, throughput_curves = {}, rescale_data = False, rescale_templates = False, method = 'analytical-batch', 
                    dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), Av_range = (0,0), Av_stepsize = 0.1, extinction_function = None, 
                    full_output = False, only_main_sequence = False, only_main_sequence_and_giants = False, delta_MS = 2, cache_new_extinction_values = True, 
                    timeit = False, use_mass_constraint = True, use_amplitude_ratio_constraint = False, main_sequence_model = None):

    '''
    Description
    -----------

    This function tries to fit the data in `template_data' using the templates provided in `templates'. We're essentially solving:

    data = A1 * template1 + A2 * template2

    where data and template contain the fluxes for N bands, and A = (R/D)**2



    Parameters
    ----------

    template_data : `utils.sed_template'
                A `utils.sed_template' that contains the flux (in template_data.flux) and flux uncertainties (in template_data.e_flux) that we want to fit

    templates : list of `utils.sed_template'
                We use the templates in this list to fit the data.

    throughput_curves : dict or None
                A dict of the throughput curve for each band. This has to have a particular format, please see the additional documentation on throughput curves. This is needed in case we need to calculate fluxes for Av values that were not precomputed.

    rescale_data : bool, default = False
                Whether to rescale the data to 1 by dividing it by its maximum value. This can be turned on to avoid numerical instabilities.

    rescale_templates: bool, default = False
                Whether to rescale all the templates to 1 by dividing each template by its own maximum value. This can be turned on to avoid numerical instabilities.

    method: string, default = 'analytical-batch'
                Which method to use to do the fit. Must be one of the following: 'analytical-batch-jax','analytical-batch','analytical-jax','analytical','numerical','numerical-jax'. We STRONGLY suggest to keep this at the default value of 'analytical-batch' or 'analytical-batch-jax', as they are much faster. More information on the different fitting methods is given below.

    dist_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper distance limits that we will fit. The units must be in kpc. The lower limit must be specified first, then the upper limit.

    radius_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper radius limits that we will fit. The units must be in solar radii. The lower limit must be specified first, then the upper limit.

    mass_range : tuple, default = (0,np.inf)
                A tuple to define the lower and upper mass limits that we will fit. The units must be in solar masses. The lower limit must be specified first, then the upper limit. These mass limits will be used, together with the log(g) associated with each template, to calculate appropriate limits on the radius of the star. This will only work if templates have associated log(g) values accessible by doing: `template.g' and if use_mass_constraint = True (see below).

    Av_range : tuple, default = (0,0)
                A tuple to define the lower and upper Av limits that we will fit. The units must be in mag. The lower limit must be specified first, then the upper limit. If equal to (0,0), no extinction will be applied. See `extinction_function' on more details on how this is implemented. 

    Av_stepsize : float, default = 0.1
                The stepsize with which we will sample Av. The different Av values that are tested are calculated as follows: ''np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)''. Lower values means higher precision, at the cost of compute time. 

    extinction_function : dust_extinction model or None, default = None
                The user can specify a dust extinction model from the "dust_extinction" Python package. More information here: https://dust-extinction.readthedocs.io/en/latest/index.html. If no law is specified, we will use the Fitzpatrick 1999 law with Rv = 3.1. For example, you can specify as follows: 
                ```
                from dust_extinction.parameter_averages import F99
                extinction_function = F99(Rv = 3.1)
                ```

    full_output : bool, default = False
                Whether to return a dictionary with only the best-fit parameters of the best-fit combination of templates (when `full_output' == False) or a pandas.DataFrame with the best-fit parameters for all combinations of templates.

    only_main_sequence : bool, default = False
                Whether to only fit stars around the main sequence. The main sequence is defined with `delta_MS' (see below).

    only_main_sequence_or_giants : bool, default = False
                Whether to only fit stars around the main sequence and above (i.e., all types of giants). The main sequence is defined with `delta_MS' (see below).

    delta_MS : float, default = 2
                This is used to specify what counts as main sequence. We first fit a line on the HR diagram through common main sequence stars. All stars that are `delta_MS' values in log-luminosity space away from this best-fit line are still considered main sequence. 

    cache_new_extinction_values : bool, default = True
                Whether to cache the fluxes for Av values that were not precomputed so that subsequent lookup will be faster. The downside of this is that the RAM will blow up if you're testing a lot of new Av values.

    timeit : bool, default = False
                Whether to perform an analysis of how long each step takes. Added for debugging purposes, keep on default value of False. 
    
    use_mass_constraint : bool, default = True
                Whether to limit the sample based on mass_range. Added for debugging purposes.

    use_amplitude_ratio_constraint : bool, default = False
                Currently not implemented. 
                
    Returns
    -------

    df_output : dict or pandas.DataFrame
                If `full_output' == False, the output will be a dictionary with the best-fit Amplitude value for the best-fit template. If `full_output' == True, the output will be a pandas DataFrame with the best-fit amplitude values for every combination of templates.


    Example
    -------

    df_output = fit_double_star(data, templates)


    
    Notes
    -----

    --- A note on measurement uncertainties ---
    If your data templates contain uncertainties in self.e_flux, this function automatically extracts them and uses them in the least-squares fitting process.

    
    --- A note on the different fitting methods ---
    We implemented three different methods to fit the data, described in more detail below. We STRONGLY suggest to keep this at the default value of 'analytical-batch' as it is much faster. 
    -'analytical': As we're essentially solving a linear problem (data = A1 * template1 + A2 * template2), we can easily solve for the best-fit A1 and A2 values analytically using a weighted least squares approach. When method == 'analytical', that's what we do for every possible combination of templates.
    -'analytical-jax': same as analytical, but we use JAX. Need to have JAX installed for this to work. 
    -'analytical-batch': The maths and approach is the same as the above, but we stack all the template vectors so we can solve all the equations in one go. This avoids using nested for-loops and is *much* faster.
    -'analytical-batch-jax': Same as analytical-batch, but we use JAX. Need to have JAX installed for this to work. 
    -'numerical': Instead of solving for A1 and A2 analytically, we use scipy's lsq_linear() function to do this numerically. This approach was incorporated in the beginning of development, but we strongly advise against using it. The other two approaches are more accurate and faster.
    -'numerical-jax': Similar to numerical, but implemented with JAX. Need to have JAX installed for this to work. 
    
    
    --- A note on the mass constraints ---
    We can put thresholds on the mass that we're fitting (as provided by mass_limits). This is because we know that: g = G*M/R^2 and A = R^2/D^2. So combined:
    g = G * M / (A * D^2)
    <=> M = g / G * (A * D^2)
    We can then input the minimum and maximum possible distance, and you then get a minimum and maximum mass range for that specific log_g. In practise, in the code we solve for A instead and limit that, but the idea is the same. 


    --- Additional notes ---
    If the uncertainties in the data (in template_data.e_flux) are not well characterized, then the resultant sigma_amp values will also not be accurate.

    '''

    
    if timeit:
        sw=stopwatch()
        sw.register_processes(['Preprocess','Fit','Rescale','Amp_limits','Find_best_fit','Output'])
        sw.start()


    ### ========== ###
    ### Preprocess ###
    ### ========== ###

    if method in ['analytical-batch-jax', 'analytical-jax','numerical-jax']:
        try:
            import template_fitting_jax
        except ImportError as e:
            raise ImportError(
                "The experimental solver requires JAX to be installed. "
            ) from e


    data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
    sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
    if len(sigma) == 0:
        sigma = None

    radius_range = (radius_range[0]*u.Rsun.to(u.kpc),radius_range[1]*u.Rsun.to(u.kpc)) #Convert Rsun to kpc
    templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


    # Setup Av_lst
    #Av_lst = np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)
    Av_lst = safe_arange(Av_range[0], Av_range[1], Av_stepsize)
    if len(Av_lst) == 0:
        Av_lst = [0]
        


    # Normalize data to itself
    if rescale_data: 
        scale = np.max(data)
        data = data/scale
        if sigma is not None:
            sigma = np.array(sigma)/scale
    else:
        scale = 1

    


    # Normalize templates to themselves
    if rescale_templates:
        template_scales = np.array([np.max(t.flux) for t in templates])
        for i, t in enumerate(templates):
            t.flux = t.flux / template_scales[i]
            t.full_flux = t.full_flux / template_scales[i]


            # Adjust extinction_cache fluxes too (by the same scale for simplicity)
            keys = t.extinction_cache.keys()
            for key in keys:
               t.extinction_cache[key]['flux'] = t.extinction_cache[key]['flux'] / template_scales[i]


    
    # Calculate limits on amplitude
    # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
    if only_main_sequence:
        amp_min = []
        amp_max = []

        for i,t in enumerate(templates):
            radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS, main_sequence_model = main_sequence_model)

            radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
            amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
            amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )

        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)

    
    elif only_main_sequence_and_giants:
        amp_min = []
        amp_max = []

        for i,t in enumerate(templates):
            radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS, main_sequence_model = main_sequence_model)

            radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
            amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
            amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )

        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)
        

    else:
        amp_min = np.array([(np.min(radius_range)/np.max(dist_range))**2]*len(templates))
        amp_max = np.array([(np.max(radius_range)/np.min(dist_range))**2]*len(templates))
    


    # Calculate limits on amp based on mass_range and log_g
    if use_mass_constraint:
        amp_min_mass = []
        amp_max_mass = []
        for i,t in enumerate(templates):
            log_g = t.g
            # G = 6.6743*10**-11 #m3 kg-1 s-2
            # g = 10**(log_g) / 100 #cm s-2 to m s-2

            # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
            # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

            # CONST contains all scaling params above

            CONST = 1.3938323614347172e-17
            amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
            amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)

        amp_min_mass = np.array(amp_min_mass)
        amp_max_mass = np.array(amp_max_mass)



    if timeit:
        sw.log_process('Preprocess')
        sw.start()


    ### ======= ###
    ### Fitting ###
    ### ======= ###


    
    

    
    ### Step 1: Calculated fluxes to be tested
    ### Alternatively, keep track of only indices, and then in the step 2, actually save the fluxes with the different Avs. Might work.
    ### Also, if memory is an issue, you can put the linear bit back below actually and it should work. 

    flux_container = []
    indices = []
    Avs = []
    for i, template1 in enumerate(templates):
        for j, template2 in enumerate(templates):

            if i > j:
                continue

            for i_av, Av in enumerate(Av_lst):
                
                # if Av < Av_range[0]: #Sometimes np.arange causes weird floating point errors...
                #     continue
                # if Av > Av_range[1]:
                #     continue

                ### Get the flux and do the fit
                if Av == 0.0:
                    flux = [template1.flux, template2.flux]
                else:
                    flux1 = template1.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
                    flux2 = template2.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
                    
                    flux = [flux1['flux'],flux2['flux']]

            

                if i == j and (method == 'analytical' or method == 'analytical-batch'): 
                    # If analytical and i == j, then analytical method does not work because data is the same and it forms a singular 
                    # matrix. So we perturb one of the templates ever so slightly.
                    
                    #template2 = copy.deepcopy(template2) #copy so we do not overwrite
                    # smallest_non_zero = np.nanmin([x for x in template2.flux if x != 0])
                    # noise = np.random.uniform(-smallest_non_zero*1e-5, smallest_non_zero*1e-5, size = len(template2.flux))
                    # template2.flux += noise

                    flux = copy.deepcopy(flux) #copy so we do not overwrite
                    flux[1] = copy.deepcopy(flux[1]) # Disconnect the two elements
                    smallest_non_zero = np.nanmin([x for x in flux[1] if x != 0])
                    noise = np.random.uniform(-smallest_non_zero*1e-5, smallest_non_zero*1e-5, size = len(flux[1]))
                    flux[1] += noise

                flux_container.append(flux)
                indices.append([i,j])
                Avs.append(Av)



    ### Step 2: Solve 
    if method == 'analytical-batch':
        """
        template_batch: (M, N, K) array 
        M = number of independent fits to perform (e.g., 1,000,000)
        N = number of data points (e.g., 6 bands)
        K = number of templates in each fit (e.g., 1 for single star and 2 for binary)
        """
        # TODO: Can also add batches() here to allow for MP?

        #flux_container = np.swapaxes(flux_container, 1, 2) #Make sure flux_container has the right shape
        amps, sigma_amps, _ = fit_linear_model_analytically_batch(np.array(data), np.swapaxes(flux_container, 1, 2), sigma = sigma)


    elif method == 'analytical-batch-jax':
        """
        template_batch: (M, N, K) array 
        M = number of independent fits to perform (e.g., 1,000,000)
        N = number of data points (e.g., 6 bands)
        K = number of templates in each fit (e.g., 1 for single star and 2 for binary)
        """
        # TODO: Can also add batches() here to allow for MP?

        #flux_container = np.swapaxes(flux_container, 1, 2) #Make sure flux_container has the right shape
        amps, sigma_amps, _ = template_fitting_jax.fit_linear_model_analytically_batch_jax(np.array(data), np.swapaxes(flux_container, 1, 2), sigma = sigma)
        amps = np.asarray(amps).copy() #JAX arrays are immutable, so need to convert back to numpy here
        sigma_amps = np.asarray(sigma_amps).copy()
        

    else: 
        amps = []
        sigma_amps = []
        for i, flux in enumerate(flux_container): 
            if method == 'analytical':
                amp, sigma_amp, _ = fit_linear_model_analytically(data, flux, sigma = sigma)
            if method == 'analytical-jax':
                amp, sigma_amp, _ = template_fitting_jax.fit_linear_model_analytically_jax(data, flux, sigma = sigma)
            elif method == 'numerical':
                amp, sigma_amp, _ = fit_linear_model_numerically(data, flux, sigma = sigma)
            elif method == 'numerical-jax':
                amp, sigma_amp, _ = template_fitting_jax.fit_linear_model_numerically_jax(data, flux, sigma = sigma)


            amps.append(amp)
            sigma_amps.append(sigma_amp)



    if timeit:
        sw.log_process('Fit')
        sw.start()


    ### Step 3: rescale amps and adjust amplitudes constraints

    for i in range(len(amps)):
        amp = amps[i]
        sigma_amp = sigma_amps[i]
        idx_templates = indices[i]
    

            
        # Rescale the fit parameters back wrt data scale
        if rescale_data:
            amp = amp*scale
            if sigma is not None:
                sigma_amp = sigma_amp * scale

        
        # Rescale the fit parameters back wrt template scale
        if rescale_templates: 
            template_scale = template_scales[idx_templates]
            amp = amp/template_scale
            sigma_amp = sigma_amp/template_scale


        
        # Add amp limits
        # amp = np.clip(amp, amp_min[[i,j]], amp_max[[i,j]])# - the below is faster
        amps_min = amp_min[idx_templates]
        amps_max = amp_max[idx_templates]
        amp = np.maximum(amps_min, np.minimum(amp, amps_max))
        

        ## OLD MASS CONSTRAINT
        # if use_mass_constraint:
        #     kpc2Rsun = 44353565926.28097
            
        #     idxs = idx_templates#[i,j]
        #     #templates[idx_templates]
        #     #for k,t in enumerate([template1,template2]):
        #     for k,t in enumerate(templates[idx_templates]):
        #         log_g = t.g
        #         a = amp[k]
                
        #         pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(a*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(a*np.max(dist_range)**2)*kpc2Rsun, log_g)
    
        #         # G = 6.6743*10**-11 #m3 kg-1 s-2
        #         # g = 10**(log_g) / 100 #cm s-2 to m s-2
                
        #         # if pred_mass_min > np.max(mass_range):
        #         #     amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
        #         #     amp[k] = amp_new
    
        #         # if pred_mass_max < np.min(mass_range):
        #         #     amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
        #         #     amp[k] = amp_new


        #         if pred_mass_min > np.max(mass_range):
        #             amp[k] = amp_max_mass[idxs[k]]
    
        #         if pred_mass_max < np.min(mass_range):
        #             amp[k] = amp_min_mass[idxs[k]]

        #NEW MASS CONSTRAINT
        if use_mass_constraint:
            amps_min_mass = amp_min_mass[idx_templates]
            amps_max_mass = amp_max_mass[idx_templates]
            amp = np.maximum(amps_min_mass, np.minimum(amp, amps_max_mass))
        
            # if amp < amp_min_mass[idx_template]:
            #     amp = np.array([amp_min_mass[idx_template]])
            # elif amp > amp_max_mass[idx_template]:
            #     amp = np.array([amp_max_mass[idx_template]])
                



        if use_amplitude_ratio_constraint:
            """
            We know A1 = (R1/D1)**2 and A2 = (R2/D2)**2. However, D1=D2. So:
            A1/A2 = (R1/R2)**2. This is regardless of what distance the binary is at! We can use that as a limit too. 
            Take R1 as largest possible radius (R_max), and R2 as smallest possible radius (R_min), then (R_max/R_min)**2 is 
            the largest possible ratio for A1/A2 as well. If A1/A2 is larger than that, then something went wrong.

            TODO: currently I don't take only_MS into account!!!! Can change radius_range here to what templates are included

            NOTE: If you know the distance, this is essentially the same as the min/max amp that we calculate above, so a 
            bit redudant then. But if you do not know the distance, this can help!
            """

            assert len(amp) == 2, "Can only do amplitude_ratio_constraint for two amplitudes."

            max_amp_ratio = (np.max(radius_range) / np.min(radius_range))**2
            amp_ratio = np.max(amp) / np.min(amp)
            if amp_ratio > max_amp_ratio:
                print(amp)
                print(amp_ratio)
                print(max_amp_ratio)
                print('AMP RATIO')





        
        # Overwrite
        amps[i] = amp
        sigma_amps[i] = sigma_amp
    





    if timeit:
        sw.log_process('Amp_limits')
        sw.start()
        

    ### ============ ###
    ### Post process ###
    ### ============ ###


    
    flux_container = np.array(flux_container)   # Shape: (N, 2, 6)
    indices = np.array(indices)           # Shape: (N, 2)
    amps = np.array(amps) #Shape: (N, 2)
    
    # Scale data flux back wrt scale
    data = data*scale
    if sigma is not None:
        sigma = np.array(sigma)*scale
    
    

    # Scale template flux back wrt template_scales
    if rescale_templates: 
        for i, t in enumerate(templates):
            t.flux = t.flux * template_scales[i]

            # Adjust extinction_cache fluxes too (by the same scale for simplicity)
            keys = t.extinction_cache.keys()
            for key in keys:
                t.extinction_cache[key]['flux'] = np.array(t.extinction_cache[key]['flux']) * template_scales[i]




        # Rescale flux container too
        temp_template_scales = template_scales[indices] 
        flux_container = flux_container * temp_template_scales[:, :, np.newaxis]

        
        


    if timeit:
        sw.log_process('Rescale')
        sw.start()
        

    # Calculate error on the fit and find best fit
    results = (amps[:, np.newaxis, :] @ flux_container).squeeze()
    errors = calculate_error(data, results, sigma)
    
    
    #all_template_fluxes = np.array([t.flux for t in templates]) #doign this because it is fater than doing template.flux >100k times in this loop below.
    # errors = []
    # for i in range(len(amps)):
    #     amp = amps[i]
    #     idx = indices[i]
    #     template = flux_container[i]

    #     result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
    #     error = calculate_error(data, result, sigma = sigma)
    #     errors.append(error)


    
    if timeit:
        sw.log_process('Find_best_fit')
        sw.start()

    



    # if exclude_negative_amps:
    #     if any(np.array(amp) < 0):
    #         error = np.inf


    


    ### Return results

    if full_output:
        
        df_output = pd.DataFrame()
        df_output['error'] = errors
        
        idx_high = []
        for ts in np.array(templates)[indices]:
            if ts[0].T > ts[1].T:
                idx_high.append(0)
            else:
                idx_high.append(1)
        mask = np.array(idx_high) == 1
    
    
    
        df_output['idx1'] = [i[0] for i in indices]
        df_output['idx2'] = [i[1] for i in indices]
        df_output.loc[mask, ['idx1', 'idx2']] = df_output.loc[mask, ['idx2', 'idx1']].values
        
        df_output['amp1'] = [i[0] for i in amps]
        df_output['amp2'] = [i[1] for i in amps]
        df_output.loc[mask, ['amp1', 'amp2']] = df_output.loc[mask, ['amp2', 'amp1']].values
    
        df_output['sigma_amp1'] = [i[0] for i in sigma_amps]
        df_output['sigma_amp2'] = [i[1] for i in sigma_amps]
        df_output.loc[mask, ['sigma_amp1', 'sigma_amp2']] = df_output.loc[mask, ['sigma_amp2', 'sigma_amp1']].values
    
        df_output['Av'] = Avs
    
    
        
        all_temperatures = np.array([t.T for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
        df_output['temperature1'] = [all_temperatures[idx[0]] for idx in indices]
        df_output['temperature2'] = [all_temperatures[idx[1]] for idx in indices]
        df_output.loc[mask, ['temperature1', 'temperature2']] = df_output.loc[mask, ['temperature2', 'temperature1']].values
    
        all_Z = np.array([t.Z for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
        df_output['Z1'] = [all_Z[idx[0]] for idx in indices]
        df_output['Z2'] = [all_Z[idx[1]] for idx in indices]
        df_output.loc[mask, ['Z1', 'Z2']] = df_output.loc[mask, ['Z2', 'Z1']].values
    
        all_g = np.array([t.g for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
        df_output['g1'] = [all_g[idx[0]] for idx in indices]
        df_output['g2'] = [all_g[idx[1]] for idx in indices]
        df_output.loc[mask, ['g1', 'g2']] = df_output.loc[mask, ['g2', 'g1']].values
    
        df_output = df_output.loc[df_output.groupby(['idx1','idx2'])['error'].idxmin()].reset_index(drop=True)

        if timeit:
            sw.log_process('Output')
            sw.output_processes()
            sw.plot_processes()
    
    
        return df_output


    else:

        # Find best fit
        idx_best = np.argmin(errors)
        error = errors[idx_best]
        idx = indices[idx_best]
        amp = amps[idx_best]
        sigma_amp = sigma_amps[idx_best]
        Av = Avs[idx_best]
        
        best_templates = np.array(templates)[idx]
        if best_templates[1].T > best_templates[0].T:
            idx = idx[::-1]
            amp = amp[::-1]
            sigma_amp = sigma_amp[::-1]
            

        row = {'idx1':idx[0],'idx2':idx[1],'amp1':amp[0],'amp2':amp[1],'sigma_amp1':sigma_amp[0],'sigma_amp2':sigma_amp[1],'Av':Av,'error':error}
        row['idx1'] = int(row['idx1'])
        row['idx2'] = int(row['idx2'])


        if timeit:
            sw.log_process('Output')
            sw.output_processes()
            sw.plot_processes()
                    
        return row



fit_double_star = fit_binary #For backwards compatibility


def fit_internal(data, templates, throughput_curves = {}, rescale_data = False, rescale_templates = False, method = 'analytical-batch',
        dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0,np.inf), Av_range = (0,0), Av_stepsize = 0.1, 
        extinction_function = None, full_output = False, criteria = 'ftest', only_main_sequence = False, only_main_sequence_and_giants = False,
        delta_MS = 2, cache_new_extinction_values = True, use_mass_constraint = True,use_amplitude_ratio_constraint = False, main_sequence_model = None):
    '''
    Description
    -----------
    This function will fit the data in `data.flux' with two models i) a single star and ii) a binary system, using the templates provided in `templates'. Essentially, we are simply running `fit_single_star()' and `fit_double_star()', and seeing how well each model fits.



    Parameters
    ----------
    data : `utils.sed_template'
                A `utils.sed_template' that contains the flux (in data.flux) and flux uncertainties (in data.e_flux) that we want to fit

    templates : list of `utils.sed_template'
                We use the templates in this list to fit the data.

    throughput_curves : dict or None
                A dict of the throughput curve for each band. This has to have a particular format, please see the additional documentation on throughput curves. This is needed in case we need to calculate fluxes for Av values that were not precomputed.

    rescale_data : bool, default = False
                Whether to rescale the data to 1 by dividing it by its maximum value. This can be turned on to avoid numerical instabilities.

    rescale_templates: bool, default = False
                Whether to rescale all the templates to 1 by dividing each template by its own maximum value. This can be turned on to avoid numerical instabilities.

    method: string, default = 'analytical-batch'
                Which method to use to do the fit. Must be one of the following: 'analytical-batch-jax','analytical-batch','analytical-jax','analytical','numerical','mumerical-jax'. We STRONGLY suggest to keep this at the default value of 'analytical-batch' or 'analytical-batch-jax', as they are much faster. More information on the different fitting methods is given below.

    dist_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper distance limits that we will fit. The units must be in kpc. The lower limit must be specified first, then the upper limit.

    radius_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper radius limits that we will fit. The units must be in solar radii. The lower limit must be specified first, then the upper limit.

    mass_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper mass limits that we will fit. The units must be in solar masses. The lower limit must be specified first, then the upper limit. These mass limits will be used, together with the log(g) associated with each template, to calculate appropriate limits on the radius of the star. This will only work if templates have associated log(g) values accessible by doing: `template.g' and if use_mass_constraint = True (see below).

    Av_range : tuple, default = (0,0)
                A tuple to define the lower and upper Av limits that we will fit. The units must be in mag. The lower limit must be specified first, then the upper limit. If equal to (0,0), no extinction will be applied. See `extinction_function' on more details on how this is implemented. 

    Av_stepsize : float, default = 0.1
                The stepsize with which we will sample Av. The different Av values that are tested are calculated as follows: ''np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)''. Lower values means higher precision, at the cost of compute time. 

    extinction_function : dust_extinction model or None, default = None
                The user can specify a dust extinction model from the "dust_extinction" Python package. More information here: https://dust-extinction.readthedocs.io/en/latest/index.html. If no law is specified, we will use the Fitzpatrick 1999 law with Rv = 3.1. For example, you can specify as follows: 
                ```
                from dust_extinction.parameter_averages import F99
                extinction_function = F99(Rv = 3.1)
                ```

    full_output : bool, default = False
                Whether to return a dictionary with only the best-fit parameters of the best-fit combination of templates (when `full_output' == False) or a pandas.DataFrame with the best-fit parameters for all combinations of templates.

    criteria : string, default = 'ftest'
                If full_output == False, this argument will determine which test will be used to decide whether the data is best modelled as a single star or binary system. The possible tests are: 'bic','aic','sse','sse_ratio','ftest','ftest_value'.
    
    only_main_sequence : bool, default = False
                Whether to only fit stars around the main sequence. The main sequence is defined with `delta_MS' (see below).

    only_main_sequence_or_giants : bool, default = False
                Whether to only fit stars around the main sequence and above (i.e., all types of giants). The main sequence is defined with `delta_MS' (see below).

    delta_MS : float, default = 2
                This is used to specify what counts as main sequence. We first fit a line on the HR diagram through common main sequence stars. All stars that are `delta_MS' values in log-luminosity space away from this best-fit line are still considered main sequence. 

    cache_new_extinction_values : bool, default = True
                Whether to cache the fluxes for Av values that were not precomputed so that subsequent lookup will be faster. The downside of this is that the RAM will blow up if you're testing a lot of new Av values.

    use_mass_constraint : bool, default = True
                Whether to limit the sample based on mass_range. Added for debugging purposes.

    use_amplitude_ratio_constraint : bool, default = False
                Currently not implemented. 
    


    Returns
    -------

    output : int, float, or dict
                If `full_output' = False, the output will be 0 (single star) or 1 (binary system), as determined by the test defined in `criteria'. If `criteria' in ['sse_ratio','ftest_pvalue'], then it will be a float between 0 and 1. If `full_output' == True, then the output will be a dict that summarizes the output of all tests. The user can then later decide which test to use. 

    Example
    -------



    Notes
    -----




    '''

    assert criteria in ['bic','aic','sse','sse_ratio','ftest','ftest_value'],'criteria must be either bic, aic, sse, sse_ratio,ftest,ftest_value'

    data = copy.deepcopy(data) #To make sure we don't overwrite data
    templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 

    
    #if Av_range[1] == 0 and Av_range[0] == 0:
    if Av_range[1] - Av_range[0] == 0:
        n_param_bb = 1
        n_param_2bb = 2
    else:
        n_param_bb = 2
        n_param_2bb = 3
    
    ### ================ ###
    ### Try one template ###
    ### ================ ###

    
    output_bb = fit_single_star(data, templates, rescale_data = rescale_data, rescale_templates = rescale_templates, method = method, 
                                dist_range = dist_range, mass_range = mass_range,
                                radius_range = radius_range, only_main_sequence = only_main_sequence,
                                only_main_sequence_and_giants = only_main_sequence_and_giants, 
                                delta_MS = delta_MS, use_mass_constraint = use_mass_constraint,
                                Av_range = Av_range, Av_stepsize = Av_stepsize, extinction_function = extinction_function, 
                                throughput_curves = throughput_curves, cache_new_extinction_values = cache_new_extinction_values,
                                main_sequence_model = main_sequence_model)



    
    temps_bb = copy.deepcopy(templates[output_bb['idx']])
    t_bb = temps_bb.get_reddened_photometry(Av = output_bb['Av'], extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
    
    error_bb = calculate_error(output_bb['amp']*np.array(t_bb['flux']), data.flux, sigma = data.e_flux)
    error_bb_log = calculate_error(np.log10(output_bb['amp']*np.array(t_bb['flux'])), np.log10(data.flux), sigma = np.log10(data.e_flux))
        
    bic_bb = bic(error_bb, len(data.flux), n_param_bb)
    aic_bb = aic(error_bb, len(data.flux), n_param_bb)
    
    bic_bb_log = bic(error_bb_log, len(data.flux), n_param_bb)
    aic_bb_log = aic(error_bb_log, len(data.flux), n_param_bb)
    


    ### ================= ###
    ### Try two templates ###
    ### ================= ###

    output_2bb = fit_double_star(data, templates, rescale_data = rescale_data, 
                                  rescale_templates = rescale_templates, method = method,
                                  dist_range = dist_range, radius_range = radius_range, mass_range = mass_range,
                                  only_main_sequence = only_main_sequence, 
                                  only_main_sequence_and_giants = only_main_sequence_and_giants,
                                  use_amplitude_ratio_constraint = use_amplitude_ratio_constraint, 
                                  delta_MS = delta_MS, use_mass_constraint = use_mass_constraint,
                                  Av_range = Av_range, Av_stepsize = Av_stepsize, extinction_function = extinction_function, 
                                  throughput_curves = throughput_curves, cache_new_extinction_values = cache_new_extinction_values,
                                  main_sequence_model = main_sequence_model)

    temps_2bb_1 = copy.deepcopy(templates[output_2bb['idx1']])
    temps_2bb_2 = copy.deepcopy(templates[output_2bb['idx2']])
    t_2bb_1 = temps_2bb_1.get_reddened_photometry(Av = output_2bb['Av'], extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
    t_2bb_2 = temps_2bb_2.get_reddened_photometry(Av = output_2bb['Av'], extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
    

    error_2bb = calculate_error(output_2bb['amp1']*np.array(t_2bb_1['flux'])+output_2bb['amp2']*np.array(t_2bb_2['flux']), data.flux, sigma = data.e_flux)
    error_2bb_log = calculate_error(np.log10(output_2bb['amp1']*np.array(t_2bb_1['flux'])+output_2bb['amp2']*np.array(t_2bb_2['flux'])), np.log10(data.flux), sigma = np.log10(data.e_flux))
        
    
    bic_2bb = bic(error_2bb, len(data.flux), n_param_2bb)
    aic_2bb = aic(error_2bb, len(data.flux), n_param_2bb)

    bic_2bb_log = bic(error_2bb_log, len(data.flux), n_param_2bb)
    aic_2bb_log = aic(error_2bb_log, len(data.flux), n_param_2bb)



    ### ====== ###
    ### Return ###
    ### ====== ###

    if not full_output:
        if criteria == 'bic':
            if bic_bb <= bic_2bb:
                return 0 #So is single star
            else:
                return 1 #So is binary
    
        if criteria == 'aic':
            if aic_bb <= aic_2bb:
                return 0 #So is single star
            else:
                return 1 #So is binary
    
        if criteria == 'sse':
            if error_bb <= error_2bb:
                return 0 #So is single star
            else:
                return 1 #So is binary
    
        if criteria == 'sse_ratio':
            return error_2bb/error_bb
    
    
        if criteria == 'ftest':
            pval = f_test(error_bb,error_2bb,n_param_bb,n_param_2bb,len(data.flux))
            if pval < 0.05:
                return 1 #Binary
            else:
                return 0 #single star
    
        if criteria == 'ftest_value':
            pval = f_test(error_bb,error_2bb,n_param_bb,n_param_2bb,len(data.flux))
            return pval 



    elif full_output:
        row = {}
        row['amp_single'] = output_bb['amp']
        row['sigma_amp_single'] = output_bb['sigma_amp']
        row['Av_single'] = output_bb['Av']
        row['sse_single'] = error_bb
        row['bic_single'] = bic_bb
        row['aic_single'] = aic_bb

        row['amp_double_1'] = output_2bb['amp1']
        row['sigma_amp_double_1'] = output_2bb['sigma_amp1']
        row['amp_double_2'] = output_2bb['amp2']
        row['sigma_amp_double_2'] = output_2bb['sigma_amp2']
        row['Av_double'] = output_2bb['Av']
        row['sse_double'] = error_2bb
        row['bic_double'] = bic_2bb
        row['aic_double'] = aic_2bb

        row['f_test_pval'] = f_test(error_bb,error_2bb,n_param_bb,n_param_2bb,len(data.flux))
        row['sse_ratio'] = error_2bb/error_bb

        row['idx_single'] = output_bb['idx']
        row['idx_double_1'] = output_2bb['idx1']
        row['idx_double_2'] = output_2bb['idx2']

        row['T_single'] = temps_bb.T
        row['T_double_1'] = temps_2bb_1.T
        row['T_double_2'] = temps_2bb_2.T

        row['g_single'] = temps_bb.g
        row['g_double_1'] = temps_2bb_1.g
        row['g_double_2'] = temps_2bb_2.g

        row['Z_single'] = temps_bb.Z
        row['Z_double_1'] = temps_2bb_1.Z
        row['Z_double_2'] = temps_2bb_2.Z



        if bic_bb_log <= bic_2bb_log:
            row['bic_log'] = 0
        else:
            row['bic_log'] = 1

        if aic_bb_log <= aic_2bb_log:
            row['aic_log'] = 0
        else:
            row['aic_log'] = 1

        if error_bb_log <= error_2bb_log:
            row['sse_log'] = 0
        else:
            row['sse_log'] = 1

        if f_test(error_bb_log,error_2bb_log,n_param_bb,n_param_2bb,len(data.flux)) < 0.05:
            row['ftest_log'] = 1
        else:
            row['ftest_log'] = 0
            
        

        if bic_bb <= bic_2bb:
            row['bic'] = 0
        else:
            row['bic'] = 1

        if aic_bb <= aic_2bb:
            row['aic'] = 0
        else:
            row['aic'] = 1

        if error_bb <= error_2bb:
            row['sse'] = 0
        else:
            row['sse'] = 1

        if f_test(error_bb,error_2bb,n_param_bb,n_param_2bb,len(data.flux)) < 0.05:
            row['ftest'] = 1
        else:
            row['ftest'] = 0

        
    return row




def fit(data, templates, throughput_curves = {}, rescale_data = False, rescale_templates = False, method = 'analytical-batch',
        dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0,np.inf), Av_range = (0,0), Av_stepsize = 0.1, 
        extinction_function = None, full_output = False, n_jobs = 1, n_batch = None, criteria = 'ftest', only_main_sequence = False, only_main_sequence_and_giants = False,
        delta_MS = 2, cache_new_extinction_values = True, use_mass_constraint = True,use_amplitude_ratio_constraint = False, main_sequence_model = None):


    '''

    Description
    -----------
    This function will fit the data in `data.flux' with two models i) a single star and ii) a binary system, using the templates provided in `templates'. Essentially, we are simply running `fit_single_star()' and `fit_double_star()', and seeing how well each model fits. This works for either one data, or a list of data. 



    Parameters
    ----------
    data : `utils.sed_template' or list of `utils.sed_template'
                A `utils.sed_template' that contains the flux (in data.flux) and flux uncertainties (in data.e_flux) that we want to fit. Alternatively, it is possible to use a list of `utils.sed_template' instead, in which case fit() will fit each data independently. 

    templates : list of `utils.sed_template'
                We use the templates in this list to fit the data.

    throughput_curves : dict or None
                A dict of the throughput curve for each band. This has to have a particular format, please see the additional documentation on throughput curves. This is needed in case we need to calculate fluxes for Av values that were not precomputed.

    rescale_data : bool, default = False
                Whether to rescale the data to 1 by dividing it by its maximum value. This can be turned on to avoid numerical instabilities.

    rescale_templates: bool, default = False
                Whether to rescale all the templates to 1 by dividing each template by its own maximum value. This can be turned on to avoid numerical instabilities.

    method: string, default = 'analytical-batch'
                Which method to use to do the fit. Must be one of the following: 'analytical-batch-jax','analytical-batch','analytical-jax','analytical','numerical','numerical-jax'. We STRONGLY suggest to keep this at the default value of 'analytical-batch' or 'analytical-batch-jax', as they are much faster. More information on the different fitting methods is given below.

    dist_range : tuple or list of tuples, default = (0, np.inf)
                A tuple to define the lower and upper distance limits that we will fit. The units must be in kpc. The lower limit must be specified first, then the upper limit. If specifying multiple data, this can also be a list of tuples that will be used for each corresponding data. 

    radius_range : tuple or list of tuples, default = (0, np.inf)
                A tuple to define the lower and upper radius limits that we will fit. The units must be in solar radii. The lower limit must be specified first, then the upper limit. If specifying multiple data, this can also be a list of tuples that will be used for each corresponding data. 

    mass_range : tuple or list of tuples, default = (0, np.inf)
                A tuple to define the lower and upper mass limits that we will fit. The units must be in solar masses. The lower limit must be specified first, then the upper limit. These mass limits will be used, together with the log(g) associated with each template, to calculate appropriate limits on the radius of the star. This will only work if templates have associated log(g) values accessible by doing: `template.g' and if use_mass_constraint = True (see below). If specifying multiple data, this can also be a list of tuples that will be used for each corresponding data. 

    Av_range : tuple or list of tuples, default = (0,0)
                A tuple to define the lower and upper Av limits that we will fit. The units must be in mag. The lower limit must be specified first, then the upper limit. If equal to (0,0), no extinction will be applied. See `extinction_function' on more details on how this is implemented. If specifying multiple data, this can also be a list of tuples that will be used for each corresponding data. 

    Av_stepsize : float, default = 0.1
                The stepsize with which we will sample Av. The different Av values that are tested are calculated as follows: ''np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)''. Lower values means higher precision, at the cost of compute time. 

    extinction_function : dust_extinction model or None, default = None
                The user can specify a dust extinction model from the "dust_extinction" Python package. More information here: https://dust-extinction.readthedocs.io/en/latest/index.html. If no law is specified, we will use the Fitzpatrick 1999 law with Rv = 3.1. For example, you can specify as follows: 
                ```
                from dust_extinction.parameter_averages import F99
                extinction_function = F99(Rv = 3.1)
                ```

    full_output : bool, default = False
                Whether to return a dictionary with only the best-fit parameters of the best-fit combination of templates (when `full_output' == False) or a pandas.DataFrame with the best-fit parameters for all combinations of templates.

    n_jobs : int, default = 1
                This will control whether we parallize the fitting process over multiple cores. If `n_jobs' == 0 or 1, then we do not parallelize. If `n_jobs > 1', then we will parallelize the fitting over n_jobs cores. Since this is CPU intensive code, n_jobs should not be higher than the maximum number of cores you have available (ideally you should keep 1-2 cores free for other stuff). 

    n_batch : int, default = None
                To avoid running out of RAM while parallelizing, we will only setup multiple processes for `n_batch' datasets at the same time. E.g., if n_batch = np.inf, and we have len(data) = 1_000_000, then we will have to copy `templates' 1_000_000 times, which can quickly spiral out of control. To mediate this, we only prepare `n_batch' processes at the same time. E.g., if `n_batch' = 8 and `n_jobs' = 4, then we will only copy `templates' (and all other parameters) 8 times and send them to 4 cores. We wait until all 8 are finished, and then do the next 8. Note that setting n_batch < n_jobs does not make sense. If `n_batch' is not specified, we set n_batch = 10*n_jobs.

    criteria : string, default = 'ftest'
                If full_output == False, this argument will determine which test will be used to decide whether the data is best modelled as a single star or binary system. The possible tests are: 'bic','aic','sse','sse_ratio','ftest','ftest_value'.
    
    only_main_sequence : bool, default = False
                Whether to only fit stars around the main sequence. The main sequence is defined with `delta_MS' (see below).

    only_main_sequence_or_giants : bool, default = False
                Whether to only fit stars around the main sequence and above (i.e., all types of giants). The main sequence is defined with `delta_MS' (see below).

    delta_MS : float, default = 2
                This is used to specify what counts as main sequence. We first fit a line on the HR diagram through common main sequence stars. All stars that are `delta_MS' values in log-luminosity space away from this best-fit line are still considered main sequence. 

    cache_new_extinction_values : bool, default = True
                Whether to cache the fluxes for Av values that were not precomputed so that subsequent lookup will be faster. The downside of this is that the RAM will blow up if you're testing a lot of new Av values.

    use_mass_constraint : bool, default = True
                Whether to limit the sample based on mass_range. Added for debugging purposes.

    use_amplitude_ratio_constraint : bool, default = False
                Currently not implemented. 
    


    Returns
    -------

    output : int, float, dict, list, pandas.DataFrame()
                If only one dataset is being fitted, and `full_output' = False, the output will be 0 (single star) or 1 (binary system), as determined by the test defined in `criteria'. If `criteria' in ['sse_ratio','ftest_pvalue'], then it will be a float between 0 and 1. If `full_output' == True, then the output will be a dict that summarizes the output of all tests. The user can then later decide which test to use. If multiple datasets are being fitted (i.e., `data' is a list of `utils.sed_template'), then the output will be a list (if `full_output' = False) or pandas.DataFrame (if `full_output' = True)

    Example
    -------



    Notes
    -----

    --- A note on measurement uncertainties ---
    If your data templates contain uncertainties in self.e_flux, this function automatically extracts them and uses them in the least-squares fitting process.


    --- A note on parallelizing ---
    
    Note that parallelizing only really makes sense if fitting one dataset takes a while, as there is an overhead time cost to setting up the parallelization, which takes a few seconds. If each fit only takes ~0.01-0.1s, then I would recommend not parallelizing. Any situation where an individual fit takes >1 sec, then multiprocesssing makes sense. 

    Also, parallelizing needs more RAM. If you find yourself running out of memory, lower the `n_batch' parameter to n_batch = n_jobs. If problem persists, lower `n_jobs' to n_jobs = 1 (effectively disabling parallelizing). 

    
    --- Note to myself ---
    
    If I'm adding a new argument to fit_internal(), I need to also add it: i) in fit() ii) under single_fit iii) multiple_fits n_jobs <= 1, iv) multiple_fits n_jobs > 1 information_base and v) fit_wrapper

    

    
    
    '''
    ### Initialize

    if isinstance(data,sed_template):
        single_fit = True
        multiple_fits = False

    elif isinstance(data,list) or isinstance(data, np.ndarray):
        multiple_fits = True
        single_fit = False

    else:
        assert isinstance(data,sed_template) or isinstance(data,list) or isinstance(data, np.ndarray),'data must either be a sed_template, a list of sed_templates, or an array of sed_templates'
        


    ### ========== ###
    ### Single Fit ###
    ### ========== ###
    ### If single_fit, just do fit()

    if single_fit:
        result = fit_internal(data, templates, throughput_curves = throughput_curves, rescale_data = rescale_data, rescale_templates = rescale_templates, method = method,
                dist_range = dist_range, radius_range = radius_range, mass_range = mass_range, Av_range = Av_range, Av_stepsize = Av_stepsize, 
                extinction_function = extinction_function, full_output = full_output, criteria = criteria, only_main_sequence = only_main_sequence, only_main_sequence_and_giants = only_main_sequence_and_giants,
                delta_MS = delta_MS, cache_new_extinction_values = cache_new_extinction_values, use_mass_constraint = use_mass_constraint,use_amplitude_ratio_constraint = use_amplitude_ratio_constraint,
                main_sequence_model = main_sequence_model)

        return result #Which can either be full_output or false




    ### ============= ###
    ### Multiple Fits ###
    ### ============= ###
    ### If multiple_fits, what you do depends on n_jobs.
    
    if multiple_fits:

        # First, make sure all things that can change per data are in an iterable.
        if np.array(dist_range).shape == (2,):
            dist_range = np.array([list(dist_range)] * len(data))
        if np.array(radius_range).shape == (2,):
            radius_range = np.array([list(radius_range)] * len(data))
        if np.array(mass_range).shape == (2,):
            mass_range = np.array([list(mass_range)] * len(data))
        if np.array(Av_range).shape == (2,):
            Av_range = np.array([list(Av_range)] * len(data))





        if n_jobs in [0,1]: #If == 0 or 1, don't use multiprocessing, and just do a for-loop over everything
            results = []

            for i in range(len(data)):
                result = fit_internal(data[i], templates, throughput_curves = throughput_curves, rescale_data = rescale_data, rescale_templates = rescale_templates, method = method,
                        dist_range = dist_range[i], radius_range = radius_range[i], mass_range = mass_range[i], Av_range = Av_range[i], Av_stepsize = Av_stepsize, 
                        extinction_function = extinction_function, full_output = full_output, criteria = criteria, only_main_sequence = only_main_sequence, only_main_sequence_and_giants = only_main_sequence_and_giants,
                        delta_MS = delta_MS, cache_new_extinction_values = cache_new_extinction_values, use_mass_constraint = use_mass_constraint,use_amplitude_ratio_constraint = use_amplitude_ratio_constraint)

                results.append(result)


            if full_output: # Turn output to dataframe if full_output = True
                results = pd.DataFrame(results)
                
            return results





        else: #Turn multiprocessing on

            
            if n_batch is None:
                n_batch = n_jobs * 10

            if n_batch < n_jobs:
                n_batch = n_jobs
            

            batch_lst = batches(list(range(len(data))), n_batch)
            results = []

            for bi,batch in enumerate(batch_lst):

                information_base = { #This contains all static arguments
                    #'data' : data,
                    'templates' : templates,
                    'throughput_curves' : throughput_curves,
                    'rescale_data' : rescale_data,
                    'rescale_templates' : rescale_templates,
                    'method' : method,
                    #'dist_range' : dist_range,
                    #'radius_range' : radius_range,
                    #'mass_range' : mass_range,
                    #'Av_range' : Av_range, 
                    'Av_stepsize' : Av_stepsize,
                    'extinction_function' : extinction_function,
                    'full_output' : full_output,
                    'criteria' : criteria,
                    'only_main_sequence' : only_main_sequence,
                    'only_main_sequence_and_giants' : only_main_sequence_and_giants,
                    'delta_MS' : delta_MS,
                    'cache_new_extinction_values' : cache_new_extinction_values,
                    'use_mass_constraint' : use_mass_constraint,
                    'use_amplitude_ratio_constraint' : use_amplitude_ratio_constraint,
                    'main_sequence_model' : main_sequence_model
                }
                
                information = []
                for i in batch: #This contains all dynamic arguments
                    info = information_base.copy()
                    info['data'] = data[i]
                    info['dist_range'] = dist_range[i]
                    info['radius_range'] = radius_range[i]
                    info['mass_range'] = mass_range[i]
                    info['Av_range'] = Av_range[i]
                    information.append(info)
                    
    
                with multiprocessing.Pool(processes=n_jobs) as pool:
                    results_batch = pool.map(fit_wrapper, information)

                results=results+results_batch
            

            if full_output: # Turn output to dataframe if full_output = True
                results = pd.DataFrame(results)
                
            return results
            
            














def fit_wrapper(information):
    '''
    Used for multiprocessing. See fit()
    '''

    result = fit_internal(information['data'], information['templates'], throughput_curves = information['throughput_curves'], 
                 rescale_data = information['rescale_data'], rescale_templates = information['rescale_templates'], 
                 method = information['method'], dist_range = information['dist_range'], radius_range = information['radius_range'], 
                 mass_range = information['mass_range'], Av_range = information['Av_range'], Av_stepsize = information['Av_stepsize'], 
                 extinction_function = information['extinction_function'], full_output = information['full_output'], 
                 criteria = information['criteria'], only_main_sequence = information['only_main_sequence'], 
                 only_main_sequence_and_giants = information['only_main_sequence_and_giants'], delta_MS = information['delta_MS'], 
                 cache_new_extinction_values = information['cache_new_extinction_values'], 
                 use_mass_constraint = information['use_mass_constraint'], 
                 use_amplitude_ratio_constraint = information['use_amplitude_ratio_constraint'],
                main_sequence_model = information['main_sequence_model'])
    
    return result
    


    




### ======== ###
### Plotting ###
### ======== ###


def plot_fit_result(test, templates, output):
    '''
    Plot the output of fit_single_star() or fit_double_star()
    '''

    if isinstance(output, pd.DataFrame):
        idx_best = np.argmin(output['error'])
        output = output.iloc[idx_best].to_dict()
        if 'idx' in output.keys():
            output['idx'] = int(output['idx'])
        else:
            output['idx1'] = int(output['idx1'])
            output['idx2'] = int(output['idx2'])
            

        
        
    
    # Figure out whether binary or single
    is_binary = False
    if 'idx1' in output.keys():
        is_binary = True

    # Setup
    wavelengths = test.wavelength
    if is_binary:
        idx1 = output['idx1']
        idx2 = output['idx2']
        amp1 = output['amp1']
        amp2 = output['amp2']
        Av = output['Av']
        template1 = templates[idx1]
        template2 = templates[idx2]
        t_1 = template1.get_reddened_photometry(Av = Av)
        t_2 = template2.get_reddened_photometry(Av = Av)
        flux = np.array(t_1['flux']) * amp1 + np.array(t_2['flux']) * amp2
        
    else:
        idx = output['idx']
        amp = output['amp']
        Av = output['Av']
        template = templates[idx]
        t = template.get_reddened_photometry(Av = Av)
        flux = np.array(t['flux']) * amp

    



    # Plot
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), sharex=True,
                               gridspec_kw={'height_ratios': [4, 1]})
    

    plt.sca(ax1)
    plt.scatter(wavelengths, test.flux, label = 'Data')
    plt.scatter(wavelengths, flux, label = 'Best-fit model', marker = 'x')
    
    plt.legend()
    plt.ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
    plt.grid(True)


    plt.sca(ax2)
    residuals = test.flux - flux
    ymax = np.max(np.abs(residuals))*1.2
    plt.scatter(wavelengths, residuals)
    plt.axhline(0, c = 'k', ls = '--')
    plt.ylabel(r"Residuals")
    plt.xlabel("Wavelength (Å)")
    plt.grid(True)
    plt.ylim(-ymax,ymax)
    plt.tight_layout()
    plt.show()





def plot_fit_on_hr_diagram(output, df_info = None, dist_range = None, n_top = 2):
    '''

    '''
    
    output = copy.deepcopy(output) #So we do not overwrite anything
    df_info = copy.deepcopy(df_info) #So we do not overwrite anything
    
    #Account for dist_range
    if dist_range is None:
        distance = np.array([df_info['distance']])
    else:
        distance = np.array(dist_range)

    
    if 'idx' in output.keys(): #It is a single star output, rename columns to mimick binary
        output['idx1'] = output['idx']
        output['amp1'] = output['amp']
        output['sigma_amp1'] = output['sigma_amp']
        output['temperature1'] = output['temperature']

        output['idx2'] = np.nan
        output['amp2'] = np.nan
        output['sigma_amp2'] = np.nan
        output['temperature2'] = np.nan

    
    plt.figure(figsize = (5,5))

    # Plot MS
    plot_hr_diagram(ax=plt.gca(), MS_color = 'grey')


    if df_info is not None:
        # Plot true position
        df_info['lum1'] = stefan_boltzmann(temperature = df_info['temperature1'], radius = df_info['radius1'])
        if ~np.isnan(df_info['temperature2']):
            df_info['lum2'] = stefan_boltzmann(temperature = df_info['temperature2'], radius = df_info['radius2'])
        else:
            df_info['lum2'] = np.nan
    
        plt.scatter([df_info['temperature1'],df_info['temperature2']], [df_info['lum1'],df_info['lum2']], marker = 'x', color = 'C3', zorder = 10, label = 'True position', lw = 2)
        
        
    # Plot best-fit position
    idx_min = np.argmin(output['error'])
    for k in range(2): #Do this twice. Second time for single star will just be NaNs.
        rad = np.sqrt(distance**2 * output[f'amp{k+1}'][idx_min])
        temp = output[f'temperature{k+1}'][idx_min]
        if ~np.isnan(temp):
            lum = [stefan_boltzmann(temperature = temp, radius = r*u.kpc.to(u.Rsun)) for r in rad]
        else:
            lum = [np.nan for r in rad]
        
        plt.plot([temp]*len(lum), lum, c = 'C1', marker = '^', lw = 3, markersize = 6)
    plt.plot([],[], c = 'C1', marker = '^', label = 'Best-fit position', lw = 0, markersize = 6)
    



    
    # Plot top X best positions 
    max_error = sorted(output['error'])[n_top]
    idxs = np.where(output['error'] <= max_error)[0]
    for i in idxs:
        if i == idx_min:
            continue


        for k in range(2): #Do this twice. Second time for single star will just be NaNs.
            rad = np.sqrt(distance**2 * output[f'amp{k+1}'][i])
            temp = output[f'temperature{k+1}'][i]
            if ~np.isnan(temp):
                lum = [stefan_boltzmann(temperature = temp, radius = r*u.kpc.to(u.Rsun)) for r in rad]
            else:
                lum = [np.nan for r in rad]
            
            plt.plot([temp]*len(lum), lum, c = 'C0', zorder = -1, marker = 'o', lw = 3, markersize = 6)
    plt.plot([], [], c = 'C0', zorder = -1, label = 'Next best fits', marker = 'o', lw = 0, markersize = 6)
    
    plt.legend()
    plt.show()











### ============== ###
### OLD CODE BELOW ###
### ============== ###


# def fit_linear_model_analytically_v0(data, templates, sigma=None):
#     """
#     This solves a weighted linear least squares problem.
    
#     Fits data = a1 * template1 + a2 * template2 + ...
    
#     INPUTS
#         data:     (N,) array
#         templates: list of (N,) arrays OR a (N, K) 2D array
#         sigma:    (N,) array of errors on data (optional)

#     OUTPUTS
#         amps:     (K,) array of best-fit amplitudes
#         sigma_amps: (K,) array of uncertainties
#         cov:      (K, K) covariance matrix


#     As soon as the problem becomes:
#     data=a⋅f(b⋅x+c) + etc
#     this won't work, and better to use scipy.optimize.least_squares, I think.
#     """
#     y = np.asarray(data)
    
#     if isinstance(templates, list):
#         M = np.vstack([np.asarray(t) for t in templates]).T  # shape (N, K)
#     else:
#         M = np.asarray(templates)  # already shape (N, K)

#     if sigma is None:
#         W = np.eye(len(y))
#         MtW = M.T
#     else:                            
#         sigma = np.asarray(sigma)
#         W_diag = 1.0 / sigma**2
#         M = M * np.sqrt(W_diag[:, np.newaxis])
#         y = y * np.sqrt(W_diag)
#         MtW = M.T
    
#     MTM = MtW @ M    
#     MTY = MtW @ y
#     cov = np.linalg.inv(MTM)    
#     amps = cov @ MTY
#     sigma_amps = np.sqrt(np.diag(cov))    
    
#     return amps, sigma_amps, cov





# def fit_single_star_v0(data, templates, sigma = None, rescale_data = False, rescale_templates = False, exclude_negative_amps = True, method = 'analytical'):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.
#     '''
    
#     data = copy.deepcopy(data.flux) #To make sure we don't overwrite data
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    


#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = [np.max(t.flux) for t in templates] 
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]

    
#     errors = []
#     indices = []
#     amps = []
#     sigma_amps = []
    

#     for i, template in tqdm(enumerate(templates)):

#         # Do the actual fitting
#         testing = [template.flux]
        
#         if method == 'analytical':
#             amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#         elif method == 'numerical':
#             amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)


#         # Rescale the fit parameters back wrt data scale
#         if rescale_data:
#             amp = amp*scale
#             if sigma is not None: #Wait, I need to rescale back regardless, no? -> No I double checked, it is okay here.
#                 sigma_amp = sigma_amp * scale


#         # Calculate error on the fit
#         result = np.sum([a * np.array(t) for a, t in zip(amp, testing)],axis = 0) #Do we want error wrt total, or relative? 
#         if sigma is None:
#             error = calculate_error(data*scale, result, sigma = None)
#         else:
#             error = calculate_error(data*scale, result, sigma = sigma*scale)


#         if exclude_negative_amps:
#             if any(np.array(amp) < 0):
#                 error = np.inf

#         errors.append(error)
#         amps.append(amp)
#         sigma_amps.append(sigma_amp)
#         indices.append([i])



#     # Find best fit
#     idx_best = np.argmin(errors)
#     error = errors[idx_best]
#     amp = amps[idx_best]
#     sigma_amp = sigma_amps[idx_best]
#     idx = indices[idx_best]
    
#     best_templates = np.array(templates)[idx]

    

#     # Scale amps, sigma_amps, template flux back wrt template_scales
#     if rescale_templates: 
#         template_scale = np.array(template_scales)[idx]
        
#         amp = amp/template_scale
#         sigma_amp = sigma_amp/template_scale
#         for i, t in enumerate(best_templates):
#             t.flux = t.flux * template_scale[i]
            
#     return best_templates, amp, sigma_amp, error, idx




# def fit_single_star_v0(data, templates, sigma = None, rescale_data = False, rescale_templates = False, exclude_negative_amps = True, method = 'analytical'):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.
#     '''
    
#     data = copy.deepcopy(data.flux) #To make sure we don't overwrite data
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    


#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = [np.max(t.flux) for t in templates] 
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]

    
#     errors = []
#     indices = []
#     amps = []
#     sigma_amps = []
    

#     for i, template in tqdm(enumerate(templates)):

#         # Do the actual fitting
#         testing = [template.flux]
        
#         if method == 'analytical':
#             amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#         elif method == 'numerical':
#             amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)


#         # Rescale the fit parameters back wrt data scale
#         if rescale_data:
#             amp = amp*scale
#             if sigma is not None: #Wait, I need to rescale back regardless, no? -> No I double checked, it is okay here.
#                 sigma_amp = sigma_amp * scale


#         # Calculate error on the fit
#         result = np.sum([a * np.array(t) for a, t in zip(amp, testing)],axis = 0) #Do we want error wrt total, or relative? 
#         if sigma is None:
#             error = calculate_error(data*scale, result, sigma = None)
#         else:
#             error = calculate_error(data*scale, result, sigma = sigma*scale)


#         if exclude_negative_amps:
#             if any(np.array(amp) < 0):
#                 error = np.inf

#         errors.append(error)
#         amps.append(amp)
#         sigma_amps.append(sigma_amp)
#         indices.append([i])



#     # Find best fit
#     idx_best = np.argmin(errors)
#     error = errors[idx_best]
#     amp = amps[idx_best]
#     sigma_amp = sigma_amps[idx_best]
#     idx = indices[idx_best]
    
#     best_templates = np.array(templates)[idx]

    

#     # Scale amps, sigma_amps, template flux back wrt template_scales
#     if rescale_templates: 
#         template_scale = np.array(template_scales)[idx]
        
#         amp = amp/template_scale
#         sigma_amp = sigma_amp/template_scale
#         for i, t in enumerate(best_templates):
#             t.flux = t.flux * template_scale[i]
            
#     return best_templates, amp, sigma_amp, error, idx




# def fit_single_star_v0(template_data, templates, rescale_data = False, rescale_templates = False, method = 'analytical', 
#                     dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), only_main_sequence = False,
#                    only_main_sequence_and_giants = False, delta_MS = 2, use_mass_constraint = True, timeit = False, full_output = False):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.


#     use_mass_constraint:

#     We know:
#     g = G*M/R^2
#     And also:
#     A = R^2/D^2
#     Combined:
#     g = G * M / (A * D^2)
#     <=> M = g / G * (A * D^2)
#     Put in possible dist range, and you get possible mass range for that log_g. 

#     '''
    

#     if timeit:
#         sw=stopwatch()
#         sw.register_processes(['Preprocess','Fit','Amp_limits1','Amp_limits2','Amp_limits3','Amp_limits4','Rescale','Find_best_fit'])
#         sw.start()


     
#     data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
#     sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
#     if len(sigma) == 0:
#         sigma = None
        
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 




#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    
#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = np.array([np.max(t.flux) for t in templates]) 
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]


    
#     # Calculate limits on amplitude
#     # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
#     if only_main_sequence:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS)
            
#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
#             radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )


#     elif only_main_sequence_and_giants:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )

#     else:
#         amp_min = [(np.min(radius_range)/np.max(dist_range))**2]*len(templates) 
#         amp_max = [(np.max(radius_range)/np.min(dist_range))**2]*len(templates)




#     # Calculate limits on amp based on mass_range and log_g
#     if use_mass_constraint:
#         amp_min_mass = []
#         amp_max_mass = []
#         for i,t in enumerate(templates):
#             log_g = t.g
#             # G = 6.6743*10**-11 #m3 kg-1 s-2
#             # g = 10**(log_g) / 100 #cm s-2 to m s-2

#             # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
#             # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

#             # CONST contains all scaling params above

#             CONST = 1.3938323614347172e-17
#             amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
#             amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)


#     if timeit:
#         sw.log_process('Preprocess')
    
#     # Do the actual fitting
#     indices = []
#     amps = []
#     sigma_amps = []

#     for i, template in enumerate(templates):



#         if timeit:
#             sw.start()

        
#         flux = template.flux
#         testing = [flux]
        
#         if method == 'analytical':
#             amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#         elif method == 'numerical':
#             amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)


#         if timeit:
#             sw.log_process('Fit')
#             sw.start()


#         # Rescale the fit parameters back wrt data scale
#         if rescale_data:
#             amp = amp*scale
#             if sigma is not None: #Wait, I need to rescale back regardless, no? -> No I double checked, it is okay here.
#                 sigma_amp = sigma_amp * scale

#         # Rescale the fit parameters back wrt template scale
#         if rescale_templates: 
#             template_scale = template_scales[i]
#             amp = amp/template_scale
#             sigma_amp = sigma_amp/template_scale


#         if timeit:
#             sw.log_process('Amp_limits1')
#             sw.start()
        

#         # Add amp limits
#         if amp < amp_min[i]:
#             amp = np.array([amp_min[i]])
#         elif amp> amp_max[i]:
#             amp = np.array([amp_max[i]])


#         if timeit:
#             sw.log_process('Amp_limits2')
#             sw.start()
            

#         if use_mass_constraint:
#             kpc2Rsun = 44353565926.28097
#             log_g = template.g
#             pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(amp*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(amp*np.max(dist_range)**2)*kpc2Rsun, log_g)

#             # G = 6.6743*10**-11 #m3 kg-1 s-2
#             # g = 10**(log_g) / 100 #cm s-2 to m s-2
            
#             if pred_mass_min > np.max(mass_range):
#                 # amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
#                 # amp = np.array([amp_new])
#                 amp = np.array([amp_max_mass[i]])

#             if pred_mass_max < np.min(mass_range):
#                 # amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
#                 # amp = np.array([amp_new])
#                 amp = np.array([amp_min_mass[i]])
                


#         if timeit:
#             sw.log_process('Amp_limits3')
#             sw.start()
            

#         amps.append(amp)
#         sigma_amps.append(sigma_amp)
#         indices.append([i])


#         if timeit:
#             sw.log_process('Amp_limits4')
            


#     if timeit:
#         sw.start()
        

#     # Scale data flux back wrt scale
#     data = data*scale
#     if sigma is not None:
#         sigma = np.array(sigma)*scale
    
    

#     # Scale template flux back wrt template_scales
#     if rescale_templates: 
#         for i, t in enumerate(templates):
#             t.flux = t.flux * template_scales[i]


#     if timeit:
#         sw.log_process('Rescale')
#         sw.start()

    
#     # Calculate error on the fit and find best fit
#     errors = []
#     for i in range(len(amps)):
#         amp = [amps[i]]
#         idx = indices[i][0]
#         template = [templates[idx].flux]
        
    
#         result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
#         error = calculate_error(data, result, sigma = sigma)
#         errors.append(error)




#     if timeit:
#         sw.log_process('Find_best_fit')
#         sw.output_processes()
#         sw.plot_processes()



#     if full_output:
#         df_output = pd.DataFrame()
#         df_output['error'] = errors
#         df_output['amp'] = [i[0] for i in amps]
#         df_output['sigma_amp'] = [i[0] for i in sigma_amps]
#         df_output['idx'] = [i[0] for i in indices]

#         all_templates_in_order = templates[indices]
#         df_output['temperature'] = [t[0].T for t in all_templates_in_order]
#         df_output['g'] = [t[0].g for t in all_templates_in_order]
#         df_output['Z'] = [t[0].Z for t in all_templates_in_order]

#         return df_output



#     else:
#         # Find best fit
#         idx_best = np.argmin(errors)
#         error = errors[idx_best]
#         amp = amps[idx_best]
#         sigma_amp = sigma_amps[idx_best]
#         idx = indices[idx_best]
        
#         best_templates = np.array(templates)[idx]
                    
#         return best_templates, amp, sigma_amp, error, idx
        
        

        



# def fit_single_star_v0(template_data, templates, rescale_data = False, rescale_templates = False, method = 'analytical', 
#                     dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), only_main_sequence = False,
#                    only_main_sequence_and_giants = False, delta_MS = 2, use_mass_constraint = True, timeit = False, full_output = False,
#                    Av_range = (0,0), Av_stepsize = 0.1, extinction_function = None, throughput_curves = {},
#                    cache_new_extinction_values = True):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.


#     use_mass_constraint:

#     We know:
#     g = G*M/R^2
#     And also:
#     A = R^2/D^2
#     Combined:
#     g = G * M / (A * D^2)
#     <=> M = g / G * (A * D^2)
#     Put in possible dist range, and you get possible mass range for that log_g. 



#     NOTE: 
#     If testing different Avs, you need to specify: Av_range, throughput_curves, extinction_function (though we will just assume F99 if not specified)
#     '''
    

#     if timeit:
#         sw=stopwatch()
#         sw.register_processes(['Preprocess','Fit','Amp_limits1','Amp_limits2','Amp_limits3','Amp_limits4','Amp_limits5','Rescale','Find_best_fit'])
#         sw.start()


     
#     data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
#     sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
#     if len(sigma) == 0:
#         sigma = None
        
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Setup Av_lst
#     Av_lst = np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)
#     if len(Av_lst) == 0:
#         Av_lst = [0]


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    
#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = np.array([np.max(t.flux) for t in templates]) 
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]
#             t.full_flux = t.full_flux / template_scales[i]

#             # Adjust extinction_cache fluxes too (by the same scale for simplicity)
#             keys = t.extinction_cache.keys()
#             for key in keys:
#                t.extinction_cache[key]['flux'] = t.extinction_cache[key]['flux'] / template_scales[i]

    
#     # Calculate limits on amplitude
#     # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
#     if only_main_sequence:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS)
            
#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
#             radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )


#     elif only_main_sequence_and_giants:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )

#     else:
#         amp_min = [(np.min(radius_range)/np.max(dist_range))**2]*len(templates) 
#         amp_max = [(np.max(radius_range)/np.min(dist_range))**2]*len(templates)




#     # Calculate limits on amp based on mass_range and log_g
#     if use_mass_constraint:
#         amp_min_mass = []
#         amp_max_mass = []
#         for i,t in enumerate(templates):
#             log_g = t.g
#             # G = 6.6743*10**-11 #m3 kg-1 s-2
#             # g = 10**(log_g) / 100 #cm s-2 to m s-2

#             # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
#             # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

#             # CONST contains all scaling params above

#             CONST = 1.3938323614347172e-17
#             amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
#             amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)


#     if timeit:
#         sw.log_process('Preprocess')
    
#     # Do the actual fitting
#     indices = []
#     amps = []
#     sigma_amps = []
#     Avs = []
#     for i, template in enumerate(templates):


#         # Av_errors = [] #temporary error list of Avs to find optimal Av
        
#         for i_av, Av in enumerate(Av_lst):


#             if timeit:
#                 sw.start()


#             if Av < Av_range[0]: #Sometimes np.arange causes weird floating point errors...
#                 continue
#             if Av > Av_range[1]:
#                 continue


#             ### Get the flux and do the fit
#             if Av == 0.0:
#                 flux = template.flux
#             else:
#                 flux = template.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
#                 flux = flux['flux']
                
#             testing = [flux]
            
#             if method == 'analytical':
#                 amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#             elif method == 'numerical':
#                 amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)
    
    
#             if timeit:
#                 sw.log_process('Fit')
#                 sw.start()
    
    


#             ### Apply amplitude constraints
            
                
#             # Rescale the fit parameters back wrt data scale
#             if rescale_data:
#                 amp = amp*scale
#                 if sigma is not None: #Wait, I need to rescale back regardless, no? -> No I double checked, it is okay here.
#                     sigma_amp = sigma_amp * scale
    
#             # Rescale the fit parameters back wrt template scale
#             if rescale_templates: 
#                 template_scale = template_scales[i]
#                 amp = amp/template_scale
#                 sigma_amp = sigma_amp/template_scale
    
    
#             if timeit:
#                 sw.log_process('Amp_limits1')
#                 sw.start()
            
    
#             # Add amp limits
#             if amp < amp_min[i]:
#                 amp = np.array([amp_min[i]])
#             elif amp> amp_max[i]:
#                 amp = np.array([amp_max[i]])
    
    
#             if timeit:
#                 sw.log_process('Amp_limits2')
#                 sw.start()
                
    
#             if use_mass_constraint:
#                 kpc2Rsun = 44353565926.28097
#                 log_g = template.g
#                 pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(amp*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(amp*np.max(dist_range)**2)*kpc2Rsun, log_g)
    
#                 # G = 6.6743*10**-11 #m3 kg-1 s-2
#                 # g = 10**(log_g) / 100 #cm s-2 to m s-2
                
#                 if pred_mass_min > np.max(mass_range):
#                     # amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
#                     # amp = np.array([amp_new])
#                     amp = np.array([amp_max_mass[i]])
    
#                 if pred_mass_max < np.min(mass_range):
#                     # amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
#                     # amp = np.array([amp_new])
#                     amp = np.array([amp_min_mass[i]])
                    
    
    
#             if timeit:
#                 sw.log_process('Amp_limits3')
#                 sw.start()





#             ### Select best-fit Av 

#             # Rescale the fit parameters back into relative space to calculate best-fit Av. 
#             # Not the most efficient but also does not take up too much time
            
#             # Rescale the fit parameters back wrt data scale
#             # amp_ = copy.deepcopy(amp)
#             # #amp_ = amp
#             # if rescale_data:
#             #     amp_ = amp_*scale

#             # # Rescale the fit parameters back wrt template scale
#             # if rescale_templates: 
#             #     template_scale = template_scales[i]
#             #     amp_ = amp_ /template_scale

            
#             # ### Calculate Av_error
#             # result = np.sum([a * np.array(t) for a, t in zip(amp_, testing)],axis = 0)
#             # Av_error = calculate_error(data, result, sigma = sigma)
#             # Av_errors.append(Av_error)
            


#             if timeit:
#                 sw.log_process('Amp_limits4')
#                 sw.start()


#             ### Append
    
#             amps.append(amp)
#             sigma_amps.append(sigma_amp)
#             indices.append([i])
#             Avs.append(Av)
    
    
#             if timeit:
#                 sw.log_process('Amp_limits5')
            


#     if timeit:
#         sw.start()
        

#     # Scale data flux back wrt scale
#     data = data*scale
#     if sigma is not None:
#         sigma = np.array(sigma)*scale
    
    

#     # Scale template flux back wrt template_scales
#     if rescale_templates: 
#         for i, t in enumerate(templates):
#             t.flux = t.flux * template_scales[i]


#             # Adjust extinction_cache fluxes too (by the same scale for simplicity)
#             keys = t.extinction_cache.keys()
#             for key in keys:
#                t.extinction_cache[key]['flux'] = np.array(t.extinction_cache[key]['flux']) * template_scales[i]



#     if timeit:
#         sw.log_process('Rescale')
#         sw.start()

    
#     # Calculate error on the fit and find best fit
#     errors = []
#     for i in range(len(amps)):
#         amp = [amps[i]]
#         idx = indices[i][0]
        
#         Av = Avs[i]
#         if Av == 0.0:
#             flux = templates[idx].flux
#         else:
#             flux = templates[idx].get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves)
#             flux = flux['flux']
        
#         template = [flux]
        
    
#         result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
#         error = calculate_error(data, result, sigma = sigma)
#         errors.append(error)




#     if timeit:
#         sw.log_process('Find_best_fit')
#         sw.output_processes()
#         sw.plot_processes()



#     if full_output:
#         df_output = pd.DataFrame()
#         df_output['error'] = errors
#         df_output['amp'] = [i[0] for i in amps]
#         df_output['sigma_amp'] = [i[0] for i in sigma_amps]
#         df_output['Av'] = Avs
#         df_output['idx'] = [i[0] for i in indices]

#         all_templates_in_order = templates[indices]
#         df_output['temperature'] = [t[0].T for t in all_templates_in_order]
#         df_output['g'] = [t[0].g for t in all_templates_in_order]
#         df_output['Z'] = [t[0].Z for t in all_templates_in_order]



#         df_output = df_output.loc[df_output.groupby('idx')['error'].idxmin()].reset_index(drop=True)
#         return df_output



#     else:
#         # Find best fit
#         idx_best = np.argmin(errors)
#         error = errors[idx_best]
#         amp = amps[idx_best]
#         sigma_amp = sigma_amps[idx_best]
#         idx = indices[idx_best]
#         Av = Avs[idx_best]
        
#         best_templates = np.array(templates)[idx]
                    
#         return best_templates, amp, sigma_amp, Av, error, idx








# def fit_double_star_v0(data, templates, sigma = None, rescale_data = False, rescale_templates = False, exclude_negative_amps = True, method = 'analytical'):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.
#     '''
    
#     data = copy.deepcopy(data.flux) #To make sure we don't overwrite data
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    


#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = [np.max(t.flux) for t in templates] 
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]

    
#     errors = []
#     indices = []
#     amps = []
#     sigma_amps = []
    

#     for i, template1 in tqdm(enumerate(templates)):
#         for j, template2 in enumerate(templates):

#             if i >= j:
#                 continue

#             # Do the actual fitting
#             testing = [template1.flux, template2.flux]
#             if method == 'analytical':
#                 amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#             elif method == 'numerical':
#                 amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)
        
        
#             # Rescale the fit parameters back wrt data scale
#             if rescale_data:
#                 amp = amp*scale
#                 if sigma is not None:
#                     sigma_amp = sigma_amp * scale
    
    
#             # Calculate error on the fit
#             result = np.sum([a * np.array(t) for a, t in zip(amp, testing)],axis = 0) #Do we want error wrt total, or relative? 
#             if sigma is None:
#                 error = calculate_error(data*scale, result, sigma = None)
#             else:
#                 error = calculate_error(data*scale, result, sigma = sigma*scale)

            
#             if exclude_negative_amps:
#                 if any(np.array(amp) < 0):
#                     error = np.inf
    
#             errors.append(error)
#             amps.append(amp)
#             sigma_amps.append(sigma_amp)
#             indices.append([i,j])

#     # Find best fit
#     idx_best = np.argmin(errors)
#     error = errors[idx_best]
#     amp = amps[idx_best]
#     sigma_amp = sigma_amps[idx_best]
#     idx = indices[idx_best]
    
#     best_templates = np.array(templates)[idx]

    

#     # Scale amps, sigma_amps, template flux back wrt template_scales
#     if rescale_templates: 
#         template_scale = np.array(template_scales)[idx]
        
#         amp = amp/template_scale
#         sigma_amp = sigma_amp/template_scale
#         for i, t in enumerate(best_templates):
#             t.flux = t.flux * template_scale[i]
            
#     return best_templates, amp, sigma_amp, error, idx






# def fit_double_star_v0(template_data, templates, rescale_data = False, rescale_templates = False, method = 'analytical', 
#                     dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), only_main_sequence = False, 
#                     only_main_sequence_and_giants = False, use_amplitude_ratio_constraint = False,
#                     delta_MS = 2, use_mass_constraint = True, timeit = False, full_output = False):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.


#     Can probably speed this up. For some reason, the v0 version was much faster...
#     '''

#     if timeit:
#         sw=stopwatch()
#         sw.register_processes(['Preprocess','Fit','Amp_limits1','Amp_limits2','Amp_limits3','Amp_limits4','Amp_limits5','Amp_limits6','Rescale','Find_best_fit'])
#         sw.start()



    
#     data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
#     sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
#     if len(sigma) == 0:
#         sigma = None
    
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    


#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = np.array([np.max(t.flux) for t in templates])
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]


    
#     # Calculate limits on amplitude
#     # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
#     if only_main_sequence:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
#             radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )

    
#     elif only_main_sequence_and_giants:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )
        

#     else:
#         amp_min = np.array([(np.min(radius_range)/np.max(dist_range))**2]*len(templates))
#         amp_max = np.array([(np.max(radius_range)/np.min(dist_range))**2]*len(templates))
    


#     # Calculate limits on amp based on mass_range and log_g
#     if use_mass_constraint:
#         amp_min_mass = []
#         amp_max_mass = []
#         for i,t in enumerate(templates):
#             log_g = t.g
#             # G = 6.6743*10**-11 #m3 kg-1 s-2
#             # g = 10**(log_g) / 100 #cm s-2 to m s-2

#             # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
#             # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

#             # CONST contains all scaling params above

#             CONST = 1.3938323614347172e-17
#             amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
#             amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)



    

#     indices = []
#     amps = []
#     sigma_amps = []


#     if timeit:
#         sw.log_process('Preprocess')
    

#     for i, template1 in enumerate(templates):
#         for j, template2 in enumerate(templates):

#             if i > j:
#                 continue

#             if i == j and method == 'analytical': 
#                 # If analytical and i == j, then analytical method does not work because data is the same and it forms a singular 
#                 # matrix. So we perturb one of the templates ever so slightly.
                
#                 template2 = copy.deepcopy(template2) #copy so we do not overwrite
#                 smallest_non_zero = np.nanmin([x for x in template2.flux if x != 0])
#                 noise = np.random.uniform(-smallest_non_zero*1e-5, smallest_non_zero*1e-5, size = len(template2.flux))
#                 template2.flux += noise

            

#             if timeit:
#                 sw.start()
                

#             # Do the actual fitting
#             testing = [template1.flux, template2.flux]
#             if method == 'analytical':
#                 amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#             elif method == 'numerical':
#                 amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)


#             if timeit:
#                 sw.log_process('Fit')
#                 sw.start()
        
#             # Rescale the fit parameters back wrt data scale
#             if rescale_data:
#                 amp = amp*scale
#                 if sigma is not None:
#                     sigma_amp = sigma_amp * scale


#             if timeit:
#                 sw.log_process('Amp_limits1')
#                 sw.start()

            
#             # Rescale the fit parameters back wrt template scale
#             if rescale_templates: 
#                 template_scale = template_scales[[i,j]]
#                 amp = amp/template_scale
#                 sigma_amp = sigma_amp/template_scale

#             if timeit:
#                 sw.log_process('Amp_limits2')
#                 sw.start()


#             # Add amp limits
#             # amp = np.clip(amp, amp_min[[i,j]], amp_max[[i,j]])# - the below is faster
#             amps_min = amp_min[[i,j]]
#             amps_max = amp_max[[i,j]]
#             amp = np.maximum(amps_min, np.minimum(amp, amps_max))
            


            




#             if timeit:
#                 sw.log_process('Amp_limits3')
#                 sw.start()

#             if use_mass_constraint:
#                 kpc2Rsun = 44353565926.28097
                
#                 idxs = [i,j]
#                 for k,t in enumerate([template1,template2]):
#                     log_g = t.g
#                     a = amp[k]
                    
#                     pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(a*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(a*np.max(dist_range)**2)*kpc2Rsun, log_g)
        
#                     # G = 6.6743*10**-11 #m3 kg-1 s-2
#                     # g = 10**(log_g) / 100 #cm s-2 to m s-2
                    
#                     # if pred_mass_min > np.max(mass_range):
#                     #     amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
#                     #     amp[k] = amp_new
        
#                     # if pred_mass_max < np.min(mass_range):
#                     #     amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
#                     #     amp[k] = amp_new


#                     if pred_mass_min > np.max(mass_range):
#                         amp[k] = amp_max_mass[idxs[k]]
        
#                     if pred_mass_max < np.min(mass_range):
#                         amp[k] = amp_min_mass[idxs[k]]


#             if timeit:
#                 sw.log_process('Amp_limits4')
#                 sw.start()

#             if use_amplitude_ratio_constraint:
#                 """
#                 We know A1 = (R1/D1)**2 and A2 = (R2/D2)**2. However, D1=D2. So:
#                 A1/A2 = (R1/R2)**2. This is regardless of what distance the binary is at! We can use that as a limit too. 
#                 Take R1 as largest possible radius (R_max), and R2 as smallest possible radius (R_min), then (R_max/R_min)**2 is 
#                 the largest possible ratio for A1/A2 as well. If A1/A2 is larger than that, then something went wrong.

#                 TODO: currently I don't take only_MS into account!!!! Can change radius_range here to what templates are included

#                 NOTE: If you know the distance, this is essentially the same as the min/max amp that we calculate above, so a 
#                 bit redudant then. But if you do not know the distance, this can help!
#                 """

#                 assert len(amp) == 2, "Can only do amplitude_ratio_constraint for two amplitudes."

#                 max_amp_ratio = (np.max(radius_range) / np.min(radius_range))**2
#                 amp_ratio = np.max(amp) / np.min(amp)
#                 if amp_ratio > max_amp_ratio:
#                     print(amp)
#                     print(amp_ratio)
#                     print(max_amp_ratio)
#                     print('AMP RATIO')

            
#             if timeit:
#                 sw.log_process('Amp_limits5')
#                 sw.start()


            
#             amps.append(amp)
#             sigma_amps.append(sigma_amp)
#             indices.append([i,j])

#             if timeit:
#                 sw.log_process('Amp_limits6')
    



#     if timeit:
#         sw.start()
        
#     # Scale data flux back wrt scale
#     data = data*scale
#     if sigma is not None:
#         sigma = np.array(sigma)*scale
    
    

#     # Scale template flux back wrt template_scales
#     if rescale_templates: 
#         for i, t in enumerate(templates):
#             t.flux = t.flux * template_scales[i]



#     if timeit:
#         sw.log_process('Rescale')
#         sw.start()

#     # Calculate error on the fit and find best fit
#     all_template_fluxes = np.array([t.flux for t in templates]) #doign this because it is fater than doing template.flux >100k times in this loop below.
#     errors = []
#     for i in range(len(amps)):
#         amp = amps[i]
#         idx = indices[i]
#         template = all_template_fluxes[idx]
#         #template = [t.flux for t in np.array(templates)[idx]]


    
#         result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
#         error = calculate_error(data, result, sigma = sigma)
#         errors.append(error)


    
#     if timeit:
#         sw.log_process('Find_best_fit')
#         sw.output_processes()
#         sw.plot_processes()


#     # if exclude_negative_amps:
#     #     if any(np.array(amp) < 0):
#     #         error = np.inf


    
    

#     if full_output:
#         df_output = pd.DataFrame()
#         df_output['error'] = errors
        
#         idx_high = []
#         for ts in np.array(templates)[indices]:
#             if ts[0].T > ts[1].T:
#                 idx_high.append(0)
#             else:
#                 idx_high.append(1)
#         mask = np.array(idx_high) == 1



#         df_output['idx1'] = [i[0] for i in indices]
#         df_output['idx2'] = [i[1] for i in indices]
#         df_output.loc[mask, ['idx1', 'idx2']] = df_output.loc[mask, ['idx2', 'idx1']].values
        
#         df_output['amp1'] = [i[0] for i in amps]
#         df_output['amp2'] = [i[1] for i in amps]
#         df_output.loc[mask, ['amp1', 'amp2']] = df_output.loc[mask, ['amp2', 'amp1']].values

#         df_output['sigma_amp1'] = [i[0] for i in sigma_amps]
#         df_output['sigma_amp2'] = [i[1] for i in sigma_amps]
#         df_output.loc[mask, ['sigma_amp1', 'sigma_amp2']] = df_output.loc[mask, ['sigma_amp2', 'sigma_amp1']].values
    
#         all_temperatures = np.array([t.T for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['temperature1'] = [all_temperatures[idx[0]] for idx in indices]
#         df_output['temperature2'] = [all_temperatures[idx[1]] for idx in indices]
#         df_output.loc[mask, ['temperature1', 'temperature2']] = df_output.loc[mask, ['temperature2', 'temperature1']].values

#         all_Z = np.array([t.Z for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['Z1'] = [all_Z[idx[0]] for idx in indices]
#         df_output['Z2'] = [all_Z[idx[1]] for idx in indices]
#         df_output.loc[mask, ['Z1', 'Z2']] = df_output.loc[mask, ['Z2', 'Z1']].values

#         all_g = np.array([t.g for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['g1'] = [all_g[idx[0]] for idx in indices]
#         df_output['g2'] = [all_g[idx[1]] for idx in indices]
#         df_output.loc[mask, ['g1', 'g2']] = df_output.loc[mask, ['g2', 'g1']].values



#         return df_output


#     else:
#         # Find best fit
#         idx_best = np.argmin(errors)
#         error = errors[idx_best]
#         amp = amps[idx_best]
#         sigma_amp = sigma_amps[idx_best]
#         idx = indices[idx_best]
        
#         best_templates = np.array(templates)[idx]
                    
#         return best_templates, amp, sigma_amp, error, idx


#         # Find best fit
#         # idx_best = np.argmin(errors)
#         # error = errors[idx_best]
#         # amp = amps[idx_best]
#         # sigma_amp = sigma_amps[idx_best]
#         # idx = indices[idx_best]
        
#         # best_templates = np.array(templates)[idx]
    
    
        
#         # return best_templates, amp, sigma_amp, error, idx



# def fit_double_star_v0(template_data, templates, rescale_data = False, rescale_templates = False, method = 'analytical', 
#                     dist_range = (0, np.inf), radius_range = (0, np.inf), mass_range = (0, np.inf), only_main_sequence = False, 
#                     only_main_sequence_and_giants = False, use_amplitude_ratio_constraint = False,
#                     delta_MS = 2, use_mass_constraint = True, timeit = False, full_output = False,
#                     Av_range = (0,0), Av_stepsize = 0.1, extinction_function = None, throughput_curves = {}, 
#                     cache_new_extinction_values = True):

#     '''
#     NOTE: If no good estimate of sigma is provided, then the resultant sigma_amp values will also not be accurate. E.g. if 
#     sigma is underestimated, then sigma_amp will also be underestimated.
#     The amp values should still be good, though.

#     rescale_data and rescale_templates will rescale the data and templates, respectively, by dividing by their corresponding max value,
#     so that their flux is <1. This is for two reasons: numerical stability (otherwise the fluxes vary between 1e9-1e11 and higher)
#     and for speed purposes. We do it for the data and templates separately, as their respectives fluxes can also still vary wildly.
#     This way, all fluxes that I fit have their max at 1.


#     Can probably speed this up. For some reason, the v0 version was much faster...

#     NOTE: 
#     If testing different Avs, you need to specify: Av_range, throughput_curves, extinction_function (though we will just assume F99 if not specified)
    
#     '''

#     if timeit:
#         sw=stopwatch()
#         sw.register_processes(['Preprocess','Fit','Amp_limits1','Amp_limits2','Amp_limits3','Amp_limits4','Amp_limits5','Amp_limits6','Rescale','Find_best_fit'])
#         sw.start()



    
#     data = copy.deepcopy(template_data.flux) #To make sure we don't overwrite data
#     sigma = copy.deepcopy(template_data.e_flux) #To make sure we don't overwrite sigma
#     if len(sigma) == 0:
#         sigma = None
    
#     templates = copy.deepcopy(templates) #To make sure we don't overwrite templates 


#     # Setup Av_lst
#     Av_lst = np.arange(Av_range[0], Av_range[1]+Av_stepsize, Av_stepsize)
#     if len(Av_lst) == 0:
#         Av_lst = [0]
        


#     # Normalize data to itself
#     if rescale_data: 
#         scale = np.max(data)
#         data = data/scale
#         if sigma is not None:
#             sigma = np.array(sigma)/scale
#     else:
#         scale = 1

    


#     # Normalize templates to themselves
#     if rescale_templates:
#         template_scales = np.array([np.max(t.flux) for t in templates])
#         for i, t in enumerate(templates):
#             t.flux = t.flux / template_scales[i]
#             t.full_flux = t.full_flux / template_scales[i]


#             # Adjust extinction_cache fluxes too (by the same scale for simplicity)
#             keys = t.extinction_cache.keys()
#             for key in keys:
#                t.extinction_cache[key]['flux'] = t.extinction_cache[key]['flux'] / template_scales[i]


    
#     # Calculate limits on amplitude
#     # since A = (R/D)**2, then with limits on R and D, we can calculate limits on A.
#     if only_main_sequence:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, radius_ul = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
#             radius_ul = np.min([radius_ul, np.max(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (radius_ul*u.Rsun.to(u.kpc) / np.min(dist_range))**2 )

    
#     elif only_main_sequence_and_giants:
#         amp_min = []
#         amp_max = []

#         for i,t in enumerate(templates):
#             radius_ll, _, _ = ms_radius_limit(t.T, delta_log_L = delta_MS)

#             radius_ll = np.max([radius_ll, np.min(radius_range)*u.kpc.to(u.Rsun)])
            
#             amp_min.append( (radius_ll*u.Rsun.to(u.kpc) / np.max(dist_range))**2 )
#             amp_max.append( (np.max(radius_range) / np.min(dist_range))**2 )
        

#     else:
#         amp_min = np.array([(np.min(radius_range)/np.max(dist_range))**2]*len(templates))
#         amp_max = np.array([(np.max(radius_range)/np.min(dist_range))**2]*len(templates))
    


#     # Calculate limits on amp based on mass_range and log_g
#     if use_mass_constraint:
#         amp_min_mass = []
#         amp_max_mass = []
#         for i,t in enumerate(templates):
#             log_g = t.g
#             # G = 6.6743*10**-11 #m3 kg-1 s-2
#             # g = 10**(log_g) / 100 #cm s-2 to m s-2

#             # amp_max_mass.append(np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2)
#             # amp_min_mass.append(np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2)

#             # CONST contains all scaling params above

#             CONST = 1.3938323614347172e-17
#             amp_max_mass.append(np.max(mass_range) / 10**log_g / (np.min(dist_range))**2 * CONST)
#             amp_min_mass.append(np.min(mass_range) / 10**log_g / (np.max(dist_range))**2 * CONST)



    

#     indices = []
#     amps = []
#     sigma_amps = []
#     Avs = []


#     if timeit:
#         sw.log_process('Preprocess')
    

#     for i, template1 in enumerate(templates):
#         for j, template2 in enumerate(templates):

#             if i > j:
#                 continue


#             for i_av, Av in enumerate(Av_lst):
                
                
    
#                 if timeit:
#                     sw.start()


#                 if Av < Av_range[0]: #Sometimes np.arange causes weird floating point errors...
#                     continue
#                 if Av > Av_range[1]:
#                     continue



#                 ### Get the flux and do the fit
#                 if Av == 0.0:
#                     flux = [template1.flux, template2.flux]
#                 else:
#                     flux1 = template1.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
#                     flux2 = template2.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves, cache_result = cache_new_extinction_values)
                    
#                     flux = [flux1['flux'],flux2['flux']]

            

#                 if i == j and method == 'analytical': 
#                     # If analytical and i == j, then analytical method does not work because data is the same and it forms a singular 
#                     # matrix. So we perturb one of the templates ever so slightly.
                    
#                     #template2 = copy.deepcopy(template2) #copy so we do not overwrite
#                     # smallest_non_zero = np.nanmin([x for x in template2.flux if x != 0])
#                     # noise = np.random.uniform(-smallest_non_zero*1e-5, smallest_non_zero*1e-5, size = len(template2.flux))
#                     # template2.flux += noise

#                     flux = copy.deepcopy(flux) #copy so we do not overwrite
#                     flux[1] = copy.deepcopy(flux[1]) # Disconnect the two elements
#                     smallest_non_zero = np.nanmin([x for x in flux[1] if x != 0])
#                     noise = np.random.uniform(-smallest_non_zero*1e-5, smallest_non_zero*1e-5, size = len(flux[1]))
#                     flux[1] += noise


#                 testing = flux

#                 # Do the actual fitting
#                 if method == 'analytical':
#                     amp, sigma_amp, _ = fit_linear_model_analytically(data, testing, sigma = sigma)
#                 elif method == 'numerical':
#                     amp, sigma_amp, _ = fit_linear_model_numerically(data, testing, sigma = sigma)
    
    
#                 if timeit:
#                     sw.log_process('Fit')
#                     sw.start()
            
#                 # Rescale the fit parameters back wrt data scale
#                 if rescale_data:
#                     amp = amp*scale
#                     if sigma is not None:
#                         sigma_amp = sigma_amp * scale
    
    
#                 if timeit:
#                     sw.log_process('Amp_limits1')
#                     sw.start()
    
                
#                 # Rescale the fit parameters back wrt template scale
#                 if rescale_templates: 
#                     template_scale = template_scales[[i,j]]
#                     amp = amp/template_scale
#                     sigma_amp = sigma_amp/template_scale
    
#                 if timeit:
#                     sw.log_process('Amp_limits2')
#                     sw.start()
    
    
#                 # Add amp limits
#                 # amp = np.clip(amp, amp_min[[i,j]], amp_max[[i,j]])# - the below is faster
#                 amps_min = amp_min[[i,j]]
#                 amps_max = amp_max[[i,j]]
#                 amp = np.maximum(amps_min, np.minimum(amp, amps_max))
                
    
    
                
    
    
    
    
#                 if timeit:
#                     sw.log_process('Amp_limits3')
#                     sw.start()
    
#                 if use_mass_constraint:
#                     kpc2Rsun = 44353565926.28097
                    
#                     idxs = [i,j]
#                     for k,t in enumerate([template1,template2]):
#                         log_g = t.g
#                         a = amp[k]
                        
#                         pred_mass_min, pred_mass_max = calculate_mass(np.sqrt(a*np.min(dist_range)**2)*kpc2Rsun, log_g), calculate_mass(np.sqrt(a*np.max(dist_range)**2)*kpc2Rsun, log_g)
            
#                         # G = 6.6743*10**-11 #m3 kg-1 s-2
#                         # g = 10**(log_g) / 100 #cm s-2 to m s-2
                        
#                         # if pred_mass_min > np.max(mass_range):
#                         #     amp_new = np.max(mass_range)*u.Msun.to(u.kg) / g * G / (np.min(dist_range)*u.kpc.to(u.m))**2
#                         #     amp[k] = amp_new
            
#                         # if pred_mass_max < np.min(mass_range):
#                         #     amp_new = np.min(mass_range)*u.Msun.to(u.kg) / g * G / (np.max(dist_range)*u.kpc.to(u.m))**2
#                         #     amp[k] = amp_new
    
    
#                         if pred_mass_min > np.max(mass_range):
#                             amp[k] = amp_max_mass[idxs[k]]
            
#                         if pred_mass_max < np.min(mass_range):
#                             amp[k] = amp_min_mass[idxs[k]]
    
    
#                 if timeit:
#                     sw.log_process('Amp_limits4')
#                     sw.start()
    
#                 if use_amplitude_ratio_constraint:
#                     """
#                     We know A1 = (R1/D1)**2 and A2 = (R2/D2)**2. However, D1=D2. So:
#                     A1/A2 = (R1/R2)**2. This is regardless of what distance the binary is at! We can use that as a limit too. 
#                     Take R1 as largest possible radius (R_max), and R2 as smallest possible radius (R_min), then (R_max/R_min)**2 is 
#                     the largest possible ratio for A1/A2 as well. If A1/A2 is larger than that, then something went wrong.
    
#                     TODO: currently I don't take only_MS into account!!!! Can change radius_range here to what templates are included
    
#                     NOTE: If you know the distance, this is essentially the same as the min/max amp that we calculate above, so a 
#                     bit redudant then. But if you do not know the distance, this can help!
#                     """
    
#                     assert len(amp) == 2, "Can only do amplitude_ratio_constraint for two amplitudes."
    
#                     max_amp_ratio = (np.max(radius_range) / np.min(radius_range))**2
#                     amp_ratio = np.max(amp) / np.min(amp)
#                     if amp_ratio > max_amp_ratio:
#                         print(amp)
#                         print(amp_ratio)
#                         print(max_amp_ratio)
#                         print('AMP RATIO')
    
                
#                 if timeit:
#                     sw.log_process('Amp_limits5')
#                     sw.start()
    
    
                
#                 amps.append(amp)
#                 sigma_amps.append(sigma_amp)
#                 indices.append([i,j])
#                 Avs.append(Av)
    
#                 if timeit:
#                     sw.log_process('Amp_limits6')
    



#     if timeit:
#         sw.start()
        
#     # Scale data flux back wrt scale
#     data = data*scale
#     if sigma is not None:
#         sigma = np.array(sigma)*scale
    
    

#     # Scale template flux back wrt template_scales
#     if rescale_templates: 
#         for i, t in enumerate(templates):
#             t.flux = t.flux * template_scales[i]

#             # Adjust extinction_cache fluxes too (by the same scale for simplicity)
#             keys = t.extinction_cache.keys()
#             for key in keys:
#                 t.extinction_cache[key]['flux'] = np.array(t.extinction_cache[key]['flux']) * template_scales[i]



#     if timeit:
#         sw.log_process('Rescale')
#         sw.start()

#     # Calculate error on the fit and find best fit
#     all_template_fluxes = np.array([t.flux for t in templates]) #doign this because it is fater than doing template.flux >100k times in this loop below.
#     errors = []
#     for i in range(len(amps)):
#         amp = amps[i]
#         idx = indices[i]

        
#         Av = Avs[i]
#         if Av == 0.0:
#             template = all_template_fluxes[idx]
#             #template = [t.flux for t in np.array(templates)[idx]]
#         else:
#             t1 = templates[idx[0]]
#             t2 = templates[idx[1]]

#             flux1 = t1.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves)
#             flux2 = t2.get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves)

#             flux1 = flux1['flux']
#             flux2 = flux2['flux']
#             template = [flux1,flux2]

            
            
#             # flux = templates[idx].get_reddened_photometry(Av = Av, extinction_function = extinction_function, throughput_curves = throughput_curves)
#             # flux = flux['flux']
        
#         #template = flux
        


    
#         result = np.sum([a * np.array(t) for a, t in zip(amp, template)],axis = 0)
#         error = calculate_error(data, result, sigma = sigma)
#         errors.append(error)


    
#     if timeit:
#         sw.log_process('Find_best_fit')
#         sw.output_processes()
#         sw.plot_processes()


#     # if exclude_negative_amps:
#     #     if any(np.array(amp) < 0):
#     #         error = np.inf


    
    

#     if full_output:
#         df_output = pd.DataFrame()
#         df_output['error'] = errors
        
#         idx_high = []
#         for ts in np.array(templates)[indices]:
#             if ts[0].T > ts[1].T:
#                 idx_high.append(0)
#             else:
#                 idx_high.append(1)
#         mask = np.array(idx_high) == 1



#         df_output['idx1'] = [i[0] for i in indices]
#         df_output['idx2'] = [i[1] for i in indices]
#         df_output.loc[mask, ['idx1', 'idx2']] = df_output.loc[mask, ['idx2', 'idx1']].values
        
#         df_output['amp1'] = [i[0] for i in amps]
#         df_output['amp2'] = [i[1] for i in amps]
#         df_output.loc[mask, ['amp1', 'amp2']] = df_output.loc[mask, ['amp2', 'amp1']].values

#         df_output['sigma_amp1'] = [i[0] for i in sigma_amps]
#         df_output['sigma_amp2'] = [i[1] for i in sigma_amps]
#         df_output.loc[mask, ['sigma_amp1', 'sigma_amp2']] = df_output.loc[mask, ['sigma_amp2', 'sigma_amp1']].values

#         df_output['Av'] = Avs
        
#         all_temperatures = np.array([t.T for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['temperature1'] = [all_temperatures[idx[0]] for idx in indices]
#         df_output['temperature2'] = [all_temperatures[idx[1]] for idx in indices]
#         df_output.loc[mask, ['temperature1', 'temperature2']] = df_output.loc[mask, ['temperature2', 'temperature1']].values

#         all_Z = np.array([t.Z for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['Z1'] = [all_Z[idx[0]] for idx in indices]
#         df_output['Z2'] = [all_Z[idx[1]] for idx in indices]
#         df_output.loc[mask, ['Z1', 'Z2']] = df_output.loc[mask, ['Z2', 'Z1']].values

#         all_g = np.array([t.g for t in templates]) #doign this because it is fater than doing template.T >100k times in this loop below.
#         df_output['g1'] = [all_g[idx[0]] for idx in indices]
#         df_output['g2'] = [all_g[idx[1]] for idx in indices]
#         df_output.loc[mask, ['g1', 'g2']] = df_output.loc[mask, ['g2', 'g1']].values



#         df_output = df_output.loc[df_output.groupby(['idx1','idx2'])['error'].idxmin()].reset_index(drop=True)
#         return df_output


#     else:
#         # Find best fit
#         idx_best = np.argmin(errors)
#         error = errors[idx_best]
#         amp = amps[idx_best]
#         sigma_amp = sigma_amps[idx_best]
#         idx = indices[idx_best]
#         Av = Avs[idx_best]
        
#         best_templates = np.array(templates)[idx]
                    
#         return best_templates, amp, sigma_amp, Av, error, idx


#         # Find best fit
#         # idx_best = np.argmin(errors)
#         # error = errors[idx_best]
#         # amp = amps[idx_best]
#         # sigma_amp = sigma_amps[idx_best]
#         # idx = indices[idx_best]
        
#         # best_templates = np.array(templates)[idx]
    
    
        
#         # return best_templates, amp, sigma_amp, error, idx
    







        




