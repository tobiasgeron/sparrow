### Imports ###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import copy
from astropy import units as u
import math
from tqdm.notebook import tqdm
import time
from dust_extinction.parameter_averages import F99
from scipy.interpolate import interp1d
from scipy import stats
import pickle
import wget
import os

#import extinction



### =============== ###
### General physics ###
### =============== ###

def blackbody(wavelength, temperature, amplitude = 1, normalize = False):
    """
    Compute blackbody spectral radiance per unit wavelength.
    
    Parameters:
    - wavelength : float or ndarray
        Wavelength(s) in Ångströms.
    - temperature : float
        Temperature in Kelvin.
    - amplitude : float
        Float to scale y-axis with. Deals with sr
    
    Returns:
    - spectral_radiance : float or ndarray in erg/s/cm^2/Å



    We first calculate the typical B_lambda (spectral radiance) in cgs units, and then convert to observed flux. 

    F_lambda = pi * B_lambda (T) * (R/D)^2

    It is indeed pi and not 4*pi or 2*pi! Because we have spectral radiance (B_lambda), whereas the stellar templates are 
    in surface flux units. Anyway. Technically, we first do:

    M_lambda = pi * B_lambda,

    where M_lambda is spectral exitance, which is now in units of erg s-1 cm-2 A-1. And now to convert to observed flux on earth, multiple by (R/D)**2 dilution factor:

    F_lambda = M_lambda * (R/D)**2

    At the moment, the amplitude argument equals the (R/D)^2 term.

    
    """
    # Convert Ångström to meters
    wavelength_m = np.array(wavelength) * 1e-10

    # Physical constants in SI
    h = 6.62607015e-34     # Planck's constant (J·s)
    c = 2.99792458e8       # Speed of light (m/s)
    k = 1.380649e-23       # Boltzmann's constant (J/K)

    # Planck's Law in SI (W sr-1 m-3)
    exponent = (h * c) / (wavelength_m * k * temperature)
    B_lambda_SI = (2 * h * c**2) / (wavelength_m**5) / (np.exp(exponent) - 1)

    # Convert SI units to cgs: W sr-1 m-3 --> erg s-1 cm-2 A-1 sr-1
    # 1 W = 1e7 erg/s
    # 1 m2 = 1e4 cm2
    # 1 m = 1e10 A --> so per m is 1/1e10 per A
    conversion = 1e7 / 1e4 / 1e10  # = 1e-7

    B_lambda_cgs = B_lambda_SI * conversion 

    # Convert to flux observed on Earth, where amplitude = (R/D)^2
    F_lambda_cgs = B_lambda_cgs * np.pi * amplitude


    # Normalise
    if normalize:
        F_lambda_cgs = F_lambda_cgs / np.max(F_lambda_cgs)



    return F_lambda_cgs

create_blackbody = blackbody


def wiens_law(temp):
    '''
    Does Wien's Law. Input temp in Kelvin, output peak wavelength in Angstrom.
    '''

    return 2.9*10**7/temp



def stefan_boltzmann(temperature = np.nan, radius = np.nan, luminosity = np.nan):
    """
    Calculates the luminosity of a star based on its radius and temperature, using the stefan_boltzmann law:
    
    L = 4 pi R**2 sigma T**4

    This assumes that the star is an ideal blackbody though, so it is not perfect. 

    Temperature is in kelvin; radius in solar radii; luminosity in solar lum.

    This function needs two of these three as input, and will output the third.
    """
    params = [temperature,radius,luminosity]
    assert len(np.where(~np.isnan(params))[0]) == 2, "You should input two out of the following three: temperature, radius, luminosity"

    sigma = 5.670374419 * 10**-5 # erg cm-2 s-1 K-4


    #Looking for luminosity
    if np.isnan(luminosity):
        radius = radius*u.Rsun.to(u.cm) #cm
    
        L = 4 * np.pi * radius**2 * sigma  * temperature** 4 # erg s-1
        L = (L * u.erg / u.s).to(u.Lsun).value # Convert to Lsun

        return L

    # Looking for radius 
    if np.isnan(radius):
        luminosity = luminosity*u.Lsun.to(u.erg / u.s) #convert Lsun to erg s-1
        
        R = np.sqrt(luminosity/(4*np.pi * sigma * temperature**4)) #cm
        R = R*u.cm.to(u.Rsun) #Convert cm to Rsun
        return R


    # Looking for temperature
    if np.isnan(temperature):
        luminosity = luminosity*u.Lsun.to(u.erg / u.s) #convert Lsun to erg s-1
        radius = radius*u.Rsun.to(u.cm) #cm

        T = (luminosity / (4*np.pi * sigma * radius**2))**(1/4) #K
        return T
    


def magnitude_addition(mag1, mag2, sign = 1):
    '''
    Adds two magnitudes together.
    Can also do subtraction, in which case sign should be -1. In which case it will do: mag1 - mag2.

    Used to e.g. calculate the magnitude based on a difference magnitude and a reference magnitude.

    See https://en.wikipedia.org/wiki/Apparent_magnitude#Magnitude_addition
    '''

    return -2.5*np.log10(10**(-mag1*0.4) + sign * 10**(-mag2*0.4))



def mag_to_absmag(lst, d = None, z = None, cosmo = None):
    '''
    Convert between apparent and absolute magnitude.

    Basic formula of magnitudes:

    m1 - m2 = -2.5 log10(F1/F2)

    For absolute magnitudes, we want to measure the magnitude of something 
    that is exactly 10 pc away. So m2 becomes m(d=10pc), or simply M. 
    And F2 becomes F(d=10pc). Since F = L/(4pi d**2), and d for F(d=10pc)
    equals 10pc, we can rewrite the equation above:

    m - M = -2.5 log10((10 pc/d)**2)
    m - M = 5 log10(d / 10 pc)
    m - M = 5 log10(d) - 5
    M = m - 5 log10(d) + 5

    where d is the (luminosity) distance to the object in pc. Units are important!
    
    Instead of passing the distances, you can also pass redshifts and an astropy
    cosmology object. E.g.:
    
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    '''

    # Check if it is a single target or a list of targets. If single, wrap it in a list.
    single_target = False
    if type(lst) in [int,float, np.floating, np.float64,np.int64]:
        lst = [lst]
        d = [d]
        z = [z]
        single_target = True


    if cosmo == None:
        if len(d) == 1:
            d = [d[0]]*len(lst)
    if cosmo != None:
        if len(z) == 1:
            z = [z[0]]*len(lst)


    if cosmo != None:
        d = cosmo.luminosity_distance(np.array(z)).to('pc').value


            

    absmags = []
    for i,m in enumerate(lst): # Don't have to do the loop...
        M = m - 5 * np.log10(d[i]) + 5
        absmags.append(M)
    

    if single_target:
        return absmags[0]
    else:
        return np.array(absmags)

    

def absmag_to_mag(lst, d = None, z = None, cosmo = None):
    '''
    Convert between apparent and absolute magnitude.

    Basic formula of magnitudes:

    m1 - m2 = -2.5 log10(F1/F2)

    For absolute magnitudes, we want to measure the magnitude of something 
    that is exactly 10 pc away. So m2 becomes m(d=10pc), or simply M. 
    And F2 becomes F(d=10pc). Since F = L/(4pi d**2), and d for F(d=10pc)
    equals 10pc, we can rewrite the equation above:

    m - M = -2.5 log10((10 pc/d)**2)
    m - M = 5 log10(d / 10 pc)
    m - M = 5 log10(d) - 5
    M = m - 5 log10(d) + 5

    where d is the (luminosity) distance to the object in pc. Units are important!
    
    Instead of passing the distances, you can also pass redshifts and an astropy
    cosmology object. E.g.:
    
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    '''

    # Check if it is a single target or a list of targets. If single, wrap it in a list.
    single_target = False
    if type(lst) in [int,float, np.floating, np.float64,np.int64]:
        lst = [lst]
        d = [d]
        z = [z]
        single_target = True


    if cosmo == None:
        if len(d) == 1:
            d = [d[0]]*len(lst)
    if cosmo != None:
        if len(z) == 1:
            z = [z[0]]*len(lst)



    if cosmo != None:
        d = cosmo.luminosity_distance(z).to('pc').value
            

    mags = []
    for i,M in enumerate(lst): # Don't have to do the loop...
        m = M + 5 * np.log10(d[i]) - 5
        mags.append(m)
    

    if single_target:
        return mags[0]
    else:
        return np.array(mags)




def njy_to_mag(njy):
    '''
    Jansky is spectral flux density. 10**-26 watts m-2 Hz-1 or 10**-23 erg s-1 cm-2 Hz-1.
    magnitude is in AB mags

    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
    https://sites.astro.caltech.edu/~george/ay21/Fluxes%20and%20Magnitudes.pdf

    Can transform Jansky to AB mags by doing: S [uJy] = 10**((23.9 - AB)/2.5), but can also just use astropy. 
    Note that the zero-point magnitude here is 23.9, as it corresponds to AB mag system and flux in uJy in example. 
    For nJy, zero-point mag in AB system is 31.4 (as m = -2.5 log10(F) + m0 <=> m0 = m + 2.5 log10(F) => m0 = 0 + 2.5 log10(3631*10**9) = 31.4)

    This function assumes inputs to be in nanoJansky. Output in AB mag.
    '''

    single_target = False
    if isinstance(njy, (int, float, np.integer, np.floating)):
        njy = [njy]
        single_target = True

    
    mags = []
    for i in njy:
        if i < 0:
            mag = np.nan
        else:
            mag = (i*u.nJy).to(u.ABmag).value
        mags.append(mag)

    if single_target:
        return mags[0]
    else:
        return mags



def mag_to_njy(mags):
    '''
    Jansky is spectral flux density. 10**-26 watts m-2 Hz-1 or 10**-23 erg s-1 cm-2 Hz-1.
    magnitude is in AB mags

    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
    https://sites.astro.caltech.edu/~george/ay21/Fluxes%20and%20Magnitudes.pdf

    Can transform Jansky to AB mags by doing: S [uJy] = 10**((23.9 - AB)/2.5), but can also just use astropy. 
    Note that the zero-point magnitude here is 23.9, as it corresponds to AB mag system and flux in uJy in example. 
    For nJy, zero-point mag in AB system is 31.4 (as m = -2.5 log10(F) + m0 <=> m0 = m + 2.5 log10(F) => m0 = 0 + 2.5 log10(3631*10**9) = 31.4)

    This function assumes inputs to be in nanoJansky. Output in AB mag.
    '''

    single_target = False
    if isinstance(mags, (int, float, np.integer, np.floating)):
        mags = [mags]
        single_target = True

    
    njy = []
    for i in mags:
        temp = (i*u.ABmag).to(u.nJy).value
        njy.append(temp)

    if single_target:
        return njy[0]
    else:
        return njy





def ms_radius_limit(temperature, delta_log_L=2, output = 'radius', main_sequence_model = None):
    """
    This function gives the limit on radius a star can have, assuming it is a MS star, and given a temperature.
    It'll give rough radius ranges, using delta_log_L. 

    INPUT
    temperature in K
    delta_log_L: error in luminosity that you can to allow. In log(Lsun)
    main_sequence_model : a main_sequence_model class, which simply has temperature and luminosity variables

    OUTPUT
    radius_lower_limit, radius, radius_upper_limit
    
    """


    # See table on https://en.wikipedia.org/wiki/Main_sequence

    if main_sequence_model is None: 
        df_ms = pd.DataFrame(columns = ['luminosity','temperature','class'])
        df_ms['luminosity'] = [800_000, 180_000, 20_000, 800, 80, 20, 6, 2.5, 1.26, 1, 0.79, 0.40, 0.16, 0.072, 0.0027, 0.0004]
        df_ms['temperature'] = [50_000, 38_000, 30_000, 16_400, 10_800, 8_620, 7_240, 6_540, 5_920, 5_780, 5_610, 5_240, 4_410, 3_800, 3_120, 2_650]
        df_ms['class'] = ['O2','O6','B0','B5','A0','A5','F0','F5','G0','G2','G5','K0','K5','M0','M5','M8']
        
        df_ms['log_L'] = np.log10(df_ms['luminosity'])
        df_ms['log_T'] = np.log10(df_ms['temperature'])
        
        coefs, cov_matrix = np.polyfit(df_ms['log_T'], df_ms['log_L'], 1, cov=True)
        log_lum = coefs[0]*np.log10(temperature) + coefs[1]

    else:
        df_msmodel = pd.DataFrame(columns = ['luminosity','temperature'])
        df_msmodel['luminosity'] = main_sequence_model.luminosity
        df_msmodel['temperature'] = main_sequence_model.temperature

        df_msmodel['log_L'] = np.log10(df_msmodel['luminosity'])
        df_msmodel['log_T'] = np.log10(df_msmodel['temperature'])

        f_msmodel = interp1d(df_msmodel['log_T'], df_msmodel['log_L'], bounds_error=False, fill_value="extrapolate", kind = 'cubic')
        log_lum = f_msmodel(np.log10(temperature))

    lum = 10**log_lum
    lum_ul = 10**(log_lum+delta_log_L)
    lum_ll = 10**(log_lum-delta_log_L)


    if output == 'radius':
        radius = stefan_boltzmann(temperature = temperature, luminosity = lum)
        radius_ul = stefan_boltzmann(temperature = temperature, luminosity = lum_ul)
        radius_ll = stefan_boltzmann(temperature = temperature, luminosity = lum_ll)
        
    
        return radius_ll, radius, radius_ul

    elif output == 'luminosity':
        return lum_ll, lum, lum_ul



def plot_hr_diagram(temperatures = [], luminosities = [], label = '', color = 'C1', ax = None, plot_MS = True, MS_color = 'C0', plot_legend = False, marker = 'o'):
    """
    Plots the HR diagram, and adds your datapoints on top. 
    """

    
    df_ms = pd.DataFrame(columns = ['luminosity','temperature','class'])
    df_ms['luminosity'] = [800_000, 180_000, 20_000, 800, 80, 20, 6, 2.5, 1.26, 1, 0.79, 0.40, 0.16, 0.072, 0.0027, 0.0004]
    df_ms['temperature'] = [50_000, 38_000, 30_000, 16_400, 10_800, 8_620, 7_240, 6_540, 5_920, 5_780, 5_610, 5_240, 4_410, 3_800, 3_120, 2_650]
    df_ms['class'] = ['O2','O6','B0','B5','A0','A5','F0','F5','G0','G2','G5','K0','K5','M0','M5','M8']
    
    
    
    if ax == None:
        plt.figure(figsize = (6,6))
    else:
        plt.sca(ax)
        
    if plot_MS: 
        plt.scatter(df_ms['temperature'], df_ms['luminosity'], label = 'Main Sequence', color = MS_color)


    plt.scatter(temperatures, luminosities, label = label, color = color, marker = marker)

    
    plt.yscale('log')
    plt.xscale('log')
    
    plt.gca().invert_xaxis()
    
    plt.xlabel('Temperature [K]')
    plt.ylabel('Luminosity [Lsun]')

    
    plt.title('HR diagram')
    if plot_legend:
        plt.legend()

    if ax == None:
        plt.show()


def plot_kiel_diagram(temperatures = [], gs = [], Zs = [], label = '', color = 'C1', ax = None, plot_legend = False, marker = 'o'):
    """
    Plots the Kiel diagram, and adds your datapoints on top. 
    """
    
    if ax == None:
        plt.figure(figsize = (6,6))
    else:
        plt.sca(ax)

    if len(Zs) > 0:
        c = Zs
    else:
        c = color

    plt.scatter(temperatures, gs, label = label, c = c, marker = marker)

    plt.xscale('log')

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    
    plt.xlabel('Temperature [K]')
    plt.ylabel('log g [dex]')

    if len(Zs)>0:
        plt.colorbar(label = 'M/H [dex]')

    
    plt.title('Kiel diagram')
    if plot_legend:
        plt.legend()

    if ax == None:
        plt.show()



def determine_hr_type(temperature, luminosity, delta_log_L = 2, main_sequence_model = None):
    
    if not isinstance(temperature, list) and not isinstance(temperature, np.ndarray):
        temperature = [temperature]

    if not isinstance(luminosity, list) and not isinstance(luminosity, np.ndarray):
        luminosity = [luminosity]

    types = []
    
    for i in range(len(temperature)):
        temp = temperature[i]
        lum = luminosity[i]

        lum_ll, _, lum_ul = ms_radius_limit(temp, output='luminosity', delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)


        if lum_ll < lum and lum < lum_ul:
            types.append('MS')


        elif lum > lum_ul:
            types.append('G')


        else:
            types.append('Unclassified')


    return types        


### ====================================== ###
### Template atlas and templates functions ###
### ====================================== ###






class sed_template:
    
    def __init__(self, wavelength, flux, mag = [], T = None, Z = None, g = None, source = None):
        '''
        Description
        -----------

        Initializes the sed_template instance. A wavelength and flux array are needed. The input units are crucial: wavelength needs to be in Angstrom, while flux needs to be in erg s-1 cm-2 A-1.


        Parameters 
        ----------

        wavelength : numpy.array
            A numpy array with wavelength values. It needs to be in Angstrom
        
        flux : numpy.array
            A numpy array with flux values. It needs to be in erg s-1 cm-2 A-1.

        mag : numpy.array or list, default = []
            A numpy array or list of magnitude values. It needs to be in AB magnitudes
            
        T : int or float
            An estimate of the temperature. It needs to be in Kelvin

        Z : int or float
            An estimate of the metallicity. It needs to be in [M/H]

        g : int or float 
            An estimate of the surface gravity. It needs to be in log(cm -1)


        Returns
        -------

        
        Notes
        -----

        The input units are crucial: wavelength needs to be in Angstrom, while flux needs to be in erg s-1 cm-2 A-1.
        
        '''
        self.wavelength = wavelength
        self.flux = flux

        self.e_flux = []

        self.mag = mag
        self.e_mag = []

        self.T = T
        self.Z = Z
        self.g = g
        self.source = source

        self.bands = []

        self.full_flux = []
        self.full_mag = []
        self.full_wavelength = []


        self.extinction_cache = {}


    def trim(self, wavelength_min = 0, wavelength_max = np.inf, throughput_curves = None):
        '''
        Description
        -----------

        This will trim the template to remove unused wavelength ranges. Can either specify wavelength_min and wavelength_max, or specify the throughput curves used.


        Parameters 
        ----------

        wavelength_min : int or float, default = 0
            The minimum wavelength to keep. In Angstrom.

        wavelength_max : int or float, default = 0
            The maximum wavelength to keep. In Angstrom.

        throughput_curves : dict or None, default = None
            A dictionary of throughput curves. Instead of specifying the minimum and maximum wavelength, you can also simply specify the throughput curves that will be used. We will only keep wavelengths that are included in the throughput curves. 


        Returns
        -------

        

        
        '''

        assert wavelength_min != 0 or wavelength_max != np.inf or throughput_curves is not None, 'You need to specify either wavelength_min, wavelength_max, or throughput_curves'

        if wavelength_min != 0 or wavelength_max != np.inf:
            idxs = np.where( (np.array(self.wavelength) >= wavelength_min) & (np.array(self.wavelength) <= wavelength_max))[0]

        elif throughput_curves is not None:
            lambda_min = np.inf
            lambda_max = 0
            for band in throughput_curves.keys():
                wavelengths = throughput_curves[band][0]
                if np.min(wavelengths) < lambda_min:
                    lambda_min = np.min(wavelengths)
                if np.max(wavelengths) > lambda_max:
                    lambda_max = np.max(wavelengths)

            idxs = np.where( (np.array(self.wavelength) >= lambda_min) & (np.array(self.wavelength) <= lambda_max))[0]

        
        self.wavelength = np.array(self.wavelength)[idxs]
        self.flux = np.array(self.flux)[idxs]

        if len(self.e_flux) > 0:
            self.e_flux = np.array(self.e_flux)[idxs]

        if len(self.mag) > 0:
            self.mag = np.array(self.mag)[idxs]

        if len(self.e_mag) > 0:
            self.e_mag = np.array(self.e_mag)[idxs]

    



    def convert_flux_to_mag(self, throughput_curves = [], convert_errors = True):
        '''
        Converts self.flux to self.mag in AB magnitudes.

        convert_errors: bool, whether to automatically convert the errors as well.

        IMPORTANT: We implemented two techniques to convert, detailed below. The first technique is meant to convert SEDs from flux (in erg s-1 cm-2 A-1) to AB magnitudes,
        while the second technique is meant to convert band fluxes (in erg s-1 cm-2 A-1) to AB magnitudes. The first technique will also work for band fluxes, 
        but is not perfect and will introduce ~0.1-1% errors, as it is in principle only applicable to continuous wavelength axes (i.e., SEDs). However, the second techinique
        needs information about the bands and their throughput curves. 

        So: the first technique is used by default when calling the method as follows: 
        self.convert_flux_to_mag()

        while the second technique is used when specifying the throughput curves:
        self.convert_flux_to_mag(throughput_curves = throughput_curves)

        The second technique will also only work after you've actually converted SEDs to band fluxes with reduce_sed(), or if you've specified the bands
        in self.bands (this is done automatically when using reduce_sed()). 

        
        Details technique #1: 
        We convert flux density in units of erg s-1 cm-2 A-1 to units of erg s-1 cm-2 Hz-1 using:
        Fnu = Flam * wavelength**2/c
        and then apply default magnitude definition:
        mag = -2.5 * np.log10(Fnu/Fnu_ref)
        where: 
        Fnu_ref = 10**(48.60/-2.5) #erg s-1 cm-2 Hz-1

        Details technique #2: 
        Technique 1 does not work for band fluxes as the first step (converting to Fnu units) assumes continuous wavelength axis. Is approximate when using
        the band pivot wavelength, but not exact. Instead, we calculate the band zeropoint flux and effective bandwidth. We first convert flux density 
        in erg s-1 cm-2 A-1 to flux in erg s-1 cm-2:
        F = Flam * effective_bandwidth
        and then use the zeropoint flux to calculate the magnitude:
        mag = -2.5*np.log10(F/ZP)
        '''

        if len(throughput_curves) == 0: #Do it the default way, works for continuous SEDs.
            Fnu_ref = 10**(48.60/-2.5) #erg s-1 cm-2 Hz-1
            c = 299_792_458 #m/s 

            Fnu = np.array(self.flux) * (np.array(self.wavelength))**2/(c*u.m.to(u.Angstrom)) #erg s-1 cm-2 Hz-1. Need to convert c to A/s, because flux is in A-1! 
            self.mag = -2.5 * np.log10(Fnu/Fnu_ref) #alternatively: -2.5 * np.log10(Fnu) - 48.6


        else: # The more correct way, used for band fluxes. 
            assert len(self.bands) > 0, "To use more correct version to convert fluxes to magnitudes, we also need to know which datapoints in self.flux correspond to which bands. See comment below function for more info."
            zeropoints = calculate_zeropoint_flux_AB(throughput_curves)
            eff_bandwidths = calculate_effective_bandwidth(throughput_curves)


            zeropoints = [zeropoints[b] for b in self.bands] #assume bands are in the same order
            eff_bandwidths = [eff_bandwidths[b] for b in self.bands] #assume bands are in the same order

            #self.mag = -2.5* np.log10(np.array(self.flux) * (np.array(eff_bandwidths)) / zeropoints)
            self.mag = -2.5* np.log10(np.array(self.flux) / zeropoints) #CHANGED THIS


        # If there were errors in flux, convert to errors in mag
        if len(self.e_flux) > 0 and convert_errors:
            self.convert_fluxerr_to_magerr()
            



    def convert_mag_to_flux(self, throughput_curves = [], convert_errors = True):
        '''
        Converts self.mag (in AB magnitudes) to self.flux (in erg s-1 cm-2 A-1).

        convert_errors: bool, whether to automatically convert the errors from mag to flux as well.

        IMPORTANT: We implemented two techniques to convert, detailed below. The first technique is meant to convert SEDs from flux (in erg s-1 cm-2 A-1) to AB magnitudes,
        while the second technique is meant to convert band fluxes (in erg s-1 cm-2 A-1) to AB magnitudes. The first technique will also work for band fluxes, 
        but is not perfect and will introduce ~0.1-1% errors, as it is in principle only applicable to continuous wavelength axes (i.e., SEDs). However, the second techinique
        needs information about the bands and their throughput curves. 

        So: the first technique is used by default when calling the method as follows: 
        self.convert_flux_to_mag()

        while the second technique is used when specifying the throughput curves:
        self.convert_flux_to_mag(throughput_curves = throughput_curves)

        The second technique will also only work after you've actually converted SEDs to band fluxes with reduce_sed(), or if you've specified the bands
        in self.bands (this is done automatically when using reduce_sed()). 

        
        Details technique #1: 
        We convert flux density in units of erg s-1 cm-2 A-1 to units of erg s-1 cm-2 Hz-1 using:
        Fnu = Flam * wavelength**2/c
        and then apply default magnitude definition:
        mag = -2.5 * np.log10(Fnu/Fnu_ref)
        where: 
        Fnu_ref = 10**(48.60/-2.5) #erg s-1 cm-2 Hz-1

        Details technique #2: 
        Technique 1 does not work for band fluxes as the first step (converting to Fnu units) assumes continuous wavelength axis. Is approximate when using
        the band pivot wavelength, but not exact. Instead, we calculate the band zeropoint flux and effective bandwidth. We first convert flux density 
        in erg s-1 cm-2 A-1 to flux in erg s-1 cm-2:
        F = Flam * effective_bandwidth
        and then use the zeropoint flux to calculate the magnitude:
        mag = -2.5*np.log10(F/ZP)
        '''

        if len(throughput_curves) == 0: #Do it the default way, works for continuous SEDs.
            Fnu_ref = 10**(48.60/-2.5) #erg s-1 cm-2 Hz-1
            c = 299_792_458 #m/s 

            Fnu = 10**(np.array(self.mag)/-2.5) * Fnu_ref

            Flam = Fnu * (c*u.m.to(u.Angstrom))/ (np.array(self.wavelength))**2
            self.flux = Flam

        else:
            assert len(self.bands) > 0, "To use more correct version to convert magnitudes to fluxes, we also need to know which datapoints in self.flux correspond to which bands. See comment below function for more info."
            zeropoints = calculate_zeropoint_flux_AB(throughput_curves)
            eff_bandwidths = calculate_effective_bandwidth(throughput_curves)

            zeropoints = [zeropoints[b] for b in self.bands] #assume bands are in the same order
            eff_bandwidths = [eff_bandwidths[b] for b in self.bands] #assume bands are in the same order

            #self.flux = 10**(np.array(self.mag)/-2.5)*np.array(zeropoints) / np.array(eff_bandwidths)
            self.flux = 10**(np.array(self.mag)/-2.5)*np.array(zeropoints)#/ np.array(eff_bandwidths) # CHANGED THIS

        # If there were errors in mag, convert to errors in flux
        if len(self.e_mag) > 0 and convert_errors:
            self.convert_magerr_to_fluxerr()




    def convert_fluxerr_to_magerr(self):
            '''
            Converting errors in flux space to magnitude space needs a bit of special attention. It depends on the value of the flux (i.e., a sigma of 1 nJy will be a different sigma in magnitudes, depending on
            the amount of flux. Conversely, a sigma of 0.2 mag means something differently at 10 mag, compared to 30 mag). 
            This means that this function will only work well if the flux values themselves are updated!!



            Flux-to-Magnitude Formula:

            $m=−2.5log⁡10(F)$
            
            Differentiating the Magnitude Formula with respect to flux:
                
            $frac{dm}{dF} = -2.5 frac{1}{ln(10) F}$ 
            
            Apply error propagation:
            
            $sigma_{m} = |frac{dm}{dF}| sigma_{F}$
            
            $Rightarrow sigma_{m} = frac{2.5}{ln(10)} frac{1}{F} sigma_{F}$ 

            Shouldn't depend on unit of flux though, as long as flux and its error are in the same units. This is also how scisql_nanojanskyToAbMagSigma() works.
            '''


            self.e_mag = 2.5 / np.log(10) * 1/np.array(self.flux) * np.array(self.e_flux)


    def convert_magerr_to_fluxerr(self):
            '''
            Converting errors in flux space to magnitude space needs a bit of special attention. It depends on the value of the flux (i.e., a sigma of 1 nJy will be a different sigma in magnitudes, depending on
            the amount of flux. Conversely, a sigma of 0.2 mag means something differently at 10 mag, compared to 30 mag). 
            This means that this function will only work well if the flux values themselves are updated!!

            Flux-to-Magnitude Formula:

            $m=−2.5log⁡10(F)$
            
            Differentiating the Magnitude Formula with respect to flux:
                
            $frac{dm}{dF} = -2.5 frac{1}{ln(10) F}$ 
            
            Apply error propagation:
            
            $sigma_{m} = |frac{dm}{dF}| sigma_{F}$
            
            $Rightarrow sigma_{m} = frac{2.5}{ln(10)} frac{1}{F} sigma_{F}$ 

            Shouldn't depend on unit of flux though, as long as flux and its error are in the same units. This is also how scisql_nanojanskyToAbMagSigma() works.
            '''
                    
            self.e_flux = np.array(self.e_mag) * np.log(10) / 2.5 * np.array(self.flux)


    def redden(self, Av, extinction_function = None, inplace = True):
        '''
        Applies effects of extinction on template. This will work in flux space, and automatically adjusts the errors and magnitudes.
        But we need accurate fluxes!

        Av: float, the value of Av
        extinction_function: The function that converts Av into A_lambda. We use dust_extinction python package to handle this. The Fitzpatrick+99 model with Rv = 3.1 is used by default. 
        More info on dust_extinction here: https://dust-extinction.readthedocs.io/en/latest/index.html, where you can also browse more models. 

        NOTE: This overwrites the current flux and magnitude values! 
        '''

        if extinction_function is None:
            extinction_function = F99(Rv = 3.1)

        Alambda_Av = extinction_function(np.array(self.wavelength)/10000 * u.micron) #dust_extinction wants wavelengths in microns
        A_lambda = Alambda_Av * Av

        #A_lambda = extinction.fitzpatrick99(np.array(self.wavelength).astype('float'), Av, Rv)
        C = 10**(-0.4 * A_lambda)

        if inplace:
            self.flux = self.flux * C

            if len(self.e_flux) > 0:
                self.e_flux = self.e_flux * C

            self.convert_flux_to_mag() #Adjust mags (and mag errs) 

        else:
            return self.flux * C


    def deredden(self, Av, extinction_function = None, inplace = True):
        '''
        Applies effects of extinction on template. This will work in flux space, and automatically adjusts the errors and magnitudes.
        But we need accurate fluxes!

        Av: float, the value of Av
        extinction_function: The function that converts Av into A_lambda. We use dust_extinction python package to handle this. The Fitzpatrick+99 model with Rv = 3.1 is used by default. 
        More info on dust_extinction here: https://dust-extinction.readthedocs.io/en/latest/index.html, where you can also browse more models. 

        NOTE: This overwrites the current flux and magnitude values! 
        '''
        if extinction_function is None:
            extinction_function = F99(Rv = 3.1)

        Alambda_Av = extinction_function(np.array(self.wavelength)/10000 * u.micron) #dust_extinction wants wavelengths in microns
        A_lambda = Alambda_Av * Av

        #A_lambda = extinction.fitzpatrick99(np.array(self.wavelength).astype('float'), Av, Rv)
        C = 10**(-0.4 * A_lambda)

        if inplace:

            self.flux = self.flux / C

            if len(self.e_flux) > 0:
                self.e_flux = self.e_flux / C

            self.convert_flux_to_mag() #Adjust mags (and mag errs) 

        else:
            return self.flux / C




    def convolve(self, throughput_curves = {}, plot = False, inplace = True):
        temp_template = convolve_sed(self, throughput_curves = throughput_curves,  plot = plot)

        if inplace:
            self.full_flux = copy.deepcopy(self.flux)
            self.full_mag = copy.deepcopy(self.mag)
            self.full_wavelength = copy.deepcopy(self.wavelength)
        
            self.flux = temp_template.flux
            self.wavelength = temp_template.wavelength
            self.mag = temp_template.mag
            self.bands = temp_template.bands
        else:
            return temp_template




    def get_reddened_photometry(self, Av, extinction_function = None, throughput_curves = {}, force_recalculation = False, cache_result = True):
        """
        This method will redden and then convolve the spectrum. Equivalent to doing:
        self.redden() and then self.convolve()
        but crucially: this does NOT overwrite self.wavelength or self.flux. Instead, it saves the output in a cache, 
        so that the next time you query that Av, it'll be faster. Useful for precomputing all Avs that you might
        want to test later. 

        self.extinction_cache[Av] itself is a dictionary. With 'flux','e_flux' (maybe later add 'mag', and 'e_mag'?). 
        
        """
        # If already calculated before, then just retrieve it 
        if Av in self.extinction_cache and force_recalculation != True:
            return self.extinction_cache[Av]
        
        # Backup the current flux and wavelengths, and load in the full fluxes and wavelengths
        if len(self.full_flux) > 0:
            flux_backup = copy.deepcopy(self.flux)
            wavelength_backup = copy.deepcopy(self.wavelength)

            self.flux = self.full_flux
            self.wavelength = self.full_wavelength
        else:
            flux_backup = copy.deepcopy(self.flux)
            wavelength_backup = copy.deepcopy(self.wavelength)


        # Redden the template
        reddened_flux = self.redden(Av = Av, extinction_function = extinction_function, inplace = False)

        # Convolve the reddened template
        reddened_template = sed_template(wavelength = self.full_wavelength, flux = reddened_flux)
        reddened_convolved_template = convolve_sed(reddened_template, throughput_curves = throughput_curves,  plot = False)


        # Put original flux and wavelength back
        self.flux = flux_backup
        self.wavelength = wavelength_backup


        # Save the reddened convolved values
        new_row = {}
        new_row['wavelength'] = reddened_convolved_template.wavelength
        new_row['flux'] = reddened_convolved_template.flux
        if cache_result:
            self.extinction_cache[Av] = new_row

        return new_row
    


    def clear_extinction_cache(self):
        self.extinction_cache = {}


        



    def plot(self, figsize = (8,5), marker = 'auto', yaxis = 'flux', buffer = 0.05, xrange = (3000,10_000), yrange = (), color = 'C0', label = None, ax = None):
        '''
        Plots SED or band fluxes.

        figsize: tuple, size of figure.
        marker: str, whether to call plt.plot (when marker == 'line') or plt.scatter (when marker == 'point')
        yaxis: str, whether to plot fluxes (when yaxis == 'flux', default) or magnitudes (when yaxis == 'mag')


        '''

        # Plot
        assert marker in ['line','point','auto'], "marker has to be either 'line' or 'point'"
        assert yaxis in ['flux','mag','full_flux','full_mag'], "yaxis has to be either 'flux','mag','full_flux','full_mag'"

        if ax == None:
            standalone = True
        else:
            standalone = False



        if standalone:
            plt.figure(figsize=figsize)
        else:
            plt.sca(ax)

        if yaxis == 'flux':
            yaxis_data = np.array(self.flux)
            e_yaxis_data = np.array(self.e_flux)
            xaxis_data = np.array(self.wavelength)
        elif yaxis == 'mag':
            yaxis_data = np.array(self.mag)
            e_yaxis_data = np.array(self.e_mag)
            xaxis_data = np.array(self.wavelength)
        elif yaxis == 'full_flux':
            yaxis_data = np.array(self.full_flux)
            e_yaxis_data = np.array([])
            xaxis_data = np.array(self.full_wavelength)
        elif yaxis == 'full_mag':
            yaxis_data = np.array(self.full_mag)
            e_yaxis_data = np.array([])
            xaxis_data = np.array(self.full_wavelength)


        if len(e_yaxis_data) == 0:
            add_errors = False
        else:
            add_errors = True
            
            
        if marker == 'auto':
            if len(xaxis_data) < 20: #if less than 20 datapoints, probably band fluxes, which are better visualized with points. 
                marker = 'point'
            else: #otherwise, probably SED, better visualized with a line.
                marker = 'line'

        if marker == 'line':
            plt.plot(xaxis_data, yaxis_data, color = color, label = label)
        elif marker == 'point':
            if add_errors:
                plt.errorbar(xaxis_data, yaxis_data, e_yaxis_data, color = color, label = label, fmt='o')
            else:
                plt.scatter(xaxis_data, yaxis_data, color = color, label = label)

        plt.xlabel("Wavelength (Å)")
        if yaxis in ['flux','full_flux']:
            plt.ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
        elif yaxis in ['mag', 'full_mag']:
            plt.ylabel(r"AB magnitude")

        if len(xrange) == 0:
            xmin, xmax = np.nanmin(xaxis_data), np.nanmax(xaxis_data)
            delta_x = xmax - xmin
            plt.xlim(xmin - buffer*delta_x,xmax + buffer*delta_x)
        else:
            xmin, xmax = xrange
            plt.xlim(xmin,xmax)


        if len(yrange) == 0:
            idx = np.where( (np.array(xaxis_data) > xmin) & (np.array(xaxis_data) < xmax) ) [0]
            
            ymin, ymax = np.nanmin(yaxis_data[idx]), np.nanmax(yaxis_data[idx])
            delta_y = ymax - ymin
            plt.ylim(ymin - buffer*delta_y,ymax + buffer*delta_y)
        else:
            ymin, ymax = yrange
            plt.ylim(ymin, ymax)

        if yaxis in ['mag', 'full_mag']:
            plt.gca().invert_yaxis()

        plt.grid(True)
        plt.tight_layout()

        if standalone:
            plt.show()



# def find_template(templates, T_target = None, g_target = None, Z_target = None):
#     '''
#     Will go through template list, and return all templates that match all criteria. 
#     '''
#     idxs = []
#     for i in range(len(templates)):
    
#         if T_target != None:
#             if templates[i].T != T_target:
#                 continue
    
    
#         if g_target != None:
#             if templates[i].g != g_target:
#                 continue
    
    
#         if Z_target != None:
#             if templates[i].Z != Z_target:
#                 continue
    
    
#         idxs.append(i)

#     return idxs



def find_template(templates, T_target = None, T_min = -np.inf, T_max = np.inf, g_target = None, g_min = -np.inf, 
                  g_max = np.inf, Z_target = None, Z_min = -np.inf, Z_max = np.inf):
    '''
    Will go through template list, and return all indices of templates that match all criteria. 

    TODO: Not sure how this would behave if any of the variables do not exist. Probably error, but check
    '''
    idxs = []
    for i in range(len(templates)):
    
        if T_target != None:
            if templates[i].T != T_target:
                continue

        if templates[i].T < T_min:
            continue

        if templates[i].T > T_max:
            continue
            
    
        if g_target != None:
            if templates[i].g != g_target:
                continue

        if templates[i].g < g_min:
            continue

        if templates[i].g > g_max:
            continue
            
    
        if Z_target != None:
            if templates[i].Z != Z_target:
                continue

        if templates[i].Z < Z_min:
            continue

        if templates[i].Z > Z_max:
            continue
    
    
        idxs.append(i)

    return idxs





def calculate_mass(radius, log_g):
    """
    If a template has log_g, and we have radius constraints (e.g., from fitting), then we can calculate its mass.

    Units log_g are assumed to be cm s-2 (default for CK3 atlas)
    Units radius are assumed to be solar radius.
    
    g = G * M / Rad**2
    => M = g * Rad**2 / G
    """

    # G = 6.6743*10**-11 #m3 kg-1 s-2
    # g = 10**(log_g) / 100 #cm s-2 to m s-2
    # radius = radius * u.Rsun.to(u.m)
    # mass = g * radius**2 / G
    # return mass * u.kg.to(u.Msun)


    # Instead of defining constants every time and doing unit conversions, we speed this up by doign this once and saving that constant below.
    # COEF is basically:
    # Rsun2m = u.Rsun.to(u.m)
    # kg2Msun = u.kg.to(u.Msun)
    # G = 6.6743*10**-11
    # COEF = Rsun2m**2 * kg2Msun / G / 100
    # Which makes the below calculation 30-40% faster

    COEF = 3.64697152731123e-05
    return (10**log_g) * (radius**2) * COEF




def calculate_radius_ratio(amp1,amp2):
    """
    If
    data = a1 * template1 + a2 * template2
    
    where a1 = (r1/d1)**2 and a2 = (r2/d2)**2

    then, if we assume d1=d2 (which is the case for a binary)

    then: a1/a2 = R1**2/R2**2 <=> sqrt(a1/a2) = R1/R2
    """

    radius_ratio = np.sqrt(amp1/amp2)

    if radius_ratio > 1:
        radius_ratio = 1/radius_ratio

    return radius_ratio
    


def calculate_mass_ratio(amp1,amp2,log_g1,log_g2):
    """
    If
    data = a1 * template1 + a2 * template2
    
    where a1 = (r1/d1)**2 and a2 = (r2/d2)**2

    then, if we assume d1=d2 (which is the case for a binary)

    then: a1/a2 = R1**2/R2**2 <=> sqrt(a1/a2) = R1/R2

    If we then know the log_g of both, we can calculate the mass ratio easily too, substituting g = G * M / R**2 <=> R**2 = G*M/g

    so: a1/a2 = (G*M1/g1) / (G*M2/g2) = M1*g2/(M2*g1)
    <=> M1/M2 =a1*g1/(a2*g2)
    """
    mass_ratio = amp1*10**log_g1 / (amp2*10**log_g2)

    if mass_ratio > 1:
        mass_ratio = 1/mass_ratio
    return mass_ratio



def download_ck_atlas(data_dir):
    '''
    Downloads the entire ck atlas in its default file structure into data_dir.

    As of 17 June 2025, the entire atlas should not be larger than 50 MB.

    See https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas 
    for more details.

    From the above link: 
    "The names of the table files are given as cksmh_ttttt.fits, where "ck", for Castelli & Kurucz, are the first two letters 
    of the atlas; "smh" is the metal abundance of the model (mh) with its sign (s); and "ttttt" is the model's effective temperature,
    using four or five digits depending on the value. For example, a model with an effective temperature of 5000 K with [M/H]= -0.5 
    is indicated by ttttt= 5000, s= m, mh= 05 or ckm05_5000.fits. [...] Within each individual table file, each column is named "gyy", 
    where "yy" corresponds to 10*log g. For example, log g = +0.5 and log g = +4.0 models are located in columns named g05 and g40, 
    respectively."
    
    '''

    # First, download catalog
    url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/catalog.fits'
    out = data_dir + 'catalog.fits'
    if os.path.exists(out):
        os.remove(out)
    wget.download(url, out = out)
    
    # Open catalog
    catalog = fits.open(data_dir + 'catalog.fits')
    lst = catalog[1].data
    files = list(set([lst[i][1][:-5] for i in range(len(lst))]))

    # Loop over files, download them
    for f in tqdm(files):
        out = data_dir + f
    
        # Create subdirectory if it does not yet exists
        idx = out.rfind('/')
        outdir = out[:idx]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    
        # Download the actual file
        url = 'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/ck04models/' + f
        if os.path.exists(out): #Remove if it already exists
            os.remove(out)
        wget.download(url,out=out)
    


def load_ck_atlas(data_dir):

    templates = [] #This will store all the sed_templates

    # Open catalog
    catalog = fits.open(data_dir + 'catalog.fits')
    files = list(catalog[1].data['FILENAME'])
    files = list(sorted(set([file[:-5] for file in files])))
    
    for filename in tqdm(files):
        # Open file
        file = fits.open(data_dir + filename)
    
        
        T = file[0].header['TEFF']
        Z = file[0].header['log_Z']
        wave = file[1].data['WAVELENGTH']
    
        for col_name in file[1].columns.names:
            
            if col_name[0] != 'g': #We only loop over different g's
                continue
    
            g_str = col_name
            g = float(f'{g_str[1]}.{g_str[2]}')
            
            flux = file[1].data[g_str]

            if sum(flux) == 0: # We don't want to load empty files
                continue
            
            # Create sed_template
            st = sed_template(wave, flux, T=T, g=g, Z=Z, source = 'ck_atlas')
            templates.append(st)
    
        # Close file
        file.close()

    return templates



class main_sequence_model():
    """
    A model for what the user wants to specify as "the main sequence". This is basically just a container of information.
    It can be an isochrone, or a zero age main sequence. Note that, is using an isochrone, you need to make sure the temperatures 
    are monotonically increasing. I.e., make sure to remove rows that correspond to RSG phases etc. 

    Example use:
        ms_model = main_sequence_model(luminosity = list_of_luminosities, temperature = list_of_temperatures)

    luminosities should be in solar luminosities, and temperatures should be in Kelvin.
    """
    def __init__(self, luminosity, temperature):
        self.luminosity = luminosity
        self.temperature = temperature

        



def create_sample(templates, n_single = 0, n_double = 0, throughput_curves = None,               
                  dist_range = (0.001,50), radius_range = (0.02,2_000), mass_range = (0.08,300), 
                  T_range = (0,np.inf), g_range = (-np.inf,np.inf), 
                  Z_range = (-np.inf,np.inf),
                  lum_limit = 8_000_000,
                  Av_range = (0,0), extinction_function = None,
                  only_main_sequence = False, only_main_sequence_and_giants = True, delta_log_L = 2, 
                  add_noise = True, noise_model = 5, noise_bkg = 1e-19, noise_scale = 0.02, 
                  gammas = None, mag5sigmas = None, sigma_sys = None,
                  sample_radius_in_logspace = True, 
                  sample_distance_linearly = False, 
                  remove_negative_flux = True, zero_flux = 1e-100, 
                  use_mass_constraint = True, binary_identical_radii = False,
                  error_modifier_range = (1,1), main_sequence_model = None,
                  sample_cooler_companions = False, sample_radius_distribution = 'uniform'):
    
    '''
    Description
    -----------

    A massive function to generate a realistic-ish sample of single and binary systems based on the templates provided. This sample can later be used to e.g., test classification methods. 



    Parameters 
    ----------

    templates : list of `utils.sed_template'
                We take templates in this list to generate the final sample.
                
    n_single : int, default = 0
                The number of single stars to generate in the sample.

    n_double : int, default = 0
                The number of double/binary stars to generate in the sample.

    throughput_curves : dict or None
                A dict of the throughput curve for each band. This has to have a particular format, please see the additional documentation on throughput curves. 

    dist_range : tuple, default = (0.001,50)
                A tuple to define the lower and upper distance limits of targets in the final sample. The units must be in kpc. The lower limit must be specified first, then the upper limit.

    radius_range : tuple, default = (0.02,2_000)
                A tuple to define the lower and upper radius limits of targets in the final sample. The units must be in solar radii. The lower limit must be specified first, then the upper limit.

    mass_range : tuple, default = (0.08,300)
                A tuple to define the lower and upper mass limits of targets in the final sample. The units must be in solar masses. The lower limit must be specified first, then the upper limit. These mass limits will be used, together with the log(g) associated with each template, to calculate appropriate limits on the radius of the star. This will only work if templates have associated log(g) values accessible by doing: `template.g' and if use_mass_constraint = True (see below).
                
    T_range : tuple, default = (0, np.inf)
                A tuple to define the lower and upper temperature limits of targets in the final sample. The units must be in Kelvin. The lower limit must be specified first, then the upper limit. This limit will only work if templates have associated temperatures accessible by doing: `template.T'.

    g_range : tuple, default = (-np.inf, np.inf)
                A tuple to define the lower and upper log(g) limits of targets in the final sample. The units must be in log(cm s-2). The lower limit must be specified first, then the upper limit. This limit will only work if templates have associated log(g) values accessible by doing: `template.g'.
    
    Z_range : tuple, default = (-np.inf, np.inf)
                A tuple to define the lower and upper metallicity (Z) limits of targets in the final sample. The lower limit must be specified first, then the upper limit. This limit will only work if templates have associated Z values accessible by doing: `template.Z'.

    lum_limit: float, default = 8_000_000
                An upper luminosity limit. Specifying this can throw an error (i.e., ``Something went wrong with radius limit calculations'') if this is set too low for the specified radius_range or mass_range. 

    Av_range : tuple, default = (0,0)
                A tuple to define the lower and upper Av limits of targets in the final sample. The units must be in mag. The lower limit must be specified first, then the upper limit. If equal to (0,0), no extinction will be applied. See `extinction_function' on more details on how this is implemented. 

    extinction_function : dust_extinction model or None, default = None
                The user can specify a dust extinction model from the "dust_extinction" Python package. More information here: https://dust-extinction.readthedocs.io/en/latest/index.html. If no law is specified, we will use the Fitzpatrick 1999 law with Rv = 3.1. For example, you can specify as follows: 
                ```
                from dust_extinction.parameter_averages import F99
                extinction_function = F99(Rv = 3.1)
                ```

    only_main_sequence : bool, default = False
                Whether to only sample stars around the main sequence. The main sequence is defined with `delta_log_L' (see below).

    only_main_sequence_or_giants : bool, default = True
                Whether to only sample stars around the main sequence and above (i.e., all types of giants). The main sequence is defined with `delta_log_L' (see below).

    delta_log_L : float, default = 2
                This is used to specify what counts as main sequence. We first fit a line on the HR diagram through common main sequence stars. All stars that are `delta_log_L' values in log-luminosity space away from this best-fit line are still considered main sequence. 

    add_noise : bool, default = True
                Whether to add noise to the sample. How noise is added is handled by `noise_model', see below. 

    noise_model : int, default = 5
                We implemented different models for adding noise, which are detailed below. We strongly recommend to keep this to the default value.
                1 - Add `noise_bkg' to the templates in flux space
                2 - Add `noise_bkg' to the templates in magnitude space
                3 - Add `noise_scale'*flux to the templates in flux space
                4 - Add `noise_bkg' + `noise_scale'*flux to the templates in flux space
                5 - Add noise to the templates according to Eq 5 of arXiv:0805.2366. We need `gammas' and `mag5sigmas' for all bands used, see below. If none are given, load the default Rubin LSST values.

    noise_bkg : float, default = 1e-19
                One of the parameters used to add noise to templates. See `noise_model'. 

    noise_scale : float, default = 0.02
                One of the parameters used to add noise to templates. See `noise_model'. 

    gammas : dict or None, default = None
                One of the parameters used to add noise to templates. See `noise_model'. A dictionary of the gamma for each band used to create the sample. Only used in noise model #5. If nothing is specified, we use the default Rubin LSST values taken from https://arxiv.org/abs/0805.2366. 

    mag5sigmas : dict or None, default = None
                One of the parameters used to add noise to templates. See `noise_model'. A dictionary of the 5 sigma limiting magnitudes for each band used to create the sample. Only used in noise model #5. If nothing is specified, we use the default Rubin LSST values taken from https://rubinobservatory.org/for-scientists/rubin-101/key-numbers. 

    sigma_sys : dict or None, default = None
                One of the parameters used to add noise to templates. See `noise_model'. A dictionary of the sigma sys (systematic photometric error) for each band used to create the sample. Only used in noise model #5. If nothing is specified, we use the default Rubin LSST values and equal it to 0.005 mag for all bands.

    sample_radius_in_logspace : bool, default = True
                Whether to sample radius in log-space. This will ensure targets are evenly spaced out on the HR diagram, as the HR diagram is in log-space. If False, targets will tend to cluster in the top half of the HR diagram. We advise to keep this to True. 

    sample_distance_linearly : bool, default = False
                Whether to sample distance linearly. If false, the distances will be sampled ~dist**3 to account for larger increase in spherical volume as distance increases. We advise to keep this to False.

    remove_negative_flux : bool, default = True
                If True, negative fluxes will be set to the value specified in `zero_flux'. Useful for when planning to convert to magnitudes later.

    zero_flux : float, default = 1e-100
                A very very small non-zero number that essentially means no flux. When `remove_negative_flux' = True, this value helps to make sure that magnitudes don't blow up to np.inf.

    use_mass_constraint : bool, default = True
                Whether to limit the sample based on mass_range. Added for debugging purposes.
                
    binary_identical_radii : bool, default = False
                Whether you want the stars in the binary system to have the same radii. Strongly recommended to set to False. This was implemented for debugging and testing purposes.

    error_modifier_range : tuple, default = (1,1)
                We draw a random number from a uniform distribution defined by `error_modifier_range'. After calculating the uncertainty, the error is multiplied with this number. Currently only implemented for `noise_model' = 5.

    main_sequence_model : main_sequence_model class or None, default = None
                The user has the option to specify a main sequence model from which this function will draw MS stars (with a width of `delta_log_L`). This must be specified in the form of a `main_sequence_model` object. Do help(main_sequence_model) for more information.

    sample_cooler_companions : boolean, default = False
                To generate a binary, by default, this code randomly samples two templates and adds them together. However, by setting sample_cooler_companions = True, then this code will first randomly sample the first star, and then randomly sample a second star from the subset of templates that have lower temperatures to the first star. This ensures that the second star is always cooler than the first star. 
    
    sample_radius_distribution : string, default = 'uniform'
                Determines how the radius is sampled. Either uniformly between the minimum and maximum values (determined by delta_log_L and/or mass limits) or as a Gaussian where the 3simga is set at the minimum and maximum values. 

                
    
    Returns
    -------

    templates_test : list
                A list of your generated sample, which is effectively a list of `utils.sed_template' with adjusted fluxes.
    
    Y_test : list
                A list of whether your sample is a single star (0) or binary (1).

    df : pandas.DataFrame
                A pandas dataframe with more information of your generated sample.


    Example
    -------

    Example #1: Create 10 binary systems

    templates_test, _, _ = create_sample(templates, n_double = 10, throughput_curves = rubin_throughput_curves)



    Example #2: Create a larger catalogue of both single and binary systems. Also specify desired distances and masses

    dist_range = (0.001, 100)
    mass_range = (0.08,300)
    radius_range = (0.02,2_000)
    
    templates_test, _, df = create_sample(templates_full, n_single = 100, n_double = 100, throughput_curves = rubin_throughput_curves, 
                                          dist_range = dist_range, mass_range = mass_range, radius_range = radius_range)
    
    
    Notes
    -----

    
    
    '''


    ### Config 
    templates = copy.deepcopy(templates) #To make sure we do not overwrite
    n_tot = n_single + n_double


    dist_min, dist_max = dist_range
    radius_min, radius_max = radius_range
    radius_min = radius_min*u.Rsun.to(u.kpc)
    radius_max = radius_max*u.Rsun.to(u.kpc) 


    df = [] #to collect extra background information

    templates_test = [] #All sample templates go here
    Y_test = [] #Whether it is a binary (1) or single star (0)



    if gammas is None:
        gammas = { #From https://arxiv.org/abs/0805.2366
            'u' : 0.038,
            'g' : 0.039,
            'r' : 0.039,
            'i' : 0.039,
            'z' : 0.039,
            'y': 0.039
            }
        
    if mag5sigmas is None:    
        mag5sigmas = { #Single exposures, #From https://rubinobservatory.org/for-scientists/rubin-101/key-numbers
            'u' : 23.9,
            'g' : 25.0,
            'r' : 24.7,
            'i' : 24.0,
            'z' : 23.3,
            'y': 22.1
            }


    if sigma_sys is None:
        sigma_sys = { #0.005, #From https://rubinobservatory.org/for-scientists/rubin-101/key-numbers
            'u' : 0.005,
            'g' : 0.005,
            'r' : 0.005,
            'i' : 0.005,
            'z' : 0.005,
            'y': 0.005
            }




    # Apply T_range, g_range, and Z_range if needed. Will only work 
    idxs_remove = []
    for i,t in enumerate(templates):
        if t.T is not None:
            if t.T < T_range[0] or t.T > T_range[1]:
                idxs_remove.append(i)
    
        if t.g is not None:
            if t.g < g_range[0] or t.g > g_range[1]:
                idxs_remove.append(i)
    
        if t.Z is not None:
            if t.Z < Z_range[0] or t.Z > Z_range[1]:
                idxs_remove.append(i)
    
    idxs_keep = [i for i in range(len(templates)) if i not in idxs_remove]  
    #templates = templates[idxs_keep] #This messes with idx selection later. Just sample from idxs_keep instead


    ### Single Stars 
    if n_single < len(idxs_keep):
        replace = False
    else:
        replace = True

    #idx_single = np.random.choice(list(range(len(templates))), n_single, replace = replace)
    idx_single = np.random.choice(list(idxs_keep), n_single, replace = replace)

    for idx in idx_single:
        df_row = {}
        template = copy.deepcopy(templates[idx]) # To make sure we do not overwrite anything

        if use_mass_constraint:
            '''
            g = G * M / R^2
            <=> R^2 = G * M / g
            <=> R = sqrt(G * M / g)
            Then input mass limits, and calculate radius lower and upper limits   
            '''

            G = 6.6743*10**-11 #m3 kg-1 s-2
            g = 10**(template.g) / 100 #cm s-2 to m s-2

            radius_ll_mc = (np.sqrt(G * np.min(mass_range)*u.Msun.to(u.kg) / g)) * u.m.to(u.Rsun)
            radius_ul_mc = (np.sqrt(G * np.max(mass_range)*u.Msun.to(u.kg) / g)) * u.m.to(u.Rsun)
            
        else:
            radius_ll_mc = radius_min*u.kpc.to(u.Rsun)
            radius_ul_mc = radius_max*u.kpc.to(u.Rsun)


            

        ### Sample radius
        if only_main_sequence:
            radius_ll, _, radius_ul = ms_radius_limit(template.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius_ll = np.max([radius_ll,radius_min*u.kpc.to(u.Rsun),radius_ll_mc])
            radius_ul_lumlim = stefan_boltzmann(temperature = template.T, luminosity = lum_limit)
            radius_ul = np.min([radius_ul,radius_ul_lumlim,radius_max*u.kpc.to(u.Rsun), radius_ul_mc])
            assert radius_ll < radius_ul, 'Something went wrong with radius limit calculations'

        elif only_main_sequence_and_giants:
            radius_ll, _, _ = ms_radius_limit(template.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius_ll = np.max([radius_ll,radius_min*u.kpc.to(u.Rsun),radius_ll_mc])
            radius_ul_lumlim = stefan_boltzmann(temperature = template.T, luminosity = lum_limit)
            radius_ul = np.min([radius_ul_lumlim,radius_max*u.kpc.to(u.Rsun),radius_ul_mc])
            assert radius_ll < radius_ul, 'Something went wrong with radius limit calculations'
        
        else:
            radius_ll = np.max([radius_min*u.kpc.to(u.Rsun),radius_ll_mc])
            radius_ul = np.min([radius_max*u.kpc.to(u.Rsun),radius_ul_mc])
            

            
        if sample_radius_in_logspace: 
            if sample_radius_distribution == 'uniform':
                radius = np.random.uniform(np.log10(radius_ll*u.Rsun.to(u.kpc)), np.log10(radius_ul*u.Rsun.to(u.kpc)))
            elif sample_radius_distribution == 'gaussian':
                mid_rad = np.mean([np.log10(radius_ll*u.Rsun.to(u.kpc)), np.log10(radius_ul*u.Rsun.to(u.kpc))])
                std_rad = (np.log10(radius_ul*u.Rsun.to(u.kpc)) - np.log10(radius_ll*u.Rsun.to(u.kpc)))/2
                radius = np.random.normal(mid_rad, std_rad/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.
            radius = 10**radius
            
        else:
            if sample_radius_distribution == 'uniform':
                radius = np.random.uniform(radius_ll*u.Rsun.to(u.kpc), radius_ul*u.Rsun.to(u.kpc))
            elif sample_radius_distribution == 'gaussian':

                mid_rad = np.mean([radius_ll*u.Rsun.to(u.kpc), radius_ul*u.Rsun.to(u.kpc)])
                std_rad = (radius_ul*u.Rsun.to(u.kpc) - radius_ll*u.Rsun.to(u.kpc))/2
                radius = np.random.normal(mid_rad, std_rad/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.
                

        

        ### Sample distance
        if sample_distance_linearly:
            distance = np.random.uniform(dist_min, dist_max)
        else:
            u_random = np.random.uniform(0, 1)
            distance = (u_random * (dist_max**3 - dist_min**3) + dist_min**3)**(1/3)


        # Update flux to observed on Earth
        amplitude = (radius/distance)**2 
        template.flux = np.array(template.flux) * amplitude


        # Apply extinction
        Av = np.random.uniform(Av_range[0],Av_range[1])
        if Av != 0:
            template.redden(Av = Av, extinction_function = extinction_function)
        
        #reduced_template = reduce_sed(template, throughput_curves)
        template.convert_flux_to_mag()
        template.convolve(throughput_curves)
        reduced_template = template
        
        reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)
    
        if add_noise:
            if noise_model == 1: 
                noise = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_template.flux).shape)
                            
                reduced_template.flux += noise
                reduced_template.flux[reduced_template.flux<0] = zero_flux

                sigma = [noise_bkg]*len(reduced_template.mag)
                reduced_template.e_flux = sigma

                reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)

            if noise_model == 2:
                # add noise on magnitude scale. Makes more sense across orders of magnitude, easier to guess with observations, 
                # more intuitie, and avoids negative fluxes. BUT: it's not really what I see happening in data. Usually
                # mag and mag_err are very much anti-correlated
                
                noise = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_template.mag).shape)
                reduced_template.mag += noise

                sigma = [noise_bkg]*len(reduced_template.mag)
                reduced_template.e_mag = sigma
                
                reduced_template.convert_mag_to_flux(throughput_curves = throughput_curves)
                


            if noise_model == 3:
                noise = np.random.normal(loc=0, scale=noise_scale*np.array(reduced_template.flux), size=np.array(reduced_template.flux).shape)
                            
                reduced_template.flux += noise
                reduced_template.flux[reduced_template.flux<0] = zero_flux

                sigma = noise_scale*np.array(reduced_template.flux)
                reduced_template.e_flux = sigma 
                
                reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)
                


            if noise_model == 4:
                noise_1 = np.random.normal(loc=0, scale=noise_scale*np.array(reduced_template.flux), size=np.array(reduced_template.flux).shape)
                noise_2 = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_template.flux).shape)
                noise = np.sqrt(noise_1**2 + noise_2**2)

                reduced_template.flux += noise
                reduced_template.flux[reduced_template.flux<0] = zero_flux

                sigma = noise_scale*np.array(reduced_template.flux) + noise_bkg ## Is this the correct addition??
                reduced_template.e_flux = sigma 

                reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)



            if noise_model == 5:

                # Add mags
                reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)

                # Find sigma in mag space - see Eq 5 in https://arxiv.org/abs/0805.2366
                sigmas = []
                for bi, band in enumerate(reduced_template.bands):
                    mag = reduced_template.mag[bi]
                    mag5sigma = mag5sigmas[band]
                    gamma = gammas[band]
                    sigmasys = sigma_sys[band]
                            
                    x = 10**(0.4 * (mag - mag5sigma))
                    sigma2_rand = (0.04 - gamma) * x + gamma * x**2
                    sigma2_sys = sigmasys**2
                    sigma = np.sqrt(sigma2_rand + sigma2_sys)
                    sigmas.append(sigma)

                error_modifier = np.random.uniform(error_modifier_range[0], error_modifier_range[1], len(sigmas))
                sigmas = sigmas * error_modifier
                
                reduced_template.e_mag = sigmas
                
                # Add error in flux space
                reduced_template.convert_mag_to_flux(throughput_curves = throughput_curves)

                
                noise = np.random.normal(loc=0, scale=reduced_template.e_flux, size=np.array(reduced_template.flux).shape)
                reduced_template.flux += noise
                reduced_template.flux[reduced_template.flux<0] = zero_flux

                # Convert back to mags 
                reduced_template.convert_flux_to_mag(throughput_curves = throughput_curves)


        reduced_template.flux = np.array(reduced_template.flux) #Make sure it is an array 
                
                
                
        Y_test.append(0)
        templates_test.append(reduced_template)
    

        df_row['idx1'] = idx
        df_row['idx2'] = np.nan
        df_row['distance'] = distance
        df_row['radius1'] = radius*u.kpc.to(u.Rsun)
        df_row['radius2'] = np.nan
        df_row['amplitude1'] = amplitude
        df_row['amplitude2'] = np.nan
        df_row['Av'] = Av
        
        df_row['temperature1'] = template.T
        df_row['temperature2'] = np.nan

        df_row['g1'] = template.g
        df_row['g2'] = np.nan

        df_row['Z1'] = template.Z
        df_row['Z2'] = np.nan
        
        df_row['delta_T'] = np.nan

        lum = stefan_boltzmann(temperature = template.T, radius = radius*u.kpc.to(u.Rsun))
        df_row['lum1'] = lum
        df_row['lum2'] = np.nan
        
        df_row['flux_ratio'] = np.nan

        if add_noise:
            df_row['snr'] = np.nanmedian(np.array(reduced_template.flux) / np.array(reduced_template.e_flux))
            df_row['snr_max'] = np.nanmax(np.array(reduced_template.flux) / np.array(reduced_template.e_flux))
            df_row['snr_min'] = np.nanmin(np.array(reduced_template.flux) / np.array(reduced_template.e_flux))
        else:
            df_row['snr'] = np.nan
            df_row['snr_max'] = np.nan
            df_row['snr_min'] = np.nan
        

        df.append(df_row)

    


    ### Binary Stars
    idx_binary = []
    for i in range(n_double):
        df_row = {}
        # idx1 = np.random.choice(list(range(len(templates))), 1)[0]
        # idx2 = np.random.choice(list(range(len(templates))), 1)[0]

        
        idx1 = np.random.choice(list(idxs_keep), 1)[0]

        if sample_cooler_companions:
            temperature_primary = templates[idx1].T
            idx_secondary = find_template(templates, T_max = temperature_primary)
            idx_secondary = intersection([idxs_keep, idx_secondary])

            idx2 = np.random.choice(list(idx_secondary), 1)[0]


        else:
            idx2 = np.random.choice(list(idxs_keep), 1)[0]

        
        
        template1 = copy.deepcopy(templates[idx1])
        template2 = copy.deepcopy(templates[idx2])


        if use_mass_constraint:
            '''
            g = G * M / R^2
            <=> R^2 = G * M / g
            <=> R = sqrt(G * M / g)
            Then input mass limits, and calculate radius lower and upper limits   
            '''

            G = 6.6743*10**-11 #m3 kg-1 s-2
            g1 = 10**(template1.g) / 100 #cm s-2 to m s-2
            g2 = 10**(template2.g) / 100 #cm s-2 to m s-2

            radius1_ll_mc = (np.sqrt(G * np.min(mass_range)*u.Msun.to(u.kg) / g1)) * u.m.to(u.Rsun)
            radius1_ul_mc = (np.sqrt(G * np.max(mass_range)*u.Msun.to(u.kg) / g1)) * u.m.to(u.Rsun)

            radius2_ll_mc = (np.sqrt(G * np.min(mass_range)*u.Msun.to(u.kg) / g2)) * u.m.to(u.Rsun)
            radius2_ul_mc = (np.sqrt(G * np.max(mass_range)*u.Msun.to(u.kg) / g2)) * u.m.to(u.Rsun)
            
        else:
            radius1_ll_mc = radius_min*u.kpc.to(u.Rsun)
            radius1_ul_mc = radius_max*u.kpc.to(u.Rsun)
            radius2_ll_mc = radius_min*u.kpc.to(u.Rsun)
            radius2_ul_mc = radius_max*u.kpc.to(u.Rsun)
    


        ### Sample radius
        if only_main_sequence:
            radius1_ll, _, radius1_ul = ms_radius_limit(template1.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius1_ll = np.max([radius1_ll,radius_min*u.kpc.to(u.Rsun),radius1_ll_mc])
            radius1_ul_lumlim = stefan_boltzmann(temperature = template1.T, luminosity = lum_limit)
            radius1_ul = np.min([radius1_ul,radius1_ul_lumlim,radius_max*u.kpc.to(u.Rsun),radius1_ul_mc])
            assert radius1_ll < radius1_ul, 'Something went wrong with radius limit calculations'
            
            radius2_ll, _, radius2_ul = ms_radius_limit(template2.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius2_ll = np.max([radius2_ll,radius_min*u.kpc.to(u.Rsun),radius2_ll_mc])
            radius2_ul_lumlim = stefan_boltzmann(temperature = template2.T, luminosity = lum_limit)
            radius2_ul = np.min([radius2_ul,radius2_ul_lumlim,radius_max*u.kpc.to(u.Rsun),radius2_ul_mc])
            assert radius2_ll < radius2_ul, 'Something went wrong with radius limit calculations'
            
        elif only_main_sequence_and_giants:
            radius1_ll, _, _ = ms_radius_limit(template1.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius1_ll = np.max([radius1_ll,radius_min*u.kpc.to(u.Rsun),radius1_ll_mc])
            radius1_ul_lumlim = stefan_boltzmann(temperature = template1.T, luminosity = lum_limit)
            radius1_ul = np.min([radius1_ul_lumlim,radius_max*u.kpc.to(u.Rsun),radius1_ul_mc])
            assert radius1_ll < radius1_ul, 'Something went wrong with radius limit calculations'
    
            radius2_ll, _, _ = ms_radius_limit(template2.T, delta_log_L = delta_log_L, main_sequence_model = main_sequence_model)
            radius2_ll = np.max([radius2_ll,radius_min*u.kpc.to(u.Rsun),radius2_ll_mc])
            radius2_ul_lumlim = stefan_boltzmann(temperature = template2.T, luminosity = lum_limit)
            radius2_ul = np.min([radius2_ul_lumlim,radius_max*u.kpc.to(u.Rsun),radius2_ul_mc])
            assert radius2_ll < radius2_ul, 'Something went wrong with radius limit calculations'

            
        else:   

            radius1_ll = np.max([radius_min*u.kpc.to(u.Rsun),radius1_ll_mc])
            radius1_ul = np.min([radius_max*u.kpc.to(u.Rsun),radius1_ul_mc])

            radius2_ll = np.max([radius_min*u.kpc.to(u.Rsun),radius2_ll_mc])
            radius2_ul = np.min([radius_max*u.kpc.to(u.Rsun),radius2_ul_mc])
        
            



        
        # OLD
        # if sample_radius_in_logspace:
        #     radius1 = np.random.uniform(np.log10(radius1_ll*u.Rsun.to(u.kpc)), np.log10(radius1_ul*u.Rsun.to(u.kpc)))
        #     radius1 = 10**radius1
            
        #     radius2 = np.random.uniform(np.log10(radius2_ll*u.Rsun.to(u.kpc)), np.log10(radius2_ul*u.Rsun.to(u.kpc)))
        #     radius2 = 10**radius2

        # else:
        #     radius1 = np.random.uniform(radius1_ll*u.Rsun.to(u.kpc), radius1_ul*u.Rsun.to(u.kpc))
        #     radius2 = np.random.uniform(radius2_ll*u.Rsun.to(u.kpc), radius2_ul*u.Rsun.to(u.kpc))

        

        if sample_radius_in_logspace: 
            if sample_radius_distribution == 'uniform':
                radius1 = np.random.uniform(np.log10(radius1_ll*u.Rsun.to(u.kpc)), np.log10(radius1_ul*u.Rsun.to(u.kpc)))
                radius2 = np.random.uniform(np.log10(radius2_ll*u.Rsun.to(u.kpc)), np.log10(radius2_ul*u.Rsun.to(u.kpc)))
            elif sample_radius_distribution == 'gaussian':
                mid_rad1 = np.mean([np.log10(radius1_ll*u.Rsun.to(u.kpc)), np.log10(radius1_ul*u.Rsun.to(u.kpc))])
                std_rad1 = (np.log10(radius1_ul*u.Rsun.to(u.kpc)) - np.log10(radius1_ll*u.Rsun.to(u.kpc)))/2
                radius1 = np.random.normal(mid_rad1, std_rad1/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.

                mid_rad2 = np.mean([np.log10(radius2_ll*u.Rsun.to(u.kpc)), np.log10(radius2_ul*u.Rsun.to(u.kpc))])
                std_rad2 = (np.log10(radius2_ul*u.Rsun.to(u.kpc)) - np.log10(radius2_ll*u.Rsun.to(u.kpc)))/2
                radius2 = np.random.normal(mid_rad2, std_rad2/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.
                
            radius1 = 10**radius1
            radius2 = 10**radius2
            
            
        else:
            if sample_radius_distribution == 'uniform':
                radius1 = np.random.uniform(radius1_ll*u.Rsun.to(u.kpc), radius1_ul*u.Rsun.to(u.kpc))
                radius2 = np.random.uniform(radius2_ll*u.Rsun.to(u.kpc), radius2_ul*u.Rsun.to(u.kpc))
                
            elif sample_radius_distribution == 'gaussian':
                mid_rad1 = np.mean([radius1_ll*u.Rsun.to(u.kpc), radius1_ul*u.Rsun.to(u.kpc)])
                std_rad1 = (radius1_ul*u.Rsun.to(u.kpc) - radius1_ll*u.Rsun.to(u.kpc))/2
                radius1 = np.random.normal(mid_rad1, std_rad1/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.

                mid_rad2 = np.mean([radius2_ll*u.Rsun.to(u.kpc), radius2_ul*u.Rsun.to(u.kpc)])
                std_rad2 = (radius2_ul*u.Rsun.to(u.kpc) - radius2_ll*u.Rsun.to(u.kpc))/2
                radius2 = np.random.normal(mid_rad2, std_rad2/3)#divided by 3 so that 3sigma (=99.7%) is within boundaries.


        if binary_identical_radii: 
                radius2 = radius1


        
        # Sample distance
        if sample_distance_linearly:
            distance = np.random.uniform(dist_min, dist_max)
        else:
            u_random = np.random.uniform(0, 1)
            distance = (u_random * (dist_max**3 - dist_min**3) + dist_min**3)**(1/3)


        # Update flux to observed on Earth
        amplitude_1 = (radius1/distance)**2 
        amplitude_2 = (radius2/distance)**2 
        
        template1.flux = np.array(template1.flux) * amplitude_1
        template2.flux = np.array(template2.flux) * amplitude_2
    
        
        combined_flux = template1.flux + template2.flux
        combined_template = sed_template(wavelength = template2.wavelength, flux = combined_flux)


        # Apply extinction
        Av = np.random.uniform(Av_range[0],Av_range[1])
        if Av != 0:
            combined_template.redden(Av = Av, extinction_function = extinction_function)
            
        
        #reduced_combined_template = reduce_sed(combined_template, throughput_curves)
        combined_template.convert_flux_to_mag()
        combined_template.convolve(throughput_curves)
        reduced_combined_template = combined_template

        
        reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)
    
        if add_noise:
            if noise_model == 1: 
                noise = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_combined_template.flux).shape)
                            
                reduced_combined_template.flux += noise
                reduced_combined_template.flux[reduced_combined_template.flux<0] = zero_flux

                sigma = [noise_bkg]*len(reduced_combined_template.mag)
                reduced_combined_template.e_flux = sigma
                
                reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)

            if noise_model == 2:
                # add noise on magnitude scale. Makes more sense across orders of magnitude, easier to guess with observations, 
                # more intuitie, and avoids negative fluxes. BUT: it's not really what I see happening in data. Usually
                # mag and mag_err are very much anti-correlated
                
                noise = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_combined_template.mag).shape)
                reduced_combined_template.mag += noise

        
                sigma = [noise_bkg]*len(reduced_combined_template.mag)
                reduced_combined_template.e_mag = sigma
                
                reduced_combined_template.convert_mag_to_flux(throughput_curves = throughput_curves)


            if noise_model == 3:
                noise = np.random.normal(loc=0, scale=noise_scale*np.array(reduced_combined_template.flux), size=np.array(reduced_combined_template.flux).shape)
                            
                reduced_combined_template.flux += noise
                reduced_combined_template.flux[reduced_combined_template.flux<0] = zero_flux

                sigma = noise_scale*np.array(reduced_combined_template.flux)
                reduced_combined_template.e_flux = sigma 
                
                reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)


            if noise_model == 4:
                noise_1 = np.random.normal(loc=0, scale=noise_scale*np.array(reduced_combined_template.flux), size=np.array(reduced_combined_template.flux).shape)
                noise_2 = np.random.normal(loc=0, scale=noise_bkg, size=np.array(reduced_combined_template.flux).shape)
                noise = np.sqrt(noise_1**2 + noise_2**2)

                reduced_combined_template.flux += noise
                reduced_combined_template.flux[reduced_combined_template.flux<0] = zero_flux

                sigma = noise_scale*np.array(reduced_combined_template.flux) + noise_bkg ## Is this the correct addition??
                reduced_combined_template.e_flux = sigma 
                
                reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)


            if noise_model == 5:
                # Add mags
                reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)

                # Find sigma in mag space - see Eq 5 in https://arxiv.org/abs/0805.2366
                sigmas = []
                for bi, band in enumerate(reduced_combined_template.bands):
                    mag = reduced_combined_template.mag[bi]
                    mag5sigma = mag5sigmas[band]
                    gamma = gammas[band]
                    sigmasys = sigma_sys[band]
                            
                    x = 10**(0.4 * (mag - mag5sigma))
                    sigma2_rand = (0.04 - gamma) * x + gamma * x**2
                    sigma2_sys = sigmasys**2
                    sigma = np.sqrt(sigma2_rand + sigma2_sys)
                    sigmas.append(sigma)

                error_modifier = np.random.uniform(error_modifier_range[0], error_modifier_range[1], len(sigmas))
                sigmas = sigmas * error_modifier

                reduced_combined_template.e_mag = sigmas
                
                # Add error in flux space
                reduced_combined_template.convert_mag_to_flux(throughput_curves = throughput_curves)

                
                noise = np.random.normal(loc=0, scale=reduced_combined_template.e_flux, size=np.array(reduced_combined_template.flux).shape)
                reduced_combined_template.flux += noise
                reduced_combined_template.flux[reduced_combined_template.flux<0] = zero_flux

                # Convert back to mags 
                reduced_combined_template.convert_flux_to_mag(throughput_curves = throughput_curves)


        reduced_combined_template.flux = np.array(reduced_combined_template.flux) #Make sure it is an array 
                
        Y_test.append(1)
        templates_test.append(reduced_combined_template)

        df_row['idx1'] = idx1
        df_row['idx2'] = idx2
        df_row['distance'] = distance
        df_row['radius1'] = radius1*u.kpc.to(u.Rsun)
        df_row['radius2'] = radius2*u.kpc.to(u.Rsun)
        df_row['amplitude1'] = amplitude_1
        df_row['amplitude2'] = amplitude_2
        df_row['Av'] = Av
        
        df_row['temperature1'] = template1.T
        df_row['temperature2'] = template2.T
        df_row['delta_T'] = np.abs(template1.T-template2.T)
        df_row['temperature_avg'] = (template1.T+template2.T)/2
        
        df_row['g1'] = template1.g
        df_row['g2'] = template2.g
        df_row['delta_g'] = np.abs(template1.g-template2.g)

        df_row['Z1'] = template1.Z
        df_row['Z2'] = template2.Z
        df_row['delta_Z'] = np.abs(template1.Z-template2.Z)

        lum1 = stefan_boltzmann(temperature = template1.T, radius = radius1*u.kpc.to(u.Rsun))
        lum2 = stefan_boltzmann(temperature = template2.T, radius = radius2*u.kpc.to(u.Rsun))
        df_row['lum1'] = lum1
        df_row['lum2'] = lum2
        
        #max_flux = [np.max(reduce_sed(template1, throughput_curves).flux), np.max(reduce_sed(template2, throughput_curves).flux)]
        #df_row['flux_ratio'] = np.min(max_flux)/np.max(max_flux)

        sum_flux =  [np.sum(reduce_sed(template1, throughput_curves).flux), np.sum(reduce_sed(template2, throughput_curves).flux)]
        df_row['flux_ratio'] = np.min(sum_flux)/np.max(sum_flux)
        
        if add_noise:
            df_row['snr'] = np.nanmedian(np.array(reduced_combined_template.flux) / np.array(reduced_combined_template.e_flux))
            df_row['snr_max'] = np.nanmax(np.array(reduced_combined_template.flux) / np.array(reduced_combined_template.e_flux))
            df_row['snr_min'] = np.nanmin(np.array(reduced_combined_template.flux) / np.array(reduced_combined_template.e_flux))
        else:
            df_row['snr'] = np.nan
            df_row['snr_max'] = np.nan
            df_row['snr_min'] = np.nan
        

        df.append(df_row)

    df = pd.DataFrame(df)


    df['type1'] = determine_hr_type(list(df['temperature1']), list(df['lum1']), delta_log_L = delta_log_L)
    df['type2'] = determine_hr_type(list(df['temperature2']), list(df['lum2']), delta_log_L = delta_log_L)
    
    df['Y_test'] = Y_test




    return templates_test, Y_test, df












def plot_sample_on_hr_diagram(df_info):
    '''
    Takes the output of create_sample() as input, and plots the targets on the HR diagram.

    Example:

    _, _, df_info = create_sample(args)
    plot_sample_on_hr_diagram(df_info)
    '''

    plt.figure(figsize = (8,4))

    plt.subplot(1,2,1)
    plot_hr_diagram(ax = plt.gca(), MS_color = 'grey')

    # Plot single stars
    idx_single = np.where(df_info['Y_test'] == 0)[0]
    temps = list(df_info['temperature1'][idx_single]) 
    lums = list(df_info['lum1'][idx_single])

    plt.scatter(temps,lums, color = 'C1')
    plt.title('Single Stars')



    # Plot binaries
    plt.subplot(1,2,2)
    plot_hr_diagram(ax = plt.gca(), MS_color = 'grey')
    idx_double = np.where(df_info['Y_test'] == 1)[0]
    temps1 = list(df_info['temperature1'][idx_double]) 
    lums1 = list(df_info['lum1'][idx_double])

    temps2 = list(df_info['temperature2'][idx_double]) 
    lums2 = list(df_info['lum2'][idx_double])

    plt.plot([temps1,temps2],[lums1,lums2], color = 'C1')
    plt.scatter(temps1,lums1, color = 'C1')
    plt.scatter(temps2,lums2, color = 'C1')

    plt.title('Binaries')
    
    plt.tight_layout()
    plt.show()




def plot_sample_on_kiel_diagram(df_info):
    '''
    Takes the output of create_sample() as input, and plots the targets on the Kiel diagram.

    Example:

    _, _, df_info = create_sample(args)
    plot_sample_on_kiel_diagram(df_info)
    '''

    plt.figure(figsize = (8,4))

    plt.subplot(1,2,1)
    # Plot single stars
    idx_single = np.where(df_info['Y_test'] == 0)[0]
    temps = list(df_info['temperature1'][idx_single]) 
    gs = list(df_info['g1'][idx_single])

    plot_kiel_diagram(temperatures = temps, gs = gs, ax = plt.gca())
    plt.title('Single Stars')



    # Plot binaries
    plt.subplot(1,2,2)

    idx_double = np.where(df_info['Y_test'] == 1)[0]
    temps1 = list(df_info['temperature1'][idx_double]) 
    gs1 = list(df_info['g1'][idx_double])

    temps2 = list(df_info['temperature2'][idx_double]) 
    gs2 = list(df_info['g2'][idx_double])

    
    plot_kiel_diagram(temperatures = temps1, gs = gs1, ax = plt.gca())
    #plt.scatter(temps1,lums1, color = 'C1')

    plt.scatter(temps2,gs2, color = 'C1')
    plt.plot([temps1,temps2],[gs1,gs2], color = 'C1')

    plt.title('Binaries')
    
    plt.tight_layout()
    plt.show()




### ==================== ###
### Throughput functions ###
### ==================== ###


def create_gaussian_throughput_curves(band_info = {}, min_wav = 2000, max_wav = 14_000, delta_wav = 1, plot = False):
    '''
    Will create a Gaussian throughput curve. 

    INPUT:
    band_info: dict with the following shape: 
        {
            band_name_1 : [central_wavelength_1, fwhm_band_1, amplitude_1], 
            band_name_2 : [central_wavelength_2, fwhm_band_2, amplitude_2], 
            ...
        } 
    e.g:
        {
            'u' : [3724, 500, 0.2],
            'g' : [4807, 1000, 0.5]
        }

    where:    
        central_wavelength: central wavelength, in A
        fwhm_band: full width half max, in A
        amplitude: max amplitude of throughput curve. Between 0 and 1

    If this is not specified, we will load Rubin-ish bands. 

        
    min_wav: minimum wavelength, in A
    max_wav: maximum wavelength, in A
    delta_wav: stepsize for wavelength, in A
    
    OUTPUT:
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_1],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }

    wavelength_band: wavelength array, in A
    throughput_band: throughput array, int between 0 and 1
    '''
    if band_info == {}:
        band_info = {
            'u' : [3724, 500, 0.2], #name : [cen_wav, fwhm, amp]
            'g' : [4807, 1000, 0.5],
            'r' : [6221, 1000, 0.55],
            'i' : [7559, 1000, 0.6],
            'z' : [8680, 1000, 0.6],
            'y' : [9753, 1000, 0.3]
            }
    
    throughput_curves = {}
    
    for i, band in enumerate(band_info.keys()):
        cen_wav, fwhm, amp = band_info[band]
        
        W_filter = np.arange(min_wav, max_wav, delta_wav)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        T_filter = amp * np.exp(-0.5 * ((W_filter - cen_wav) / sigma)**2)

        throughput_curves[band] = [W_filter, T_filter]




    #### Plot
    if plot:

        for i, band in enumerate(band_info.keys()):
            plt.plot(throughput_curves[band][0], throughput_curves[band][1], ls = '--', label = band)
        
        plt.legend()
        
        plt.xlabel('Wavelength [A]')
        plt.ylabel('Throughput')
        
        plt.title('Gaussian throughput curves')
        plt.show()

    return throughput_curves




def load_throughput_curves(band_info, input_wavelength_units = u.nm, output_wavelength_units = u.Angstrom, plot = False):
    """
    Will load throughput curves from a directory. 

    INPUT:
    band_info: dict with the following shape:
        {
            band_name_1 : location_of_throughput_curve.dat,
            ...
        }

    where:
    location_of_throughput_curve.dat: the location of the throughput curve. 
    It should be a .dat file with two columns: wavelength and throughput. It'll be read by doing:

    df_throughput = pd.read_csv(location_of_throughput_curve.dat, sep=r's+',comment='#',header=None)

    IMPORTANT: Specify the wavelength units of the input!!! Most of the code assuming wavelength units to be in A.
    However, this piece of code assumes the wavelength in the .dat file is in nm. This is because that is the 
    most common default unit in which these throughput curves are shared. If this is not the case, adjust 
    input_wavelength_units accordingly. 


    OUTPUT:
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_1],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }

    wavelength_band: wavelength array, in A
    throughput_band: throughput array, float between 0 and 1
    """


    throughput_curves = {}
    
    for i, band in enumerate(band_info.keys()):
        fileloc = band_info[band]

        df_throughput = pd.read_csv(fileloc,sep=r'\s+',comment='#',header=None)
        df_throughput.columns = ['wavelength', 'throughput']
        
        throughput_curves[band] = [df_throughput['wavelength']*input_wavelength_units.to(output_wavelength_units), df_throughput['throughput']]



    #### Plot
    if plot:

        for i, band in enumerate(band_info.keys()):
            plt.plot(throughput_curves[band][0], throughput_curves[band][1], ls = '-', label = band)
        
        plt.legend()
        
        plt.xlabel('Wavelength [A]')
        plt.ylabel('Throughput')
        
        plt.title('Loaded throughput curves')
        plt.show()

    
    return throughput_curves




def single_band_integration(W_star, F_star, W_filter, T_filter):
    """
    Integrate model spectrum with throughput curve to obtain observed flux.

    INPUT:
    W_star: model wavelength array, in A
    F_star: model flux array, erg/s/cm2/A
    W_filter: filter wavelength array, in A
    T_filter: filter throughput array, between 0-1

    OUTPUT:
    pivot wavelength, in A
    integrated observed flux, in erg/s/cm2/A
    """

    # # Step 0: Prepare data
    # W_filter = np.array(df_throughput['wavelength_A']) # Wavelengths (A)
    # T_filter = np.array(df_throughput['throughput']) # Throughputs (0-1)
    
    # W_star = np.array(wavelengths) # Stellar Wavelengths (A)
    # F_star = np.array(blackbody_spectrum) # Stellar Fluxes (e.g., erg/s/cm^2/A)

    # Step 1: only integrate over the range where the filter has throughput.
    # Find the indices where the throughput is greater than zero -- Though does this cause issues? 
    # valid_indices = T_filter > 0
    # W_filter = W_filter[valid_indices]
    # T_filter = T_filter[valid_indices]
    
    
    # Step 2: Interpolation
    # The model spectrum will be on a different grid than the throughput curve. 
    # Interpolate the model spectrum to be on the same grid as throughput
    star_interp_func = interp1d(W_star, F_star, kind='linear', bounds_error=False, fill_value=0)
    F_star_interp = star_interp_func(W_filter)
    
    
    # Step 3: Bandpass integration
    F_observed_lambda = F_star_interp * T_filter * W_filter #The multiplication CHANGED THIS *W_filter
    F_filter_integrated = np.trapezoid(F_observed_lambda, W_filter) #The integration


    # Step 4: Calculate effective bandwidth
    delta_lambda = np.trapezoid(T_filter * W_filter, W_filter) #The integration CHANGED THIS *W_filter
    F_filter_integrated = F_filter_integrated / delta_lambda
    

    # Step 5: Calculate pivot wavelength
    numerator = np.trapezoid(W_filter * T_filter, W_filter)
    denominator = np.trapezoid((1.0 / W_filter) * T_filter, W_filter)
    lambda_P_sq = numerator / denominator
    lambda_P = np.sqrt(lambda_P_sq)

    return lambda_P,F_filter_integrated



def calculate_effective_bandwidth(throughput_curves, normalize = False):
    """
    Given some throughput curves, this will calculate the effective bandwidth. 

    INPUT:
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_2],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }
    where:
    wavelength_band: wavelength array, in A
    throughput_band: throughput array, int between 0 and 1. Dimensionless


    OUTPUT:
    effective_bandwidths: dict with the following shape:
        {
            band_name_1 : effective_bandwidth_1,
            band_name_2 : effective_bandwidth_2,
            ...
        }
    where:
    effective_bandwidth_1 is in A.

    We do this by integrating the throughput curve over wavelength:
    Delta lambda_eff = int T_lambda (lambda) * dlambda

    OPTIONAL INPUT:
    normalize: bool. If False (default), we calculate: 
    Delta lambda_eff = int T_lambda (lambda) * dlambda
    However, bandwidths are often quoted normalized. So if True, we do:
    Delta lambda_eff = int T_lambda (lambda) * dlambda / max(T_lambda)
    """

    pivots = calculate_pivot(throughput_curves)
    
    
    effective_bandwidths = {}
    for band in throughput_curves.keys():
        band_wavelength = throughput_curves[band][0]
        band_throughput = throughput_curves[band][1]

        delta_lambda = np.trapezoid(band_throughput * band_wavelength, band_wavelength) #The integration #CHANGED THIS, added * band_wavelength
        delta_lambda = delta_lambda / pivots[band] # CHANGED THIS

        if normalize:
            delta_lambda = delta_lambda / np.max(band_throughput)
            
        effective_bandwidths[band] = delta_lambda


    return effective_bandwidths


def calculate_pivot(throughput_curves):
    """
    By default, this calculates the pivot wavelength.

    INPUT:
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_2],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }
    where:
    wavelength_band: wavelength array, in A
    throughput_band: throughput array, int between 0 and 1. Dimensionless


    OUTPUT:
    effective_bandwidths: dict with the following shape:
        {
            band_name_1 : pivot_wavelength_1,
            band_name_2 : pivot_wavelength_2,
            ...
        }
    where:
    pivot_wavelength_1 is in A. Actually units will be in whatever units your wavelength was. 
    
    """

    pivots = {}
    for band in throughput_curves.keys():
        band_wavelength = throughput_curves[band][0]
        band_throughput = throughput_curves[band][1]

        numerator = np.trapezoid(band_wavelength * band_throughput, band_wavelength)
        denominator = np.trapezoid((1.0 / band_wavelength) * band_throughput, band_wavelength)
        lambda_P_sq = numerator / denominator
        lambda_P = np.sqrt(lambda_P_sq)

        pivots[band] = lambda_P

    return pivots


def calculate_zeropoint_flux_AB(throughput_curves, throughput_units = 'Tl'):
    """
    Using a throughput curve, will calculate the zeropoint flux (NOT the zeropoint magnitude or cte!),
    assuming the AB mag system where we assume a monochromatic hypothetical source with a constant 
    flux density F_nu = 3631 * 10^{−23} erg s−1 cm−2 Hz-1. Actually it is 10**(48.6/-2.5) erg s-1 cm-2 Hz-1 exactly.
    

    INPUT:
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_2],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }
    where:
    wavelength_band: wavelength array, in A
    throughput_band: throughput array, int between 0 and 1. Dimensionless

    OPTIONAL INPUT:
    throughput_units: str, either 'Tl' or 'Tv'. I'm going to assume it is Tl, which means that thoughput curve is in function of 
    wavelength. If it is in function of frequency, can do Tv instead. 


    OUTPUT:
    zeropoint_flux: dict with the following shape:
        {
            band_name_1 : zeropoint_flux_1,
            band_name_2 : zeropoint_flux_2,
            ...
        }
    where:
    zeropoint_flux_1 is band specific, and in erg s-1 cm-2 A-1. #CHANGED THIS 
    """

    # Constants
    Fv_AB = 10**(48.6/-2.5) #erg s-1 cm-2 Hz-1
    c = 299_792_458 #m/s
    
    zeropoints = {}
    for band in throughput_curves.keys():
        band_wavelength = throughput_curves[band][0]
        band_throughput = throughput_curves[band][1]

        if throughput_units == 'Tl':
            Fl_AB = Fv_AB / band_wavelength**2 * c * 1e10 #erg s-1 cm-2 A-1
        elif throughput_units == 'Tv':
            Fl_AB = Fv_AB


        # zeropoint = np.trapezoid(Fl_AB * band_throughput, band_wavelength)
        # zeropoints[band] = zeropoint

        numerator = np.trapezoid(Fl_AB * band_throughput * band_wavelength, band_wavelength) # CHANGED THIS 
        denominator = np.trapezoid(band_throughput * band_wavelength, band_wavelength)# CHANGED THIS 
        zeropoints[band] = numerator / denominator# CHANGED THIS 

    return zeropoints




def convolve_sed(template, throughput_curves = {},  plot = False):
    """
    INPUT:
    template: a sed_template object that is going to be reduced
    throughput_curves: dict with the following shape:
        {
            band_name_1 : [wavelength_band_1, throughput_band_2],
            band_name_2 : [wavelength_band_2, throughput_band_2],
            ...
        }

    wavelength_band: wavelength array, in A
    throughput_band: throughput array, int between 0 and 1

    This can be the output of either create_gaussian_throughput_curves() or load_throughput_curves(), or you can create your own.
    If no throughput curves are specified, this will first run create_gaussian_throughput_curves() to create Rubin-like Gaussian
    throughput curves. This works okay-ish, but for actual science it is recommended to actually load in the Rubin (or any other) 
    throughput curves. 

    We basically do: $$F_{\text{band}} = int_0^{infty} F_{lambda}(lambda) cdot T_{lambda}(lambda) cdot d_lambda$$

    OUTPUT:
    reduced_template: a sed_template object with reduced Rubin-like (or any other survey) data, based on the throughput curves.


    NOTES:
    If template has magnitudes attached, we will automatically calculate magnitudes for each band as well. This is done through the 
    band zeropoint fluxes, calculated directly with the throughput curves. 
    """


    # Create Gaussian Rubin-like throughput curves
    if throughput_curves == {}:
        bands_info = {
            'u' : [3724, 500, 0.2], #name : [cen_wav, fwhm, amp]
            'g' : [4807, 1000, 0.5],
            'r' : [6221, 1000, 0.55],
            'i' : [7559, 1000, 0.6],
            'z' : [8680, 1000, 0.6],
            'y' : [9753, 1000, 0.3]
            }
    
        throughput_curves = create_gaussian_throughput_curves(bands_info)


    if len(template.mag) > 0:
        convert_to_mags = True
        zeropoints = calculate_zeropoint_flux_AB(throughput_curves)
        eff_bandwidths = calculate_effective_bandwidth(throughput_curves)
    else:
        convert_to_mags = False

        
    
    fluxes = []
    wavelengths = []
    mags = []

    bands = []

    flux = template.flux
    wavelength = template.wavelength

    for band in throughput_curves.keys():
        band_wavelength = throughput_curves[band][0]
        band_throughput = throughput_curves[band][1]

        pivot_wavelength, flux_obs = single_band_integration(wavelength, flux, band_wavelength, band_throughput)
        
        fluxes.append(flux_obs)
        wavelengths.append(pivot_wavelength)
        bands.append(band)

        if convert_to_mags:
            zeropoint = zeropoints[band]
            eff_bandwidth = eff_bandwidths[band]
            #mag = -2.5* np.log10(flux_obs * eff_bandwidth / zeropoint)
            mag = -2.5* np.log10(flux_obs / zeropoint)
            mags.append(mag)

    # Copy template (so that other data copies), but overwrite flux, wavelength, and magnitude
    reduced_template = copy.deepcopy(template)
    reduced_template.flux = fluxes
    reduced_template.wavelength = wavelengths
    reduced_template.mag = mags
    reduced_template.bands = bands



    if plot:
        buffer = 0.2
        min_lam, max_lam = np.min(reduced_template.wavelength), np.max(reduced_template.wavelength)
        min_lam, max_lam = min_lam*(1-buffer), max_lam*(1+buffer)

        min_flux, max_flux = np.min(reduced_template.flux)*(1-buffer), np.max(reduced_template.flux)*(1+buffer)

                
        #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        if convert_to_mags:
            nfig = 2
        else:
            nfig = 1
            
        plt.figure(figsize = (7,4*nfig))
        
        # Plot the full SED - flux
        plt.subplot(nfig,1,1)

        plt.scatter(reduced_template.wavelength, reduced_template.flux, marker = '+', c = 'k', zorder = 20)
        plt.plot(template.wavelength, template.flux, zorder = -1)
        plt.ylabel('Flux [erg s-1 cm-2 A-1]')
        plt.xlim(min_lam,max_lam)
        plt.ylim(min_flux,max_flux)
    
        
        
        ax2 = plt.gca().twinx()
        plt.sca(ax2)
        
        # Plot the throughput curves
        for i,band in enumerate(throughput_curves.keys()):
            plt.plot(throughput_curves[band][0],throughput_curves[band][1])
        
        plt.ylim(0,1)
        plt.ylabel('Throughput')


        if convert_to_mags:
            plt.subplot(2,1,2)

            delta_mag = np.max(reduced_template.mag) - np.min(reduced_template.mag)
            min_mag, max_mag = np.min(reduced_template.mag) - buffer * delta_mag, np.max(reduced_template.mag) + buffer * delta_mag
    
            plt.scatter(reduced_template.wavelength, reduced_template.mag, marker = '+', c = 'k', zorder = 20)
            plt.plot(template.wavelength, template.mag, zorder = -1)
            plt.ylabel('AB magnitude')
            plt.xlim(min_lam,max_lam)
            plt.ylim(min_mag,max_mag)
            plt.gca().invert_yaxis()
            
            ax3 = plt.gca().twinx()
            plt.sca(ax3)
            
            # Plot the throughput curves
            for i,band in enumerate(throughput_curves.keys()):
                plt.plot(throughput_curves[band][0],throughput_curves[band][1])
            
            plt.ylim(0,1)
            plt.ylabel('Throughput')
        
        plt.show()
            
    
    
    
    return reduced_template


reduce_sed = convolve_sed 







### ============== ###
### Compare models ###
### ============== ###





def bic(sse,n,k):
    '''
    Calculates the Bayesian Information Criterion.
    sse: sum of squared error
    n: number of datapoints
    k: number of parameters in model
    '''

    return k * np.log(n) + n * np.log(sse/n)


def aic(sse,n,k):
    '''
    Akaike Information Criterion
    sse: sum of squared error
    n: number of datapoints
    k: number of parameters in model
    '''

    return 2*k + n * np.log(sse/n)


def f_test(error1,error2,p1,p2,n):
    '''
    Calculates the F-test statistic and the chance that model 2 fits better than model 1. 
    It essentially asks: is the improvement in fit provided by the extra parameters statistically significant? 
    Only works on nested models. From wikipedia:
    "Consider two models, 1 and 2, where model 1 is 'nested' within model 2. Model 1 is the restricted model, 
    and model 2 is the unrestricted one. That is, model 1 has p1 parameters, and model 2 has p2 parameters, 
    where p1 < p2, and for any choice of parameters in model 1, the same regression curve can be achieved by 
    some choice of the parameters of model 2. [...] The model with more parameters will always be able to 
    fit the data at least as well as the model with fewer parameters. Thus typically model 2 will give a 
    better (i.e. lower error) fit to the data than model 1. But one often wants to determine whether model 
    2 gives a significantly better fit to the data. One approach to this problem is to use an F-test."

    The null hypothesis is that model 2 does not provide a significantly better fit than model 1

    INPUTS:
    error1: sse of model 1
    error2: sse of model 2
    p1: number of parameters of model 1
    p2: number of parameters of model 2
    n: number of datapoints that were fit

    OUTPUTS:
    pvalue of test. If lower than a predefined threshold (e.g., 0.05), than model 2 (complex model) fits better than model 1. 
    If higher, keep model 1 (simpler model). 


    Useful links:
    https://en.wikipedia.org/wiki/F-test
    https://academic.oup.com/mnras/article/523/3/4029/7190641 - James' work where he 
    
    '''
    
    f_stat = (error1 - error2) / (error2) * (n - p2) / (p2 - p1)
    pval = stats.f.sf(f_stat, p2-p1, n-p2)

    return pval








### ===== ###
### Varia ###
### ===== ###


def calculate_error_v0(real, data, sigma = None):
    '''
    Calculate squared error
    '''
    if sigma is None:
        return np.sum((real - data)**2)
    else: 
        if len(sigma) == 0:
            return np.sum((real - data)**2)
        else:
            return np.sum(((real - data)/sigma)**2)


def check_pickled_size(obj, name="Object"):
    pickled_data = pickle.dumps(obj)
    size_mb = len(pickled_data) / (1024**2)
    print(f"{name} size: {size_mb:.2f} MB")


def calculate_error(real, data, sigma=None):
    '''
    Works with large arrays
    '''
    data = np.atleast_2d(data)
    diff = real - data
    
    if sigma is not None and len(sigma) > 0:
        diff = diff / sigma
        
    err = np.sum(diff**2, axis=1)
    return err[0] if len(err) == 1 else err # Return a number if input was only one 


def get_precision(x):
        s = str(x)
        if '.' not in s:
            return 0
        return len(s.split('.')[1].rstrip('0'))

def safe_arange(start, stop, step):
    '''
    Analogue to np.arange, but it fixes the numerical errors often induced by np.arange. 
    E.g., doing: 
    lst = np.arange(0,1,0.1)
    print(lst[3])
    will output 0.30000000000000004.

    Here, we calculate precision based on the stepsize, and round to it.

    safe_arange also makes sure the stop is included. I.e., it is inclusive.
    
    '''
   
    p_start = get_precision(start)
    p_stop = get_precision(stop)
    p_step = get_precision(step)
    precision = max(p_start, p_stop, p_step) + 1

    return np.round(np.arange(start, stop + (step * 0.5), step), precision)


def batches(lst,n_batch = 100):
    '''
    Divides the list into batches.
    Can handle non-perfect cases where x%n_batches != 0
    '''
    new_lst = []
    i = 0
    while(len(lst)-i > n_batch):
        new_lst.append(lst[i:i+n_batch])
        i+=n_batch
    
    new_lst.append(lst[i:])#append final bit
    return new_lst



def denannify(lst):
    '''
    Will remove all np.nan's from a list
    '''
    arr = np.array(lst)
    lst = list(arr[~np.isnan(arr)])

    return lst

def deinfify(lst):
    '''
    Will remove all np.if's from a list
    '''
    arr = np.array(lst)
    lst = list(arr[~np.isinf(arr)])

    return lst


def intersection(L):
    '''
    Calculates intersection of multiple lists.
    L must be list of lists of integers. 
    '''
    
    if len(L) == 1:
        return L[0]
    
    else:
        L_temp = L[:-2]
        L_temp.append(list(set(L[-1]) & set(L[-2])))
        return intersection(L_temp)

def union(L):
    '''
    Will create union of multiple lists.
    L must be a list of lists of integers.
    Will also remove duplicates
    '''
    
    if len(L) == 1:
        return L[0]
    
    else:
        L_temp = L[:-2]
        L_temp.append(list(set(list(L[-1]) + list(L[-2]))))
        return union(L_temp)
    


def nCr(n,r):
    '''
    number of combinations
    '''
    f = math.factorial
    return f(n) / f(r) / f(n-r)


def plot_diagonal(ax = [], c = 'k', ls = '--', lw = 1):
    '''
    ax can be the axis over which to plot the diagonal. If empty,
    will run plt.gca(), so use current axis.
    '''
    if ax == []:
        ax = plt.gca()
        
    xlims,ylims = ax.get_xlim(), ax.get_ylim()   
    m = np.min([xlims[0], ylims[0]])
    M = np.max([xlims[1], ylims[1]])
    
    ax.plot([m,M], [m,M], c=c,ls=ls, lw=lw) #if something is broken, change this line back to plt.plot()



class stopwatch:
    '''
    Make a stopwatch class that makes timing stuff and debugging easier.
    '''
    def __init__(self):
        # Start when initialised
        self.start_time = time.time()
        self.lap_time = time.time()
        
        
    def start(self):
        # Start again
        self.start_time = time.time()
        self.lap_time = time.time()

    def reset(self):
        # Wrapper around start
        self.start()
        
    
    def read(self, output = True):
        # Read how much time has passed since start
    
        time_passed = time.time() - self.start_time 
        
        if output:
            self.output(time_passed)
            
            
    def lap(self, output = True):
        laptime_passed = time.time() - self.lap_time 
        if output:
            self.output(laptime_passed)
        
        self.lap_time = time.time()

                
    def output(self, val, precision = 3):
        # Print out time
        print(f'{np.round(val, precision)} seconds have passed.')


    def register_processes(self, processes):
        self.processes = {}
        for p in processes:
            self.processes[p] = []

    def log_process(self, process):
        time_passed = time.time() - self.start_time 
        self.processes[process].append(time_passed)


    def output_processes(self):
        for p in self.processes.keys():
            mean = np.mean(self.processes[p])
            std = np.std(self.processes[p])
            total = np.sum(self.processes[p])
            n = len(self.processes[p])
            
            print(f'{p} - mean time = {mean:.2f} +- {std:.2f} sec; total time = {total:.2f} sec; n = {n}')
        
        print()


    def plot_processes(self):


        # Plot total share of runtime
        means = [np.sum(self.processes[p]) for p in self.processes.keys()]
        total = np.sum(means)
        labels = self.processes.keys()
        plt.figure(figsize = (3,3))
        plt.pie(means/total, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.show()

        # Plot individual times

        n_plots = 0
        for i,p in enumerate(self.processes.keys()):
            if len(self.processes[p]) > 1:
                n_plots +=1

        if n_plots>0:
            plt.figure(figsize = (5, 2*n_plots))
            for i,p in enumerate(self.processes.keys()):
                if len(self.processes[p]) == 1:
                    continue
                plt.subplot(len(self.processes), 1, i+1)
                plt.plot(self.processes[p])
                plt.title(p)
                plt.ylabel('Time [s]')
                plt.xlabel('Iteration')
            
            plt.tight_layout()
            plt.show()