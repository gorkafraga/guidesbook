# EEG features
## Time-frequency analysis
Here we study the changes in power at the different frequencies over time. This contrasts with a 'static' spectral analysis where we examine the spectral distribution of an entire time period (e.g., using a Fast Fourier Transform-FFT). To capture the temporal dynamics of frequencies we repeatedly perform the dot product of the EEG signal by a wavelet that is sliding/shifting overtime. This way we obtain the power distribution at each time point. Besides complex wavelet convolution, other commonly used methods are the short-time fast Fourier transform and filter-Hilbert (not covered in this section). 

This approach allows, among others, the analysis of Event Related Spectral Perturbations (**ERSP**) and Inter-trial Phase Coherence (**ITC**).

EEGlab team provides a detailed [video lecture on EEG time-frequency analysis](https://www.youtube.com/watch?v=eUFf5eFpdLg&t=240s&ab_channel=EEGLAB)

### Wavelets
A primary method for time-frequency analysis is the wavelet convolution. There are different kinds of wavelets and they have two main properties: (1) they tapper down to zero (at the beginning and the end) and (2) they integrates to zero (positive and negative parts would average out to zero). A frequently used wavelet is the Morlet wavelet, which can be created by point-wise multiplication of a Sine wave and Gaussian function (Gaussian-modulated sinusoid)


A real-valued Morlet (convolution with EEG signal will give us *amplitude* values at each time point): 


<img src='https://user-images.githubusercontent.com/13642762/208658414-236ae8f6-cc09-4d18-998d-e22901ed2d37.gif' height = '150px' width = '250px'> <img src='https://user-images.githubusercontent.com/13642762/208658801-befbbc18-d9d2-4735-b438-19115ea5260f.png' height = '150px' width = '250px'> <img src = 'https://user-images.githubusercontent.com/13642762/208659831-43075e06-c3a4-471a-8c6d-30dd869e0730.png' height='140px' width='250px' >

A complex-valued Morlet wavelet (the result of the convolution with the EEG signal will give us a complex value for each time point, allowing to extract instantaneous *power* as well as *phase angle* information): 


<img src='https://user-images.githubusercontent.com/13642762/208878213-6160bda5-0c51-48de-aa82-78f2976da054.gif' height='400px' width='600px'>


There are several basic considerations when running a time frequency analysis:
- Sampling rate. Typically downsampling to to 250 Hz is done to reduce file size and computation time
- Frequencies to analyse. Depending on hypotheses. But it is recommended that the data have at least 3 full cycles of the lowest frequency to analyze, and higher frequency must be below Nyquist frequency (1/2 of sampling rate)

A key parameter of the Morlet that will have an impact in *temporal and spectral precision* of our analysis:
- *Width* of the wavelet or number of **cycles**: more cycles increase frequency precision (we can capture more frequency bins) and fewer cycles increases temporal precision (we can capture more data point). We can alslo use fewer cycles at lower frequenceis and gradually increase the number of cycles as frequency increases, thus having more temporal precision at lower frequencies and more frequency precision at higher frequencies. 
- The time-frequency trade off can also be described in terms of *full-with at half-maximum*(**FWHM**) in the time and/or frequency domain. See [Cohen,2019](https://doi.org/10.1016/j.neuroimage.2019.05.048) for an article arguing about the relevance of this term. The FWHM is the distance in time between 50 % gain before the peak to 50% gain after the peak 




<img src='https://user-images.githubusercontent.com/13642762/208679988-6fd496fc-55ba-4847-a65d-bfed329f2dc1.png' height='200px' width='300px'>

<sub>Cohen 2019</sub>



Illustration of wavelet convolution:

<img src='https://user-images.githubusercontent.com/13642762/208693831-a8c77e31-8413-4590-bcb1-f72e5965e04a.png'  height = '310px' width ='300px'>
<sub> [Morales and Bowers, 2022](https://www.sciencedirect.com/science/article/pii/S1878929322000111)</sub>


An example of TF results visualization (scalogram):

 <img src='https://user-images.githubusercontent.com/13642762/208690235-41937eac-b838-4cd4-8668-8f8aee54e54b.png' height = '210px' width ='300px'>

<sub> Morales and Bowers, 2022</sub>

See a recent comprehensive review on [Morales and Bowers, 2022](https://www.sciencedirect.com/science/article/pii/S1878929322000111)

#### The 'cone' of influence
Wavelet coefficients are computed by convolving the wavelet kernel with the time series. Similarly to any convolution of signals, there is zero padding at the edges of the time series and therefore the wavelet coefficients are weaker at the beginning and end of the time series. In addition we can have an event we are not interested in, for instance, we may look at prestimulus interval in a larger epoch containing post stimulus activity and want to avoid the times immediately before 0 where stimulus. 

The cone of influence [see COI-matlab documentation](https://ch.mathworks.com/help/wavelet/ug/boundary-effects-and-the-cone-of-influence.html), are areas in the scalogram potentially affected by edge-effect artifacts. These arise from areas where the stretched wavelets extend beyond the edges of the observation interval.  These are referred as edge effects in continuous wavelet transform (CWT). Some analyses, for example, compute the COI for the time windows of interest and only take the wavelet coefficients inside the respective COI.  

<img src='https://user-images.githubusercontent.com/13642762/208690651-3e6f1451-cab3-46c5-9f69-2f3daa1a2fd2.png' height = '210px' width ='300px'>

<sub>An example of COI from [Dash et al., 2020](https://www.frontiersin.org/articles/10.3389/fnins.2020.00290/full) </sub>


There is a further discussion on this on the EEGlab Wiki: https://sccn.ucsd.edu/wiki/Makoto's_preprocessing_pipeline

### Preprocessing

Good summary of common artifacts: 
https://pressrelease.brainproducts.com/eeg-artifacts-handling-in-analyzer/
