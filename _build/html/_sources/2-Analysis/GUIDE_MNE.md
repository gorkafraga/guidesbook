# MNE toolbox 
MNE is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.

## Warning 
Be careful with assignments in python... 
![image](https://user-images.githubusercontent.com/13642762/210753779-8db80c17-139f-42c6-b9d0-a9d21da77059.png)


## MNE Report 
The mne.Report functions are great to generate HTML with a summary of your data, plots and even code chunks. The document is interactive, you can choose to hide or expand some images, and some of the figures (e.g., topographies with slider) are interactive.  
Documentation: https://mne.tools/dev/auto_tutorials/intro/70_report.html#sphx-glr-auto-tutorials-intro-70-report-py

### Example of a single-subject report
In this example we have epoched data (mne epoched object saved in .fif files). One file per subject, containing the epochs for the different conditions. 
We want to make a report html file for each subject containing: 

* A figure with Event description (event labels in time)
* Number of epochs for each condition  
* Visualizations of the *evoked* data (average of conditions): 
  - ERP waveforms and global field power
  - ERP images (x-time , y-epoch, colormap-amplitude)
  - Topographies with interactive slider 
  - Time frequency power
#### Read data  
   ` epochs = mne.read_epochs(glob(dirinput + 's04_DiN_epoched_ICrem.fif')[0])  # glob searches file name pattern in dirinput`
#### Initialize the report 
     report = mne.Report(title='My example report')
      # Add epochs will automatically add an ERP image but this will average across all conditions 
      report.add_epochs(epochs=epochs, title='Epochs', psd=False)  #  PSD plot can be turned on 
#### Create evoked objects 
      #To get the ERPs per condition we need to compute the averages 

#### Add elements to the report 
      (...)

#### Write to .html file

## Quality assesment 
 
 - Check out this code to use PSD plots for spotting bad channels  
 https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/results/demos/plot_psd.py

## Multivariate pattern analysis 

* Example 1. Simple to follow example of classification. https://natmeg.se/mne_multivariate/mne_multivariate.html

* Example 2. MVPA in infant data. https://github.com/BayetLab/infant-EEG-MVPA-tutorial

