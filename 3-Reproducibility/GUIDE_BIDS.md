# Brain Imaging Data Structure - BIDS
BIDS is an attempt to define some simple ways of organizing our data from different neuroimaging modalities so that it can be shared. 

Visit the extensive documentation at: 
https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html

## Meta data
Language-agnostic files with relevant information. These are proposed in two formats: 
  - **.json**. Java Script Object Notation. Human-readable text to store metadata. JSON online editor: https://jsoneditoronline.org/
 
  - **.tsv**. Tab-delimited files

- 

## File names and folder organization

- The MNE toolbox has some functions to help following a BIDS-compatible file names: https://mne.tools/mne-bids/stable/auto_examples/create_bids_folder.html
- Details about EEG specific files in:  https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html

## Examples
Get oriented with some examples from BIDS site: https://github.com/bids-standard/bids-examples/tree/master/




##  Automated pipelines 
### DISCOVER-EEG

An open, fully automated EEG pipeline for biomarker discovery in clinical neuroscience
https://www.biorxiv.org/content/10.1101/2023.01.20.524897v1
