# Brain Imaging Data Structure - BIDS
BIDS is an attempt to define some simple ways of organizing our data from different neuroimaging modalities so that it can be shared. 

Visit the extensive documentation at: 
https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html

## Meta data
Language-agnostic files with relevant information.https://bids-standard.github.io/bids-starter-kit/folders_and_files/metadata.html. 
Two types of files are proposed
### JSON 
.json Java Script Object Notation. Human-readable text to store metadata. JSON online editor: https://jsoneditoronline.org/

  ``` python
  {
    "key": "value",
    "key2": "value2",
    "key3": {
        "subkey1": "subvalue1"
    }
}
```
Read in Python 
``` python
import json
with open('myfile.json', 'r') as ff:
    data = json.load(ff)
 ```
 Write in Python 
 
 ``` python
import json
data = {'field1': 'value1', 'field2': 3, 'field3': 'field3'}
with open('my_output_file.json', 'w') as ff:
    json.dump(data, ff)
```
     
### Tab-delimited 
.tsv Tab-delimited files
 For example , they can contain events, channel locations, etc. 

Read in Python 

``` python
import pandas as pd
pd.read_csv('./ds001/participants.tsv', delimiter='\t')
```
 Write in Python 
 
 ``` python
df.to_csv('my_new_file.tsv', sep='\t')
```

## File names and folder organization
- The MNE toolbox has some functions to help following a BIDS-compatible file names:
  https://mne.tools/mne-bids/stable/auto_examples/create_bids_folder.html
- Details about EEG specific files in:  
  https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html

## Examples
Get oriented with more examples from BIDS site: https://github.com/bids-standard/bids-examples/tree/master/



##  Automated pipelines 
### DISCOVER-EEG

An open, fully automated EEG pipeline for biomarker discovery in clinical neuroscience
https://www.biorxiv.org/content/10.1101/2023.01.20.524897v1
