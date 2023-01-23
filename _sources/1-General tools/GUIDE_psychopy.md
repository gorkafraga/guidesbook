# PsychoPy 
Open software for stimuli presentation. Runs on Python, uses OpenGL. It can convert the code to JS (using PsychoJS) to launch the experiment in Pavlovia.
Documentation:[ PsychoPy Documentation ](https://psychopy.org/gettingStarted.html)

Note: besides generalities, this doc focuses on an existing experiment from R.Becker (DiN project, T.Houweling analysis) that needs to be adapted. 

## Main components 
#### **PsychoPy Builder**.
Use builder to modify the scripts. It is a GUI that also allows to insert code chunks (e.g., for custom randomization or loops). The script cannot be modified outside the GUI. For the GUI, xls files with list of stimuli and design parameters are used. Log files can be customized. 
Note: It is advised to use the builder for online experiments as it will generate the JS version of the code, even for advanced programmers. 

When you run an experiment PsychoPy Builder translates the Builder experiment into python code and then executes it. You can always see the generated code and run it into a Python program, ```but you cannot go from code back to a Builder representation```! 

#### **PsychoPy Coder**. 
Basic code editor. It has an output window and Demo menu with examples.  But remember is a one-way street from Builder to Code. 


## Experiments 

Experiments are **.psyexp** files that you can open in Builder.  

- *Routines* panel has tabs for the different components
- *Flow* panel shows an overview of the loops and routines. 
 
Each component will typically has code, text (e.g., displayed instructions or feedback) and responses assigned. 
Use control + arrow keys to navigate through tabs in the 'routines' panel . Clicking on the blocks in the 'Flow' panel view will take you to the corresponding tab in 'routines' panel.  

## Interaction with Openvibe 
In our experiment there are blocks with stimuli presentation triggered by EEG signal, preprocessed in real time with Openvibe. This operation is captured in the  block 'signal check'.   

## Troubleshooting
#### Connecting with OpenVibe outside the lab 

- To run a real time EEG outside the lab you need some 'fake' eeg data in Openvibe. You will need to first *run the openvibe script* (some debugging script running simulated data) until you see it *streaming simulated data*.  

- Then open Psychopy and run your experiment

#### Issues with version control 
The Psychopy version control can cause problems:  you can trying leaving the option blank in the psychopy settings, instead of specifying a version 

#### Problems with python packages 
I had some problems with python packages that were needed for the experiment but were not installed in the LABS's PC. e.g., I needed xlsxwrite (error will be prompted in the log when running the experiment). You can download the package and copying the content into the `libs` folder within your psychopy folder (usually in Windows in the 'Program files' directory). When the psychopy scripts does `'import packagename'` it should then be able to load the package. 
 
#### Permissions
Make sure the experiment folder, where output files will be written, has the appropriate read/write permissions

#### JS warnings in runner log
In R.Becker's scripts there may be some warnings related to JavaScript. Their source is not yet identified but the experiment was running well (the warnings were also there during previous data collection)

