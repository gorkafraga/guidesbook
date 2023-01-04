# PsychoPy 

Open software for stimuli presentation. Runs on Python, uses OpenGL. It can convert the code to JS (using PsychoJS) to launch the experiment in Pavlovia.
Documentation: https://www.psychopy.org/PsychoPyManual.pdf 

## Main components 

Use PsychoPy Builder to modify the scripts. It is a GUI that also allows to insert code chunks (e.g., for custom randomization or loops). The script cannot be modified outside the GUI. For the GUI, xls files with list of stimuli and design parameters are used. Log files can be customized. 
Note: It is advised to use the builder for online experiments as it will generate the JS version of the code !  

When you run an experiment PsychoPy Builder translates the Builder experiment into python code and then executes it. You can always see the generated code and run it into a Python program, but you cannot go from code back to a Builder representation ! 
PsychoPy Coder is a basic code editor. It has an output window and Demo menu with examples.  


## Experiments 

Experiments are .psyexp files that you can open in Builder.  
They have a 'Routines' panel with tabs for the different components and a 'Flow' panel with an overview of the loops and routines. 
 
Each component will typically has code, text (e.g., displayed instructions or feedback) and responses assigned. 
Use control + arrow keys to navigate through tabs in the 'routines' panel . Clicking on the blocks in the 'Flow' panel view will take you to the corresponding tab in 'routines' panel.  

## Interaction with Openvibe 

In our experiment there are blocks with stimuli presentation triggered by EEG signal, preprocessed in real time with Openvibe. This operation is captured in the  block 'signal check'.   

 
