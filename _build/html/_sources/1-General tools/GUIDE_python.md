# Python - setup  
This is a  very quick guide for setting up a python environment. 
We are likely to have multiple instances of python in our PC. We need to ensure we know which one we are using.

There are many ways. This is just one I found simple enough. It requires:
 - Anaconda3 (python environment control)
 - Spyder (IDE)

We can download Anaconda 3, from there create environments with a python version and different modules (packages). Then we launch a light IDE like Spyder (similar to Rstudio) from a given environment. 
See extensive documentation: https://google.github.io/styleguide/pyguide.html  

## Basic setup
Main documentation: https://docs.spyder-ide.org/current/installation.html  

* **Download and install Anaconda3** 
 (If using WINDOWS: Set environment variables. Add to PATH: Anaconda3, Anaconda3/Scripts and Anaconda3/Library/bin folder ) 

* **Check installation** : Open the command prompt and type something like: ```conda info --envs```

* **Create an environment** with some modules, a version of python etc. In Anaconda prompt type:   
  ``` conda create –n spyder-env –y```   or 
  ```conda create -n spyder-env spyder spyder-kernels numpy scipy pandas matplotlib sympy cython``` (example with some basic packages) 
  The environment will be saved in the folder 'Anaconda3/envs'
  We can have multiple environments
  
* **Activate** environment in the prompt  
```conda activate spyder-env ```
Now the console prompt should have changed indicating you are within the created environment spyder-env. 
Whatever we do know will affect only that environment.\
WARNING: Anaconda3 comes with some *base* environment. Do NOT install stuff in there. Create your own environments, activate them and only within those environment mess around with conda or PIP installation of modules. 

* **Launch spyder** 
Type 'Spyder' to launch spyder from wihin this environment.

* **Installing modules** 
 You can also install modules like scikit-learn by typing this either in the console or in Spyder (but must be within the enviornment): 
    ```conda install spyder-kernels scikit-learn seaborn```   

## Regular usage 
Once we have our environment setup the usual workflow will involve: 
- Open command prompt
- Activate the environment you want to work on 
- Type 'spyder' and keep on working on Spyder IDE 
 
## Exporting environment across platforms 
For example iff you created an environment in one computer and want to run those analysis in a remote desktop with a different operating system. You can export the environment and import it in the other computer. But if the OS are different the dependencies might change (but you still want to have the same version of python and packages if possible). 
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment 
* Create an environment: specify python version 
* Export your environment:  e.g., conda env export > spyder-env.yml --from-history
* Import (...)  

## Mixed tips

## keywords:  
pip = package manager works from any environment  
Conda = package and environment  
Spyder = IDE for python (like Rstudio) 
Anaconda =  Python distributor with multiple programs 
https://medium.datadriveninvestor.com/what-is-pip-conda-anaconda-spyder-jupyter-notebook-pycharm-pandas-tensorflow-and-django-36d54778d85c 

## Troubleshooting  
> Issues during installation included problems with spyder recognizing the right python version or not recognizing the 'conda' command (e.g., because Anaconda folders were not added in the OS path). PIP is another package manager. Recommended to stick to conda for installing packages whenever possible.  

> Spyder is not showing plotly plots? Set up the default renderer:  
   import plotly.io as io 
   io.renderers.default='browser' 

> Instead of Conda, there is also Mamba which seems to be more efficient solver than Conda..  
 

 
