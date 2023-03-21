# Jupyter book
Create a publication-quality book and documentation made from a bunch of markdown files
Documentation: https://jupyterbook.org/en/stable/intro.html 

## Basic workflow
We can create a book from a collection of markdown files with guidelines. 
See link: https://jupyterbook.org/en/stable/start/your-first-book.html

Basically we need to: 
- Install Jupyter-book in our Conda environment
- Create a folder with
 	* A configuration file (_config.yml)
	* A table of contents file (_toc.yml)
	* Your book’s content (e.g., .md files) 
- From our conda environment type: ````jupyter book build mybookname/````

## Troubles running Jupyter-book
Check if you type jupyter-book in your environment and you receive some errors instead of the jupyter book options. 
It seems there are some compatibility problems with latest Python versions. Then you can create an environment with python=3.7  for which Jupyter book seems to have been tested. `conda create mybook-env python=3.7` 

## Publish to github pages
- Clone our github repository (must be publi and we need to select the gh-pages branch in the repository settings/branch)
- Install ghp-import in our conda environment: ```` pip install ghp-import```` 
- Then in our local cloned repository we make changes in the book files, build the book:
```python 
jupyter book build mybookname
```
- In conda prompt we go to the folder containing the book and type:
```python
ghp-import -n -p -f _build/html
```
- Then we push the changes to the origin and it should become accessible in our github pages: *mygithubpagename.github.io/myBookRepo*

## Rendering mermaid diagrams in your book
The package 'mermaid' allows pretty flowcharts in md files that you can view in github. However the charts won't be displayed in the book unless you do:

* ```pip install sphinxcontrib-mermaid```

* In your .md file with the flow chart use curly brackets to specify mermaid. For example:  
   ```` 
	````{mermaid}
	stateDiagram-v2
       [*] --> Still
       Still --> [*]
       Still --> Moving
       Moving --> Still
       Moving --> Crash
       Crash --> [*]````
   ````
* In your book's _config.yml file include these lines:
```` 
sphinx:
  extra_extensions:
  - sphinxcontrib.mermaid
````

See: https://blog.ouseful.info/2021/11/02/previewing-sphinx-and-jupyter-book-rendered-mermaid-and-wavedrom-diagrams-in-vs-code/
For editing charts with live preview go to https://mermaid.live/ (you can copy then the code into your md file) 
 
