# Jupyter book
Create a publication-quality book and documentation 
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

## Publish to github pages
- Clone our github repository (must be publi and we need to select the gh-pages branch in the repository settings/branch)
- Install ghp-import in our conda environment: ```` pip install ghp-import```` 
- Then in our local cloned repository we make changes in the book files, build the book (````jupyter book build mybookname````)
- In conda prompt we go to the folder containing the book and type:````ghp-import -n -p -f _build/html````
- Then we push the changes to the origin and it should become accessible in our github pages: *mygithubpagename.github.io/myBookRepo*

