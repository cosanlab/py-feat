0. Install packages
```
pip install jupyter-book
pip install ghp-import
```

1. Make changes. 
Add notebooks or markdowns to the `notebooks/content/` directory.
Add images to the `notebooks/content/images` directory. 

2. Update Table of content in `notebooks/_toc.yml` file

3. Build html 
```
jupyter-book build notebooks
```

4. Add & commit changes
```
git add . 
git commit -m "added changes"
```

5. Upload to gh-pages
```
ghp-import -n -p -f notebooks/_build/html
```