# Knowledge_Agent

run

Export env to YAML (for sharing):
```
conda env export > environment.yml
```

Recreate env from YAML:

```
conda env create -f environment.yml
```

Setup the Envoronment Variables

```
source environment_setup.sh
```


Start streamlit to start the dialogue application
 
```
streamlit run streamlit_ui.py
```


