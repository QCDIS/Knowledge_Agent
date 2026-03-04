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
then activate the environment

Setup the Envoronment Variables

```
source environment_setup.sh
```


Start streamlit to start the dialogue application
 
```
streamlit run streamlit_ui.py
```

Start streamlit using nohup
```
nohup streamlit run /home/ntislam/Development/Knowledge_Agent/dev/streamlit_ui.py --server.port=8501 > streamlit.log 2>&1ps aux | grep streamlit
```


