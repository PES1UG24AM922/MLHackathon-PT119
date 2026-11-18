# PathSnap exploratory data analysis

[Open the executed notebook](PathSnap_EDA.ipynb) to see dataset validation, sampling and connectivity diagnostics, and the train-only GPS error and candidate radius analysis. The [generated outputs](PathSnap_EDA_outputs/) include CSV tables and plots. Download and open the [Folium map](PathSnap_EDA_outputs/06_train_trip_00000001_map.html) in a browser to view the training trip and its reference route.

## Reproduce the analysis

Place the organizer-provided `Student folder.zip` in the repository root or in `eda/`. From the repository root, start JupyterLab with the analysis dependencies installed.

```sh
python -m pip install jupyterlab numpy pandas scipy matplotlib folium
cd eda
python -m jupyterlab PathSnap_EDA.ipynb
```

Run all notebook cells to regenerate `PathSnap_EDA_outputs/`. The notebook reads the archive directly and reports `train` and `student_test` separately.
