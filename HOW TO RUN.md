## How to Run the Project

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the pipeline

```powershell
python run_ingest.py
python run_clean.py
python run_annotate.py
python run_readability.py
python run_stylometry.py
```

### 4. Render the final report

```powershell
quarto render report.qmd
```

The final report is generated as:

```text
report.html
```

---

## Technical Environment

The project was developed and tested on:

* Windows
* Python virtual environment
* Apache Spark / PySpark
* Spark NLP
* Quarto
* Plotly visualizations

---

## Spark Engineering Notes

The project uses Spark DataFrames and staged Parquet outputs to make the workflow reproducible.

Important engineering choices include:

* chunking long novels before Spark NLP annotation
* saving intermediate outputs as Parquet
* separating pipeline stages into reusable scripts
* avoiding repeated recomputation of expensive NLP stages
* comparing Parquet and ORC storage formats

Running Spark locally on Windows introduced some practical issues, especially stale JVM processes and Spark port conflicts. These were handled during development by carefully restarting Spark sessions and clearing stale Java processes when necessary.

---

## Limitations

The results should be interpreted as exploratory rather than statistically conclusive.

Main limitations:

* Type-token ratio is sensitive to text length.
* Dialogue detection is incomplete for em-dash dialogue.
* Spark NLP is computationally heavy on a local Windows machine.
* More detailed Spark UI profiling could be added.
* More statistical testing would be needed for stronger literary conclusions.

---

## Possible Improvements

With more time, the project could be improved by:

* preserving line-position information for better dialogue detection
* adding more systematic Spark UI profiling
* measuring shuffles, caching, and execution times more precisely
* improving cross-language comparability
* adding statistical tests for author and movement comparisons
* expanding the corpus with more novels and authors

---

## Final Report

The final rendered report is available in:

```text
report.html
```

The source Quarto file is:

```text
report.qmd
```

````

After saving it, push the update:

```powershell
git add README.md
git commit -m "Add project README"
git push
````
