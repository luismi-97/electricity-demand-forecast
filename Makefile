.DEFAULT_GOAL := help
PY ?= python

help:           ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:        ## install the package with dev, nn, api and dashboard extras
	$(PY) -m pip install -e ".[dev,nn,api,dashboard]"

test:           ## run the unit tests
	$(PY) -m pytest -q

data:           ## download load and weather (resumes from the last stored timestamp)
	$(PY) -m edf.ingest smard
	$(PY) -m edf.ingest weather

official:       ## import the official TSO forecast: make official CSV=data/raw/smard_forecast.csv
	$(PY) -m edf.ingest official --csv $(CSV)

validate:       ## validation-period experiments (notebooks 01-04); the final test is NOT included
	$(PY) notebooks/01_eda.py
	$(PY) notebooks/02_baselines.py
	$(PY) notebooks/03_gbm.py
	$(PY) notebooks/04_nn_intervals_official.py

final-test:     ## ONE-SHOT evaluation on the locked test set (commit + tag first!)
	$(PY) notebooks/05_final_test.py --confirm-final-test

report:         ## refresh reports/RESULTS.md and the results block of the README from the CSVs
	$(PY) -m edf.report

train:          ## train the frozen configuration on all data and save the artifacts (models/v1)
	$(PY) -m edf.train_final

serve:          ## run the API locally on http://localhost:8000 (docs at /docs)
	uvicorn edf.api:app --reload --port 8000

dashboard:      ## run the Streamlit dashboard on http://localhost:8501
	streamlit run dashboard/app.py

docker-build:   ## build the API image (add WITH_NN=1 to include TensorFlow)
	docker build -t edf-api $(if $(WITH_NN),--build-arg WITH_NN=1,) .

docker-run:     ## run the API image with data/ and models/ mounted
	docker run --rm -p 8000:8000 -v $(CURDIR)/data:/app/data -v $(CURDIR)/models:/app/models edf-api

.PHONY: help install test data official validate final-test report train serve dashboard docker-build docker-run
