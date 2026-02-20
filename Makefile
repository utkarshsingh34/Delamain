.PHONY: data train-forecast train-rl eval api dashboard test clean

# Phase 1: Download and preprocess data
data:
	python data/scripts/download.py
	python data/scripts/preprocess.py
	python data/scripts/eda.py

# Phase 3: Train demand forecaster
train-forecast:
	python forecasting/train.py

# Phase 4: Train RL agent
train-rl:
	python optimizer/rl_agent.py

# Phase 4: Run evaluation benchmark
eval:
	python eval/compare.py

# Phase 5: Start FastAPI server
api:
	uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Phase 7: Start Streamlit dashboard
dashboard:
	streamlit run interface/app.py

# Run tests
test:
	pytest tests/ -v

# Clean generated artifacts
clean:
	rm -rf data/raw/* data/processed/*
	rm -rf models/*
	rm -rf cache/*
	rm -rf logs/*
	rm -rf eval/results/*
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info
