# LLM From Scratch

A framework-free project structure to build and study an LLM end-to-end.

## Project Goals
- Learn and implement core math/NLP/modeling concepts from first principles
- Build small, testable modules phase-by-phase
- Keep experiments reproducible

## Quick Start
1. Create a virtual environment
2. Install dependencies from `requirements.txt`
3. Run `python scripts/run_phase1_demo.py`
4. Open `web/roadmap.html` in a browser to view the roadmap UI

## Suggested Workflow
- Start from `src/math` and `src/nlp`
- Build the model in `src/model`
- Add training loops in `src/training`
- Keep experiments in `experiments/`
- Save checkpoints in `models/`
