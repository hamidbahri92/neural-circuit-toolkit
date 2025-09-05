# Neural Circuit Discovery Toolkit

Research implementation for exploring mechanistic interpretability concepts in neural networks.

## Overview
Demonstrates circuit discovery techniques using activation patching and differential analysis on mock neural networks for educational and research purposes.

## Features
- Behavioral circuit discovery from activation patterns
- Content-addressed blob storage  
- Safety checks and rollback mechanisms
- Transactional weight modifications

## Requirements
```bash
pip install numpy scipy matplotlib networkx
Quick Start
bashexport ATLAS_BLOB_DIR="blobs"
python3 -c "from atlas.cli.demo import run_demo; run_demo()"
Project Structure

atlas/ - Core implementation modules
blobs/ - Neural circuit data files
docs/ - Comprehensive documentation
RUN.md - Quick start guide

Disclaimer
Educational/research project exploring established interpretability techniques. Currently implemented with mock models for demonstration purposes.
