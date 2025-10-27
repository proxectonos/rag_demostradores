# rag_demostradores

A collection of demonstrators and evaluation tools for Retrieval-Augmented Generation (RAG) systems. This project provides backend services, demonstration interfaces, and evaluation modules to experiment with and assess RAG pipelines.

## Project Structure

- **backend/**: Backend services and APIs for RAG workflows.
- **demostrador/**: Frontend or interactive demonstration interfaces.
- **evaluator/**: Scripts and modules for evaluating RAG systems and metrics.
- **ragas/**: Integration and utilities for RAGAS (Retrieval-Augmented Generation Assessment Suite).

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   ```
2. Set up the Python environment (recommended: use a virtual environment):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   (Check for requirements.txt in relevant subfolders if needed)

## Usage

- **Backend**: See `backend/` for API launch instructions.
- **Demonstrator**: Run demonstration interfaces from `demostrador/`.
- **Evaluation**: Use scripts in `evaluator/` to assess RAG models.
- **RAGAS**: Integrate and run RAGAS tools from `ragas/`.

Refer to subfolder READMEs for detailed instructions.

## Configuration

- Environment variables and config files may be required (see `load_env.sh` and configs in subfolders).

## Contributing

See [CONTRIBUTING.md](../mergekit/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the terms of the LICENSE file in this directory.

## Contact

For support or questions, please open an issue or contact the maintainers.
