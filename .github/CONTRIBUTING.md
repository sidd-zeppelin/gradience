# Contributing to Gradience

First off, thank you for your interest in contributing to Gradience! Whether you're fixing a bug, improving documentation, proposing a new feature, or implementing a new tensor operation, your contributions are greatly appreciated.

Gradience aims to be an educational, well-engineered deep learning framework built from first principles. We value clean code, thoughtful discussions, and high-quality documentation.

---

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/<your-username>/gradience.git
cd gradience
```

### 2. Install Dependencies

Gradience uses `uv` for dependency management.

Install all project dependencies:

```bash
uv sync
```

Activate the virtual environment if needed:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

---

## Running the Project

Execute Python scripts using:

```bash
uv run python examples/example.py
```

or

```bash
uv run python your_script.py
```

---

## Running Tests

Run the full test suite:

```bash
pytest
```

Run a single test:

```bash
pytest tests/test_tensor.py
```

---

## Code Style

Please follow these guidelines:

* Follow PEP 8 where practical.
* Prefer descriptive variable names.
* Keep functions focused and small.
* Add docstrings for public APIs.
* Write comments that explain *why*, not *what*.

Before submitting a pull request, format and lint the project:

```bash
ruff check .
ruff format .
```

---

## Writing Tests

Every new feature should include tests whenever practical.

Examples include:

* Tensor operations
* Gradient correctness
* Broadcasting
* Shape inference
* Neural network layers
* Optimizers

---

## Commit Messages

Write concise, descriptive commit messages.

Examples:

```
Add matrix multiplication operation
Implement backward pass for ReLU
Fix gradient accumulation bug
Improve tensor documentation
```

Avoid messages like:

```
fix
update
changes
misc
```

---

## Pull Requests

Before opening a pull request, please ensure that:

* The project builds successfully.
* All tests pass.
* New functionality includes tests.
* Documentation is updated if necessary.
* Code has been formatted.

When describing a pull request, explain:

* What changed
* Why it changed
* Any implementation details reviewers should know

---

## Reporting Issues

If you've found a bug, please include:

* Python version
* Operating system
* Steps to reproduce
* Expected behavior
* Actual behavior
* Full traceback (if applicable)

---

## Feature Requests

Feature requests are welcome.

When proposing a feature, please explain:

* The problem you're trying to solve
* Your proposed solution
* Possible alternatives
* Any trade-offs

---

## Questions

If you're unsure about an implementation or design decision, feel free to open a discussion before starting work. We'd rather discuss ideas early than rework large changes later.

---

Thank you for helping make Gradience better.
