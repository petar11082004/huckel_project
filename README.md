# Submission branch for marking

# Hückel Project

A small Python project for building and analyzing simple Hückel (tight-binding) Hamiltonians for conjugated systems.  
Developed as part of Exercise 1 in computational quantum chemistry practice.


---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/petar11082004/huckel_project.git
   cd huckel_project

   ```

## How To Run

The user-facing program is `main.py`.

Run it with:

```bash
python3 main.py
```

`main.py` provides an interactive menu where you can choose the system type and view the calculated Hückel eigenvalues and degeneracies.

## Project Structure

- `main.py` — interactive runner (this is the file most users need)
- `huckel.py` — core Hückel matrix construction and diagonalization logic
- `test_huckel.py` — unit tests for matrix construction and degeneracies

Most users only need `main.py`.

If you would like a detailed explanation of the implementation in `huckel.py` or `test_huckel.py`, I can provide it on request.
