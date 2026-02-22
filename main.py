import sys

import numpy as np

from huckel import HuckelSystem


def print_menu() -> None:
    print("\nHuckel Calculator")
    print("Select a system:")
    print("  1. Linear polyene")
    print("  2. Cyclic polyene")
    print("  3. Tetrahedron")
    print("  4. Cube")
    print("  5. Dodecahedron")
    print("  6. Naphthalene")
    print("  7. Octahedron (optional / non-sp2)")
    print("  8. Icosahedron (optional / non-sp2)")
    print("  9. Quit")


def read_int(prompt: str, min_value: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"Please enter an integer >= {min_value}.")
            continue
        return value


def yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(prompt + suffix).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def build_system(choice: str) -> HuckelSystem | None:
    if choice == "1":
        n = read_int("Number of carbons (n) for linear polyene: ", min_value=1)
        return HuckelSystem(n, "linear")
    if choice == "2":
        n = read_int("Number of carbons (n) for cyclic polyene: ", min_value=3)
        return HuckelSystem(n, "cyclic")
    if choice == "3":
        return HuckelSystem.platonic_solid("tetrahedron")
    if choice == "4":
        return HuckelSystem.platonic_solid("cube")
    if choice == "5":
        return HuckelSystem.platonic_solid("dodecahedron")
    if choice == "6":
        return HuckelSystem.naphthalene()
    if choice == "7":
        return HuckelSystem.platonic_solid("octahedron")
    if choice == "8":
        return HuckelSystem.platonic_solid("icosahedron")
    if choice == "9":
        return None

    print("Invalid selection. Please choose a number from 1 to 9.")
    return "invalid"  # type: ignore[return-value]


def print_results(system: HuckelSystem) -> None:
    result = system.calculate_mos()
    print(f"\nSystem: {system.system_type}")
    print(f"n = {system.n}, alpha = {system.alpha}, beta = {system.beta}")

    if yes_no("Show Huckel matrix?", default=False):
        print("\nHuckel matrix:")
        print(system.matrix)

    print("\nEigenvalues (sorted):")
    print(np.array2string(result.eigenvalues, precision=6, suppress_small=True))

    print("\nDegeneracies (energy: multiplicity):")
    for energy, multiplicity in result.degeneracies.items():
        print(f"  {energy: .6f}: {multiplicity}")


def main() -> int:
    print("Interactive Huckel pi-energy calculator")
    while True:
        print_menu()
        choice = input("Enter choice (1-9): ").strip()
        system = build_system(choice)

        if system is None:
            print("Exiting.")
            return 0

        if not isinstance(system, HuckelSystem):
            continue

        try:
            print_results(system)
        except Exception as exc:  # pragma: no cover - runtime UI safeguard
            print(f"Error while calculating system: {exc}")

        if not yes_no("\nCalculate another system?", default=True):
            print("Exiting.")
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
