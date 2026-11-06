"""run_all.py -- Run the whole Iteration 2 pipeline in order.

    python src/run_all.py

1. download the dataset if it is missing   (download_data.py)
2. describe the real data, make plots      (load_data.py)
3. run the simulation check                (simulation.py)
"""
import download_data
import load_data
import simulation


def main():
    download_data.ensure_data()
    print("\n" + "=" * 70 + "\nSTEP 2: data description\n" + "=" * 70)
    load_data.main()
    print("\n" + "=" * 70 + "\nSTEP 3: simulation check\n" + "=" * 70)
    simulation.main()


if __name__ == "__main__":
    main()
