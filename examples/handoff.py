"""Run from the repository root after installing: python examples/handoff.py."""

from pathlib import Path

from ecse541_benchmark import Architecture, FEATURE_NAMES, load_benchmark


def main():
    repository = Path(__file__).resolve().parents[1]
    benchmark = load_benchmark(repository / "data")
    ids = benchmark.eligible_ids
    reference = benchmark.objectives(ids)
    features = benchmark.features(ids)

    print(f"Architectures: {len(benchmark):,}; training trials: {len(benchmark.trials):,}")
    print(f"Eligible: {len(ids):,}; objectives: {reference.shape}; features: {features.shape}")
    print("Objective columns: validation error (percentage points), GPU batch latency (ms)")
    print("Both objectives are minimized.")

    # Part 5 maps its six categorical genes back to an official benchmark ID.
    cell = Architecture(benchmark.architectures[int(ids[0])].operations)
    official_id = benchmark.id_for(cell)
    print(f"Example genes: {cell.operations}; official ID: {official_id}; encoding key: {cell.encoding_id}")
    print(f"Reference objectives for this ID: {benchmark.objectives([official_id])[0]}")
    print(f"First feature name: {FEATURE_NAMES[0]}")

    # Parts 3–4 work on a separate copy and store one prediction per ID/run.
    predictions = reference.copy()
    print(f"Prediction workspace: {predictions.shape}; reference values are retained separately.")


if __name__ == "__main__":
    main()
