from data_models.firm_state import FirmState
from estimators.dummy_estimator import DummyEstimator
from theoretical.simple_theoretical import SimulatorEngine


def main():
    print("--- Initializing Full Simulation ---")

    # 1. Create a sensible initial state for a company for t-1 (e.g., end of 2025)
    # Let's make one that is balanced to start with.
    # Assets = 100k + 50k = 150k
    # Liab+Eq = 30k + 20k + 100k = 150k
    initial_state_2025 = FirmState(
        MA=100000.0, CA=50000.0, LL=30000.0, CL=20000.0, SC=80000.0, URE=20000.0
    )

    # 2. Choose and create the estimator strategy
    my_estimator = DummyEstimator()

    # 3. Create the simulator engine, injecting the estimator and config
    simulator = SimulatorEngine(estimator=my_estimator, tax_rate=0.28)

    print(f"\n--- Simulating Year 2026 for firm... ---")
    print(f"Initial State (2025): {initial_state_2025}")

    # 4. Run the simulation for one year
    try:
        final_state_2026 = simulator.run_one_year(initial_state_2025)

        # 5. Print the results
        print("\n--- Simulation Complete ---")
        print(f"Final State (2026): {final_state_2026}")
        print("\nBalance sheet remained balanced. Engine test successful!")
    except AssertionError as e:
        print(f"\nENGINE ERROR: {e}")


if __name__ == "__main__":
    main()
