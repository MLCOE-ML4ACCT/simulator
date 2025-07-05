from data_models.firm_state import FirmState
from data_models.flow_variables import FlowVariables


def test_base_structure():
    # 1. Create an instance of FirmState with some sample data
    try:
        initial_state = FirmState(MA=100000.0, CA=50000.0, LL=30000.0, SC=120000.0)
        print("Successfully created FirmState object:")
        print(initial_state)
    except Exception as e:
        print(f"Failed to create FirmState object: {e}")
        return

    # 2. Create an instance of FlowVariables with some sample data
    try:
        flow_vars = FlowVariables(OIBD=20000.0, I_MA=10000.0, FE=500.0)
        print("\nSuccessfully created FlowVariables object:")
        print(flow_vars)
    except Exception as e:
        print(f"Failed to create FlowVariables object: {e}")
        return


if __name__ == "__main__":
    test_base_structure()
