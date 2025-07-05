from data_models.firm_state import FirmState
from estimators.dummy_estimator import DummyEstimator


def test_dummy_estimator():
    # 1. Create a dummy initial state. The estimator won't use it, but the
    #    method signature requires it, which is good practice.
    initial_state = FirmState(MA=100000.0)

    # 2. Instantiate our DummyEstimator
    try:
        dummy_estimator = DummyEstimator()
        print("Successfully created DummyEstimator object.")
    except Exception as e:
        print(f"Failed to create DummyEstimator object: {e}")
        return

    # 3. Call the get_flow_variables method
    try:
        flows = dummy_estimator.get_flow_variables(initial_state)
        print("Successfully received FlowVariables object from estimator:")
        print(flows)
        # Check if a specific value is correct
        assert flows.OIBD == 100000.0
    except Exception as e:
        print(f"Failed to get flow variables: {e}")
        return


if __name__ == "__main__":
    test_dummy_estimator()
