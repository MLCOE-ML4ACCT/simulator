# Rationale for Enforced Simulation Logic

This document details the motivation and implementation of the constraints added to our simulation engine, primarily found in `theoretical/simple_theoretical_enforce.py`. These constraints are crucial for ensuring the economic and accounting integrity of the simulation results.

## Why We Need Enforcement

In a dynamic micro-simulation model like ours, firms' financial states evolve over many periods. Without constraints, the econometric models, while statistically sound, can produce results that are not feasible in the real world. For example, an unconstrained model might predict a negative asset value or a depreciation amount that exceeds the asset's worth.

These small, period-by-period inaccuracies can compound over time, leading to simulations that diverge from reality and produce nonsensical financial statements. The purpose of the "enforced" simulation logic is to impose real-world accounting and economic rules to prevent this, ensuring the long-term stability and validity of our model.

## What We Added and Why

The core of the enforcement logic is to ensure that all financial flows and stocks remain within plausible bounds. Below is a detailed breakdown of the key constraints we've implemented.

### 1. Non-Negativity of Asset and Liability Stocks

- **What was added:** We enforce that balance sheet items like Machinery (`MAt`), Buildings (`BUt`), Current Assets (`CAt`), and others cannot become negative. This is done by constraining the *outflows* (e.g., sales, depreciation, payments).
- **Implementation:**
  - For assets like `MAt` and `BUt`, we calculate a `scaling_factor`. If the sum of outflows (e.g., `SMAt + EDEPMAt`) is greater than the available stock (`MAt_1 + IMAt`), all outflows are scaled down proportionally.
  - For other balance sheet flows (`dOFAt`, `dCAt`, `dLLt`, etc.), we ensure the change (`d<Variable>`) is not so negative that it wipes out the initial stock. For example, `dCAt` is constrained to be greater than or equal to `-CAt_1`.
- **Reason:** This is a fundamental accounting principle. A firm cannot have negative physical assets or owe a negative amount of liability in this context. This prevents the simulation from producing impossible balance sheets.

### 2. Logical Constraints on Financial Flows

- **What was added:** We've put floors on several calculated flows to ensure they are economically sound.
- **Implementation & Reason:**
  - **Investment (`IMAt`, `IBUt`):** Constrained to be non-negative (`>= 0`). A firm cannot have negative investment.
  - **Economic Depreciation (`EDEPMAt`, `EDEPBUt`):** Constrained to be non-negative. Depreciation is a cost and cannot be a negative value (which would imply a spontaneous increase in value).

### 3. Tax Depreciation (`TDEPMAt`) Channeling

The tax depreciation for machinery and equipment (`TDEPMAt`) is a critical flow variable that is constrained by a "channel" with both a floor and a ceiling. This ensures both the internal consistency of the balance sheet and compliance with external tax regulations.

- **The Floor: Preventing Negative Accumulated Depreciation**
  - **Constraint:** `TDEPMAt >= EDEPMAt - ASDt_1`
  - **Reason:** The floor is derived from the non-negativity constraint of the Accumulated Supplementary Depreciation (`ASDt`) stock. The update equation is `ASDt = ASDt_1 + TDEPMAt - EDEPMAt`. To ensure `ASDt` never drops below zero, `TDEPMAt` cannot be smaller than the difference between the economic depreciation and the previous period's accumulated depreciation.
  - **Motivation:** This maintains the accounting integrity of the balance sheet.

- **The Ceiling: Compliance with Tax Law**
  - **Constraint:** `TDEPMAt <= MTDMt`
  - **Reason:** The ceiling is defined by `MTDMt`, which represents the **M**aximum **T**ax **D**epreciation for **M**achinery allowed by law for a given period. This value is calculated based on the tax rules outlined in the source paper (Shahnazarian, 2004), specifically the maximum of the Declining Balance and Rest Value methods (Formulas 2.49-2.54).
  - **Motivation:** This ensures the simulation adheres to real-world tax regulations, preventing firms from claiming more depreciation than legally permitted. This is a more accurate and direct implementation of the tax code compared to the previous internal consistency check.

### Summary of Key Enforcements

| Variable / Logic | Constraint | Justification & Motivation |
| :--- | :--- | :--- |
| **`EDEPMAt`** | `tf.maximum(0.0, EDEPMAt)` | **Reason:** Economic depreciation cannot be negative. <br/><br/> **Motivation:** To ensure the simulation produces economically sensible results. |
| **`MAt` & `BUt` Stocks** | Scale outflows if they exceed available stock. | **Reason:** An asset's book value cannot be negative. A firm cannot sell or depreciate more assets than it owns. <br/><br/> **Motivation:** To enforce fundamental accounting identities and prevent impossible financial states. |
| **Balance Sheet Flows** (`dCAt`, `dLLt`, etc.) | `d<Var>t = tf.maximum(d<Var>t, -vars_t_1["<Var>"])` | **Reason:** Prevents the decrease in a balance sheet account from being larger than its opening balance. <br/><br/> **Motivation:** To ensure all stock variables remain non-negative. |
| **`TDEPMAt`** | Bounded by a floor derived from `ASDt` and a ceiling defined by `MTDMt`. | **Reason:** The floor prevents `ASDt` (Accumulated Supplementary Depreciation) from becoming negative. The ceiling (`MTDMt`) ensures the tax depreciation claimed does not exceed the maximum allowed by tax law (Formula 2.52 in the source paper). <br/><br/> **Motivation:** To enforce both the internal consistency of the balance sheet and compliance with external tax regulations. |

By implementing these constraints, we ensure that our simulation produces financial statements that are not only statistically predicted but also adhere to the strict rules of accounting and economic theory. This greatly enhances the reliability and credibility of our model's outputs.
