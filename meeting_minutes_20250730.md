## **Project Meeting Minutes**

**Date:** July 30, 2025

**Participants:**

* Yifan
* Hongxiao
* Chak
* Elton

---

### **Meeting Purpose**

To establish a core modeling philosophy, refine the simulator's decision-making logic ("policy function"), narrow the scope of the initial project proposal, and reinforce the team's research and development culture.

---

### **1. Modeling Philosophy: Principled Over Ad-Hoc**

Chak opened the discussion by emphasizing the need for a principled modeling approach, warning against the use of ad-hoc fixes.

* **The Core Principle:** The model's internal structure must reflect economic reality. We should not simply apply patches when outputs are unrealistic.
* **The Why:** Ad-hoc solutions (e.g., manually capping a value that goes "out of bounds") are brittle and treat the symptom, not the cause. A principled approach builds a robust and generalizable model.
* **Analogy:** If a variable like "investment" can't be negative, we don't wait for it to go negative and then force it to zero. Instead, we choose a statistical distribution that *cannot* be negative in the first place, such as a **truncated normal distribution**. We design the rules of the world correctly from the start rather than policing the outcomes later.

---

### **2. Refining the Policy Function & Parsimonious Models**

The conversation turned to the simulator's core logic: how simulated firms will make decisions.

* **The Challenge:** How do we create a realistic "policy function" that determines an agent's actions (e.g., how much debt to issue next year)? Hongxiao noted that many academic papers simplify this by assuming a steady state and linear relationships.
* **Chak's Mandate: The Pursuit of Parsimony.** A good model is not the most complex one; it is the simplest one that still captures the essence of the problem. This means we must be disciplined in what we include.
* **Core Principle:** Do not confuse statistical significance with practical importance.
    * **The Why:** In a world of massive datasets, many variables may be "statistically significant" (e.g., have a low p-value from a t-test). However, their actual impact, or **effect size**, might be negligible. Including these variables adds complexity without adding explanatory power. We aim to build a **parsimonious model**.
    * **Actionable Test:** When evaluating a variable, we will look beyond the t-statistic and also assess its effect size (e.g., using Cohen's d). If the effect is tiny, we will exclude it to keep our model lean and interpretable.

---

### **3. Defining the Project Proposal Scope**

The team aligned on a focused scope for the initial project proposal.

* **Decision:** The first phase of this project will focus on two key areas:
    1.  **Distributional Analysis:** Accurately modeling the statistical distribution of firm-level financial variables.
    2.  **Macroeconomic Linkages:** Simulating how these distributions respond to changes in the macroeconomic environment.
* **Technical Note on Valuation:** Hongxiao highlighted a critical technical issue in standard company valuation. The Free Cash Flow to Firm (FCFF) model often involves circular logic (WACC needs firm value, which needs WACC). He noted that the **Adjusted Present Value (APV)** method, as used in prior literature (e.g., a 2010 paper), elegantly avoids this circularity. This will be our preferred approach.

---

### **4. Team Culture: Deep Understanding is Non-Negotiable **

Chak concluded with a directive on our team's research and development culture.

* **The Core Principle:** We must move beyond "filling in equations" and achieve a profound understanding of the problems we are solving.
* **The Why:** To build a tool that is **state-of-the-art (SOTA)**, we must first deeply understand the current SOTA and the foundational theories it's built upon. Without this, we risk hitting a ceiling of competence and becoming mere operators of a system we don't truly understand.
* **Analogy: Avoiding the Peter Principle.** The Peter Principle warns that individuals are promoted to their level of incompetence. In R&D, this happens when a team stops learning the fundamentals and can no longer innovate, only maintain. We will avoid this by prioritizing continuous, deep learning.
* **Actionable Mandate:**
    1.  **Literature First:** Significant time must be dedicated to literature reviews of both academic papers and top-tier industrial implementations.
    2.  **Master the Foundations:** It is impossible to build a credible corporate finance simulator without a deep, intuitive understanding of cornerstone concepts like the **Modigliani-Miller (M&M) Theorem**. We must understand the theory, its assumptions, and why it matters in the real world.
