Of course. Here is the consolidated and verified list of all variables, their corresponding formulas or definitions, and their sources within the paper, presented in a nested bullet point format for clarity.

### **Variable Definitions and Formulas**

* **Balance Sheet Variables**
    * **K**: Assets 
        * **Formula**: $K_{t}=CA_{t}+FA_{t}$ 
        * **Source**: Eq. 2.2 
    * **CA**: Current Assets 
        * **Formula**: $CA_{t}=CA_{t-1}+dca_{t}$ 
        * **Source**: Eq. 2.27 
    * **FA**: Fixed Assets 
        * **Formula**: $FA_{t}=MA_{t}+BU_{t}+OFA_{t}$ 
        * **Source**: Eq. 2.1 
    * **MA**: Machinery and Equipment 
        * **Formula**: $MA_{t}=MA_{t-1}+I_{t}^{MA}-S_{t}^{MA}-EDEP_{t}^{MA}$ 
        * **Source**: Eq. 2.28 
    * **BU**: Buildings 
        * **Formula**: $BU_{t}=BU_{t-1}+I_{t}^{BU}-EDEP_{t}^{BU}$ 
        * **Source**: Eq. 2.29 
    * **OFA**: Other Fixed Assets 
        * **Formula**: $OFA_{t}=OFA_{t-1}+dofa_{t}$ 
        * **Source**: Eq. 2.30 
    * **CMA**: The Taxable Residual Value of Machinery and Equipment 
        * **Formula**: $CMA_{t}=MA_{t}-ASD_{t}$ 
        * **Source**: Eq. 2.44 
    * **WC**: Working Capital 
        * **Formula**: $WC=CA-CL$ 
        * **Source**: Section 3.1 
    * **B**: Liabilities 
        * **Formula**: $B_{t}=CL_{t}+LL_{t}+UR_{t}+EC_{t}$ 
        * **Source**: Eq. 2.3 
    * **CL**: Current Liabilities 
        * **Formula**: $CL_{t}=CL_{t-1}+dcl_{t}$ 
        * **Source**: Eq. 2.24 
    * **LL**: Long-Term Liabilities 
        * **Formula**: $LL_{t}=LL_{t-1}+dll_{t}$ 
        * **Source**: Eq. 2.23 
    * **UR**: Untaxed Reserves 
        * **Formula**: $UR_{t}=ASD_{t}+PF_{t}+OUR_{t}$ 
        * **Source**: Eq. 2.4 
    * **ASD**: Accumulated Supplementary Depreciation 
        * **Formula**: $ASD_{t}=ASD_{t-1}+(TDEP_{t}^{MA}-EDEP_{t}^{MA})$ 
        * **Source**: Eq. 2.31 
    * **OUR**: Other Untaxed Reserves 
        * **Formula**: $OUR_{t}=OUR_{t-1}+dour_{t}$ 
        * **Source**: Eq. 2.32 
    * **$PF_t^{t-5}$**: Remaining Periodical Reserves From t-5 in period t 
        * **Formula**: $PF_{t}^{t-5}=PF_{t-1}^{t-4}-zpf_{t}^{t-5}$ 
        * **Source**: Eq. 2.33 
    * **$PF_t^{t-1}$**: Remaining Periodical Reserves From t-1 in period t 
        * **Formula**: $PF_{t}^{t-1}=PF_{t-1}-zpf_{t}^{t-1}$ 
        * **Source**: Eq. 2.37 
    * **$PF_t^t$**: Periodical Reserves in Current Period t 
        * **Formula**: $PF_{t}=p_{t}^{allo}$ 
        * **Source**: Eq. 2.38 
    * **EC**: Equity Capital 
        * **Formula**: $EC_{t}=SC_{t}+RR_{t}+URE_{t}$ 
        * **Source**: Eq. 2.6 
    * **SC**: Share Capital 
        * **Formula**: $SC_{t}=SC_{t-1}+dsC_{t}$ 
        * **Source**: Eq. 2.25 
    * **RR**: Restricted Reserves 
        * **Formula**: $RR_{t}=RR_{t-1}+drr_{t}$ 
        * **Source**: Eq. 2.26 
    * **URE**: Unrestricted Equity 
        * **Formula**: $URE_{t}=URE_{t-1}+NBI_{t}-DIV_{t-1}-\Delta RR_{t}-cashfI_{t}$ 
        * **Source**: Eq. 2.22 

* **Income Statement & Tax Variables**
    * **OIBD**: Operating Income Before Depreciation (Accounting Income) 
        * **Formula**: No single formula, it's the difference between operating revenues and operating expenses. 
        * **Source**: Section 2.2 
    * **$EDEP^{MA}$**: Economic Depreciation of Machinery and Equipment 
        * **Formula**: No explicit formula, it is an estimated value. 
        * **Source**: Sections 1.2, 2.2 
    * **$EDEP^{BU}$**: Economic Depreciation of Buildings 
        * **Formula**: No explicit formula, it is an estimated value. 
        * **Source**: Sections 1.2, 2.2 
    * **OIAD**: Operating Income after Economic Depreciation 
        * **Formula**: $OIAD_{t}=OIBD_{t}-EDEP_{t}^{MA}-EDEP_{t}^{BU}$ 
        * **Source**: Eq. 2.9 
    * **FI**: Financial Income 
        * **Formula**: No single formula, it includes interest income, dividends received, etc. 
        * **Source**: Section 1.2 
    * **FE**: Financial Expenses 
        * **Formula**: No single formula, it includes interest costs, value adjustments, etc. 
        * **Source**: Section 1.2 
    * **EBA**: Earnings Before Allocations 
        * **Formula**: $EBA_{t}=OIAD_{t}+FI_{t}-FE_{t}$ 
        * **Source**: Eq. 2.10 
    * **$TDEP^{MA}$**: Tax Depreciation of Machinery and Equipment 
        * **Formula**: This is a decision variable, but it's constrained by legal rules (see $TDDB^{MA}$, $TDSL^{MA}$, $TDRV^{MA}$). 
        * **Source**: Sections 2.6, 2.6 
    * **OA**: Other Allocations 
        * **Formula**: No single formula, it includes received and given group contributions. 
        * **Source**: Section 1.2 
    * **zpf**: Change in Periodical Reserves 
        * **Formula**: $\Delta PF_{t}=p_{t}^{allo}-zpf_{t}^{t-5}-zpf_{t}^{t-4}-zpf_{t}^{t-3}-zpf_{t}^{t-2}-zpf_{t}^{t-1}-PF_{t-1}^{t-5}$ 
        * **Source**: Eq. 2.14 
    * **$p^{allo}$**: Allocations to Periodical Reserves 
        * **Formula**: $p_{t}^{allo}\le \eta \times pbase_{t}$ 
        * **Source**: Eq. 2.56, 2.57 
    * **EBT**: Earnings Before Taxes 
        * **Formula**: $EBT_{t}=EBA_{t}-\Delta ASD-\Delta PF_{t}-OA_{t}$ 
        * **Source**: Eq. 2.11 
    * **TL**: Tax Liability 
        * **Formula**: No single formula, it is an estimate of the firm's tax payment. 
        * **Source**: Section 1.2 
    * **NI**: Net Income 
        * **Formula**: $NI_{t}=EBT_{t}-TL_{t}$ 
        * **Source**: Eq. 2.16 
    * **TA**: Tax Adjustments 
        * **Formula**: $TA_{t}=OTA_{t}-TDEP_{t}^{BU}-OL_{t-1}$ 
        * **Source**: Eq. 2.18 
    * **OTA**: Other Tax Adjustments 
        * **Formula**: No single formula, includes various adjustments like non-taxable revenues and non-deductible expenses. 
        * **Source**: Section 2.2 
    * **$TDEP^{BU}$**: Tax Depreciation of Buildings 
        * **Formula**: It is a decision variable, though constrained by tax rules. 
        * **Source**: Sections 2.2, 3.1 
    * **$OL_{t-1}$**: Losses From Previous Years 
        * **Formula**: A stock variable carried over from previous periods. 
        * **Source**: Section 2.2 
    * **TAX**: Calculated Tax Payments 
        * **Formula**: $TAX_{t}=\tau~max[0,(EBT_{t}-TL_{t}+TA_{t})]$ 
        * **Source**: Eq. 2.17 
    * **ROT**: Reduction Of Taxes 
        * **Formula**: No explicit formula, it's for reductions like for foreign taxes paid. 
        * **Source**: Sections 1.2, 2.2 
    * **FTAX**: Final Tax Payments 
        * **Formula**: $FTAX_t = TAX_t - ROT_t$ 
        * **Source**: Eq. 2.19 
    * **$OL_t$**: The Stock of Old Losses 
        * **Formula**: $OL_{t}=min[0,(EBT_{t}-TL_{t}+TA_{t})]$ 
        * **Source**: Eq. 2.21 
    * **NBI**: Net Business Income 
        * **Formula**: $NBI_t = EBT_t - FTAX_t$ 
        * **Source**: Eq. 2.20 

* **Flow Variables**
    * **$I^{MA}$**: Net Investment in Machinery and Equipment 
    * **$I^{BU}$**: Net Investment in Buildings 
    * **dca**: Net Change in Current Assets 
    * **dofa**: Net Change in Other Fixed Assets 
    * **dcl**: Net Change in Current Liabilities 
    * **dll**: Net Change in Long-Term Liabilities 
    * **dour**: Net Change in Other Untaxed Reserves 
    * **dsc**: Net Change in Share Capital 
    * **drr**: Net Changes in Restricted Reserves 
    * **dURE**: Net Change in Unrestricted Equity (Retained Earnings) 
    * **cashfl**: Cash flow 
        * **Formula**: $cashfl_{t}=OIBD_{t}+FI_{t}-FE_{t}+OA_{t}-FTAX_{t}-DIV_{t-1}+dsc_{t}+dcl_{t}+dll_{t}+dour_{t} -I_{t}^{MA}+S_{t}^{MA}-I_{t}^{BU}-dofa_{t}-dca_{t}$ 
        * **Source**: Eq. 2.39 
    * **$S^{MA}$**: Sales of Machinery and Equipment 
    * **IG**: Investment Grant 
    * **DIV**: Dividends Paid to Shareholders 
        * **Formula**: $DIV_{t}=max[0,min(cashfl_{t},mcash_{t})]$ 
        * **Source**: Eq. 2.43 
    * **GC**: Net Group Contribution 

* **Financial Ratios**
    * **CR**: The Current Ratio 
        * **Formula**: $CR_t = \frac{CA_t}{CL_t}$ 
        * **Source**: Eq. 7.1 
    * **DR**: The Debt Ratio 
        * **Formula**: $DR_t = \frac{CL_t + LL_t + \tau(ASD_t + PF_t + OUR_t)}{K_t}$ 
        * **Source**: Eq. 7.2 
    * **DER**: The Debt/Equity Ratio 
        * **Formula**: $DER_t = \frac{CL_t + LL_t + \tau(ASD_t + PF_t + OUR_t)}{SC_t + RR_t + URE_t + (1-\tau)(ASD_t + PF_t + OUR_t)}$ 
        * **Source**: Eq. 7.3 
    * **ECR**: The Equity Capital Ratio 
        * **Formula**: $ECR_t = \frac{SC_t + RR_t + URE_t + (1-\tau)(ASD_t + PF_t + OUR_t)}{K_t}$ 
        * **Source**: Eq. 7.5 
    * **FQ**: The Financial Q 
        * **Formula**: $FQ_t = \frac{CA_t - CL_t - LL_t - \tau(ASD_t + PF_t + OUR_t)}{K_t}$ 
        * **Source**: Footnote 58 
    * **ICR**: The Interest Coverage Ratio 
        * **Formula**: $ICR_t = \frac{OIBD_t - EDEP_t^{MA} - EDEP_t^{BU} + FI_t}{FE_t}$ 
        * **Source**: Eq. 7.7 
    * **ROA**: Return on Total Assets 
        * **Formula**: $ROA_t = \frac{OIBD_t - EDEP_t^{MA} - EDEP_t^{BU} + FI_t + TL_t}{K_t}$ 
        * **Source**: Eq. 7.8 
    * **ROE**: Return on Equity 
        * **Formula**: $ROE_t = \frac{OIBD_t - EDEP_t^{MA} - EDEP_t^{BU} + FI_t - FE_t + TL_t}{SC_t + RR_t + URE_t + (1-\tau)(ASD_t + PF_t + OUR_t)}$ 
        * **Source**: Eq. 7.11 
    * **DI**: Average Debt Interest 
        * **Formula**: $DI_t = \frac{FE_t}{CL_t + LL_t + \tau(ASD_t + PF_t + OUR_t)}$ 
        * **Source**: Eq. 7.10 
    * **ROI**: Return on Investment 
        * **Formula**: $ROI_t = ROA_t$ 
        * **Source**: Eq. 7.12 
    * **RROI**: Required Return on Investment 
        * **Formula**: $RROI_t = \frac{i_t}{1-\tau_t^{eff}}$ 
        * **Source**: Eq. 7.13 
    * **$\tau^{eff}$**: Effective Corporate Tax Rate 
        * **Formula**: $\tau_t^{eff} = \frac{FTAX_t}{EBA_t}$ 
        * **Source**: Eq. 7.14 
    * **ER**: Excess Return on Investment 
        * **Formula**: $ER_t = ROI_t - RROI_t$ 
        * **Source**: Eq. 7.15 

* **Legal Constraints**
    * **$TDDB^{MA}$**: Tax Depreciation of Machinery and Equipment (Declining Balance Method) 
        * **Formula**: $TDDB_{t}^{MA}=(M/12)\delta^{db}[CMA_{t-1}+I_{t}^{MA}-S_{t}^{MA}-IG_{t}]$ 
        * **Source**: Eq. 2.49 
    * **$TDSL^{MA}$**: Tax Depreciation of Machinery and Equipment (Straight-Line Method) 
        * **Formula**: $TDSL_{t}^{MA}=CMA_{t-1}+I_{t}^{MA}-S_{t}^{MA}-IG_{t} -[\delta_{t}^{s}(I_{t}^{MA}-IG_{t})+\delta_{t-1}^{s}I_{t-1}^{MA}+\delta_{t-2}^{s}I_{t-2}^{MA}+\delta_{t-3}^{s}I_{t-3}^{MA}]$ 
        * **Source**: Eq. 2.50 
    * **$TDRV^{MA}$**: Tax Depreciation of Machinery and Equipment (Rest Value Method) 
        * **Formula**: $TDRV_{t}^{MA}=(M/12)\delta^{rv}[CMA_{t-1}+I_{t}^{MA}+S_{t}^{MA}-IG_{t}]$ 
        * **Source**: Eq. 2.51 
    * **MTDM**: Maximum Amount of Tax Depreciation that Firms May Deduct from Their Taxable Income 
        * **Formula**: It's the maximum of the different depreciation methods available to the firm. 
        * **Source**: Section 2.6, Eq. 2.53, 2.54 
    * **dmtdm**: Difference Between MTDM and TDEPMA 
        * **Formula**: $dmtdm_t = MTDM_t - TDEP_t^{MA}$ 
        * **Source**: Eq. 5.1 
    * **ddmtdm**: Indicator Whether Firms Change Their Utilization of Depreciation Allowances 
        * **Formula**: $ddmtdm_{t}=(MTDM_{t}-TDEP_{t}^{MA})-(MTDM_{t-1}-TDEP_{t-1}^{MA})$ 
        * **Source**: Eq. 5.2 
    * **MPA**: The Maximum Amount of Allocations that Firms are Allowed to Make to Periodical Reserves 
        * **Formula**: $MPA_t = max[0, (\eta \times pbase_t)]$ 
        * **Source**: Eq. 2.57 
    * **dmpa**: Difference Between the Maximum Amount of Allocations to Periodical Reserves and the Allocations Made by the Firms 
        * **Formula**: $dmpa_t = MPA_t - p_t^{allo}$ 
        * **Source**: Eq. 5.3 
    * **ddmpa**: The Change in the Utilization of Tax Rules Regarding Allocations to Periodical Reserves 
        * **Formula**: $ddmpa_{t}=(MPA_{t}-p_{t}^{allo})-(MPA_{t-1}-p_{t-1}^{allo})$ 
        * **Source**: Eq. 5.4 
    * **dcashfl**: The Change in Cash Flow 
        * **Formula**: This is a derived variable for analysis, not a direct equation. 
        * **Source**: Section 5.1 
    * **mcash**: Maximum Dividends Firms Can Pay to Their Shareholders 
        * **Formula**: $mcash_t = URE_{t-1} + NBI_t - drr_t$ 
        * **Source**: Eq. 2.41 
    * **dmcash**: The Difference Between the Maximum Dividends Firms Can Pay to Their Shareholders and the Amount of Dividends They Actually Pay 
        * **Formula**: $dmcash_t = mcash_t - cashfl_t$ 
        * **Source**: Eq. 5.5 
    * **ddmcash**: The Change of Firms’ Dividend Policy so that they Come Closer to the Legal Constraint on Dividend Payments 
        * **Formula**: $ddmcash_{t}=(mcash_{t}-cashfl_{t})-(mcash_{t-1}-cashfl_{t-1})$ 
        * **Source**: Eq. 5.6 

* **Parameters**
    * **$\delta^{DB}$**: The Depreciation Rate (Declining Balance Method) 
        * **Definition**: 30% 
        * **Source**: Section 2.6 
    * **$\delta^S$**: The Depreciation Rate (Straight-Line Method) 
        * **Definition**: 20% 
        * **Source**: Section 2.6 
    * **$\delta^{RV}$**: The Depreciation Rate (Rest Value Method) 
        * **Definition**: 25% 
        * **Source**: Section 2.6 
    * **M**: Number of Months in the Firms’ Income Year 
        * **Formula**: No formula, it is a characteristic of the firm's accounting period. 
        * **Source**: Section 2.6 
    * **$\tau$**: Corporate Tax Rate 
        * **Formula**: No formula, it is a statutory rate. 
        * **Source**: Section 2.2 

* **Macroeconomic Variables**
    * **BNP**: Gross National Product 
        * **Formula**: No formula provided, it's an exogenous macroeconomic variable. 
        * **Source**: Section 5.2 
    * **ranta10**: The Interest Rate on a Government Bond with a Maturity of 10 Years 
        * **Formula**: No formula provided, it's an exogenous macroeconomic variable. 
        * **Source**: Section 5.2 
    * **inf**: Inflation 
        * **Formula**: No formula provided, it's an exogenous macroeconomic variable. 
        * **Source**: Section 5.2 
    * **realr**: Real Interest Rate 
        * **Formula**: $realr_t = \frac{1 + ranta10_t}{1 + inf_t} - 1 = \frac{ranta10_t - inf_t}{1 + inf_t}$ 
        * **Source**: Eq. 5.8 

* **Other Variables**
    * **Public**: Indicator for Firms that are Public Companies 
        * **Definition**: 1 if share capital is 500,000 SEK or more; 0 if share capital is 100,000 SEK or more (but less than 500,000). 
        * **Source**: Section 5.3 
    * **FAAB**: Indicator for Firms that are Closed Companies 
        * **Definition**: 1 if the firm is a closed company (files form K10); 0 otherwise. 
        * **Source**: Section 5.3 
    * **Largcity**: Indicator for Firms Located in Large Cities 
        * **Definition**: 1 if the firm is located in a larger city; 0 otherwise. 
        * **Source**: Eq. 5.11 
    * **Ruralare**: Indicator for Firms Located in Rural Areas 
        * **Definition**: 1 if the firm is located in a more sparsely populated area; 0 otherwise. 
        * **Source**: Eq. 5.12 
    * **Market**: Index for Competition in the Market 
        * **Formula**: $market_{t-1} = \frac{1}{\text{count}}$ 
        * **Source**: Eq. 5.13 
    * **MarketW**: The Market Share of the Firm 
        * **Formula**: $marketw_{t-1} = \frac{K_{t-1}}{\text{msum}}$ 
        * **Source**: Eq. 5.14 

* **Variables Controlling for the Decisions Made by the Firms**
    * **$DI^{MA}$**: Possible Investment in Machinery and Equipment 
        * **Definition**: $DI_t^{MA} = 1$ if $I_t^{MA} > 0$; 0 if $I_t^{MA} = 0$. 
        * **Source**: Eq. 5.15 
    * **$DI^{BU}$**: Possible Investment in Buildings 
        * **Definition**: $DI_t^{BU} = 1$ if $I_t^{BU} \neq 0$; 0 if $I_t^{BU} = 0$. 
        * **Source**: Eq. 5.16 
    * **Ddofa**: Possible Net Change in Other Fixed Assets 
        * **Definition**: $Ddofa_t = 1$ if $dofa_t \neq 0$; 0 if $dofa_t = 0$. 
        * **Source**: Eq. 5.17 
    * **Ddll**: Possible Net Change in Long-Term Liabilities 
        * **Definition**: $Ddll_t = 1$ if $dll_t \neq 0$; 0 if $dll_t = 0$. 
        * **Source**: Eq. 5.18 
    * **Ddsc**: Possible Net Change in Share Capital 
        * **Definition**: $Ddsc_t = 1$ if $dsc_t \neq 0$; 0 if $dsc_t = 0$. 
        * **Source**: Eq. 5.19 
    * **$DTDEP^{MA}$**: Possible Tax Depreciation of Machinery and Equipment 
        * **Definition**: $DTDEP_t^{MA} = 1$ if $TDEP_t^{MA} > 0$; 0 if $TDEP_t^{MA} = 0$. 
        * **Source**: Eq. 5.20 
    * **Dzpf**: Possible Change in Periodical Reserves 
        * **Definition**: $Dzpf_t = 1$ if $zpf_t > 0$; 0 if $zpf_t = 0$. 
        * **Source**: Eq. 5.21 
    * **Ddour**: Possible Net Change in Other Untaxed Reserves 
        * **Definition**: $Ddour_t = 1$ if $dour_t \neq 0$; 0 if $dour_t = 0$. 
        * **Source**: Eq. 5.22 