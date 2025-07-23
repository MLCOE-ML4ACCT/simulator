import tensorflow as tf

def generate_company_tensors(batch_size=10000):
    """
    Generate TensorFlow tensors for company accounting variables
    
    Args:
        batch_size: Batch size, default is 10000
        
    Returns:
        dict: Dictionary containing all variable tensors, each tensor has shape [batch_size, 1]
    """
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Initial variable: dca = CA ~ Uniform[0.2, 0.6] × 1million
    CA_past = tf.random.uniform([batch_size, 1], minval=0.2*1000000, maxval=0.6*1000000, dtype=tf.float32)
    dca_past = CA_past  # CA_{t-1} = dca_{t-1}
    
    # Calculate other variables based on proportional relationships
    # CA:URE = 1:1, so URE = CA
    URE_past = CA_past
    
    # URE:SC:RR = 10:3:3
    # Total weight = 10 + 3 + 3 = 16
    # URE is known, so total equity = URE * 16/10 = URE * 1.6
    total_equity = URE_past * 16.0 / 10.0
    SC_past = total_equity * 3.0 / 16.0  # SC = total equity * 3/16
    RR_past = total_equity * 3.0 / 16.0  # RR = total equity * 3/16
    
    # MA:URE = 1:1.5, so MA = URE / 1.5
    MA_past = URE_past / 1.5
    
    # MA:BU:OFA = 1:1:6
    # Total weight = 1 + 1 + 6 = 8
    # MA is known, so total fixed assets = MA * 8/1 = MA * 8
    total_fixed_assets = MA_past * 8.0
    BU_past = total_fixed_assets * 1.0 / 8.0  # BU = total fixed assets * 1/8 = MA
    OFA_past = total_fixed_assets * 6.0 / 8.0  # OFA = total fixed assets * 6/8 = MA * 6
    
    # Financing relationship: CA_{t-1} = dca_{t-1} = CL_{t-1} = dcl_{t-1}
    CL_past = CA_past
    dcl_past = CA_past
    
    # Accounting identity: CA + MA + BU + OFA = CL + LL + URE + SC + RR
    # Since CA = CL, so: MA + BU + OFA = LL + URE + SC + RR
    # Therefore: LL = (MA + BU + OFA) - (URE + SC + RR)
    total_equity_sum = URE_past + SC_past + RR_past
    total_fixed_assets_sum = MA_past + BU_past + OFA_past
    LL_past = total_fixed_assets_sum - total_equity_sum  # Corrected formula
    dll_past = LL_past
    
    # Variables fixed to 0
    ASD_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # Accumulative Supplementary Depreciation
    OUR_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # Other untaxed reserves
    
    # CMA_{t-1} = MA_{t-1} (because ASD_{t-1} = 0)
    CMA_past = MA_past
    
    # Other flow variables set to 0
    EDEPBU_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # EDEP_{t-1}^{BU}
    TDEPBU_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # TDEP_{t-1}^{BU}
    p_allo_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # p_{t-1}^{allo}
    dcashfl_past = tf.zeros([batch_size, 1], dtype=tf.float32)  # dcashfl_{t-1}
    ddmpa_past = tf.zeros([batch_size, 1], dtype=tf.float32)   # ddmpa_{t-1}
    ddmcash_past = tf.zeros([batch_size, 1], dtype=tf.float32) # ddmcash_{t-1}
    
    # Generate dummy variables for company location
    # 0: largcity, 1: ruralare, 2: others
    loc_code = tf.random.uniform([batch_size, 1], minval=0, maxval=3, dtype=tf.int32)
    largcity = tf.cast(tf.equal(loc_code, 0), tf.float32)
    ruralare = tf.cast(tf.equal(loc_code, 1), tf.float32)
    
    # Collect all variables into a dictionary
    tensors = {
        'CA_past': CA_past,
        'dca_past': dca_past,
        'URE_past': URE_past,
        'SC_past': SC_past,
        'RR_past': RR_past,
        'MA_past': MA_past,
        'BU_past': BU_past,
        'OFA_past': OFA_past,
        'CL_past': CL_past,
        'dcl_past': dcl_past,
        'LL_past': LL_past,
        'dll_past': dll_past,
        'ASD_past': ASD_past,
        'OUR_past': OUR_past,
        'CMA_past': CMA_past,
        'EDEPBU_past': EDEPBU_past,
        'TDEPBU_past': TDEPBU_past,
        'p_allo_past': p_allo_past,
        'dcashfl_past': dcashfl_past,
        'ddmpa_past': ddmpa_past,
        'ddmcash_past': ddmcash_past,
        'largcity': largcity,
        'ruralare': ruralare
    }
    
    return tensors

# Example usage
if __name__ == "__main__":
    # Generate tensors
    company_tensors = generate_company_tensors(batch_size=10000)
    
    print(f"Generated tensors for {len(company_tensors)} variables")
    print("Variable list:")
    for name, tensor in company_tensors.items():
        print(f"  {name}: {tensor.shape}") 
    
    # Print the range of SC_past
    SC_past = company_tensors['SC_past']
    sc_min = tf.reduce_min(SC_past).numpy()
    sc_max = tf.reduce_max(SC_past).numpy()
    print(f"SC_past min value: {sc_min:.2f}, max value: {sc_max:.2f}") 

    # Print unique value distribution for largcity and ruralare
    largcity = company_tensors['largcity']
    ruralare = company_tensors['ruralare']
    print(f"Unique values in largcity: {tf.unique(tf.reshape(largcity, [-1])).y.numpy()}")
    print(f"Unique values in ruralare: {tf.unique(tf.reshape(ruralare, [-1])).y.numpy()}")
    print(f"Count of largcity=1: {int(tf.reduce_sum(largcity).numpy())}")
    print(f"Count of ruralare=1: {int(tf.reduce_sum(ruralare).numpy())}")
    print(f"Count of both being 0: {int(tf.reduce_sum((1-largcity)*(1-ruralare)).numpy())}") 