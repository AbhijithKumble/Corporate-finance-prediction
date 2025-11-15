"""
Financial Feature Engineering Script
Derives useful financial ratios and metrics from cleaned data for forecasting
Based on CNN-LSTM financial forecasting research paper
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration - Using script directory for absolute paths
SCRIPT_DIR = Path(__file__).parent
DATA_CLEANED_PATH = SCRIPT_DIR / 'data' / 'data_cleaned'
OTHER_DATA_PATH = SCRIPT_DIR / 'data' / 'other_data'
OTHER_DATA_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Financial Feature Engineering for Forecasting")
print("=" * 80)

# Load all financial data files
print("\n1. Loading financial data files...")
data_dict = {}
csv_files = list(DATA_CLEANED_PATH.glob('*.csv'))

for file in csv_files:
    metric_name = file.stem
    try:
        df = pd.read_csv(file)
        data_dict[metric_name] = df
        print(f"   ✓ Loaded {metric_name}: {df.shape[0]} companies")
    except Exception as e:
        print(f"   ✗ Error loading {metric_name}: {e}")

print(f"\n   Total metrics loaded: {len(data_dict)}")
# Get time period columns (all columns except 'Company Name')
if not data_dict:
    print("   ERROR: No data files loaded! Please check the data path.")
    exit(1)

sample_df = list(data_dict.values())[0]
time_periods = [col for col in sample_df.columns if col != 'Company Name']
company_names = sample_df['Company Name'].values
print(f"   Time periods: {len(time_periods)}")
print(f"   Companies: {len(company_names)}")

def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two arrays, handling zeros and NaN"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator, 
                          out=np.full_like(numerator, default, dtype=float),
                          where=(denominator != 0) & ~np.isnan(denominator) & ~np.isnan(numerator))
    return result

def get_value(df, company_name, period):
    """Get value for a company and time period"""
    if company_name not in df['Company Name'].values:
        return 0.0
    row = df[df['Company Name'] == company_name]
    if len(row) == 0 or period not in row.columns:
        return 0.0
    val = row[period].iloc[0]
    return float(val) if pd.notna(val) else 0.0

def calculate_growth_rate(current, previous):
    """Calculate percentage growth rate"""
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0.0
    return ((current - previous) / abs(previous)) * 100

print("\n2. Calculating derived financial features...")

# Initialize result dictionaries for each feature
features = {
    'current_ratio': [],
    'quick_ratio': [],
    'debt_to_asset_ratio': [],
    'debt_to_equity_ratio': [],
    'roe': [],
    'roa': [],
    'net_profit_margin': [],
    'operating_profit_margin': [],
    'asset_turnover': [],
    'revenue_growth_rate': [],
    'profit_growth_rate': [],
    'pe_ratio': [],
    'working_capital': [],
    'total_assets': [],
    'total_liabilities': [],
    'operating_cashflow_to_sales': [],
    'free_cashflow': [],
    'inventory_turnover': [],
    'current_assets_to_total_assets': [],
    'long_term_debt_to_total_assets': [],
    'equity_to_assets_ratio': [],
    'eps_growth_rate': [],
    'price_to_book_ratio': [],
    'dividend_yield': [],
    'retained_earnings_ratio': []
}

# Process each company
print(f"   Processing {len(company_names)} companies...")
for idx, company in enumerate(company_names):
    if (idx + 1) % 500 == 0:
        print(f"   Progress: {idx + 1}/{len(company_names)} companies processed")
    
    # Initialize row data for this company
    row_data = {'Company Name': company}
    
    # Process each time period
    for i, period in enumerate(time_periods):
        # Get base financial metrics
        current_assets = get_value(data_dict.get('current_assets', pd.DataFrame()), company, period)
        current_liabilities = get_value(data_dict.get('current_liabilities_and_provision', pd.DataFrame()), company, period)
        net_sales = get_value(data_dict.get('net_sales', pd.DataFrame()), company, period)
        share_holder_funds = get_value(data_dict.get('share_holder_funds', pd.DataFrame()), company, period)
        closing_price = get_value(data_dict.get('closing_price', pd.DataFrame()), company, period)
        eps_before = get_value(data_dict.get('eps_before', pd.DataFrame()), company, period)
        long_term_borrowings = get_value(data_dict.get('long_term_borrowings', pd.DataFrame()), company, period)
        short_term_borrowings = get_value(data_dict.get('short_term_borrowings', pd.DataFrame()), company, period)
        net_fixed_assets = get_value(data_dict.get('net_fixed_assets', pd.DataFrame()), company, period)
        investments = get_value(data_dict.get('investments', pd.DataFrame()), company, period)
        other_income = get_value(data_dict.get('other_income_and_extraordinary', pd.DataFrame()), company, period)
        operating_cashflow = get_value(data_dict.get('net_cashflow_from_operating_activities', pd.DataFrame()), company, period)
        investing_cashflow = get_value(data_dict.get('net_cashflow_from_investing_activities', pd.DataFrame()), company, period)
        change_in_stock = get_value(data_dict.get('change_in_stock', pd.DataFrame()), company, period)
        depreciation = get_value(data_dict.get('depreciation', pd.DataFrame()), company, period)
        dividend_rate = get_value(data_dict.get('dividend_rate', pd.DataFrame()), company, period)
        reserves = get_value(data_dict.get('reserves', pd.DataFrame()), company, period)
        paid_up_capital = get_value(data_dict.get('paid_up_capital', pd.DataFrame()), company, period)
        other_non_current_assets = get_value(data_dict.get('other_non_current_assets', pd.DataFrame()), company, period)
        long_term_investments = get_value(data_dict.get('long_term_investments', pd.DataFrame()), company, period)
        short_term_investments = get_value(data_dict.get('short_term_investments', pd.DataFrame()), company, period)
        
        # Calculate net profit (approximation: net sales - expenses)
        # Using other_income as proxy for profit when available
        net_profit = other_income if other_income != 0 else 0
        
        # Calculate operating profit (net sales - operating expenses approximation)
        operating_profit = net_sales * 0.1 if net_sales > 0 else 0  # Approximation
        
        # Calculate total assets
        total_assets = (current_assets + net_fixed_assets + 
                       investments + long_term_investments + 
                       short_term_investments + other_non_current_assets)
        
        # Calculate total liabilities
        total_liabilities = (current_liabilities + long_term_borrowings + 
                            short_term_borrowings)
        
        # Calculate total debt
        total_debt = long_term_borrowings + short_term_borrowings
        
        # Calculate total equity
        total_equity = share_holder_funds + reserves + paid_up_capital
        
        # 1. LIQUIDITY RATIOS
        # Current Ratio = Current Assets / Current Liabilities
        current_ratio = safe_divide(current_assets, current_liabilities)
        
        # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
        # Using change_in_stock as proxy for inventory
        quick_assets = current_assets - abs(change_in_stock)
        quick_ratio = safe_divide(quick_assets, current_liabilities)
        
        # 2. LEVERAGE RATIOS
        # Debt-to-Asset Ratio = Total Debt / Total Assets
        debt_to_asset = safe_divide(total_debt, total_assets)
        
        # Debt-to-Equity Ratio = Total Debt / Total Equity
        debt_to_equity = safe_divide(total_debt, total_equity)
        
        # 3. PROFITABILITY RATIOS
        # Return on Equity (ROE) = Net Profit / Shareholder Equity
        roe = safe_divide(net_profit, total_equity) * 100
        
        # Return on Assets (ROA) = Net Profit / Total Assets
        roa = safe_divide(net_profit, total_assets) * 100
        
        # Net Profit Margin = Net Profit / Net Sales
        net_profit_margin = safe_divide(net_profit, net_sales) * 100
        
        # Operating Profit Margin = Operating Profit / Net Sales
        operating_profit_margin = safe_divide(operating_profit, net_sales) * 100
        
        # 4. EFFICIENCY RATIOS
        # Asset Turnover = Net Sales / Total Assets
        asset_turnover = safe_divide(net_sales, total_assets)
        
        # Inventory Turnover = Net Sales / Inventory (using change_in_stock as proxy)
        inventory = abs(change_in_stock) if change_in_stock != 0 else 1
        inventory_turnover = safe_divide(net_sales, inventory)
        
        # 5. GROWTH RATIOS
        revenue_growth = 0.0
        profit_growth = 0.0
        eps_growth = 0.0
        
        if i > 0:
            prev_period = time_periods[i-1]
            prev_net_sales = get_value(data_dict.get('net_sales', pd.DataFrame()), company, prev_period)
            prev_net_profit = get_value(data_dict.get('other_income_and_extraordinary', pd.DataFrame()), company, prev_period)
            prev_eps = get_value(data_dict.get('eps_before', pd.DataFrame()), company, prev_period)
            
            revenue_growth = calculate_growth_rate(net_sales, prev_net_sales)
            profit_growth = calculate_growth_rate(net_profit, prev_net_profit)
            eps_growth = calculate_growth_rate(eps_before, prev_eps)
        
        # 6. MARKET RATIOS
        # P/E Ratio = Price per Share / Earnings per Share
        pe_ratio = safe_divide(closing_price, eps_before, default=0.0)
        pe_ratio = pe_ratio if (pe_ratio > 0 and pe_ratio < 10000) else 0.0  # Filter outliers
        
        # Price-to-Book Ratio = Market Price / Book Value per Share
        book_value_per_share = safe_divide(total_equity, paid_up_capital) if paid_up_capital > 0 else 0
        price_to_book = safe_divide(closing_price, book_value_per_share) if book_value_per_share > 0 else 0
        
        # Dividend Yield = Dividend Rate / Closing Price
        dividend_yield = safe_divide(dividend_rate, closing_price) * 100 if closing_price > 0 else 0
        
        # 7. OTHER METRICS
        # Working Capital = Current Assets - Current Liabilities
        working_capital = current_assets - current_liabilities
        
        # Operating Cash Flow to Sales
        operating_cashflow_to_sales = safe_divide(operating_cashflow, net_sales) * 100
        
        # Free Cash Flow = Operating Cash Flow + Investing Cash Flow
        free_cashflow = operating_cashflow + investing_cashflow
        
        # Current Assets to Total Assets
        current_assets_to_total = safe_divide(current_assets, total_assets) * 100
        
        # Long-term Debt to Total Assets
        long_term_debt_to_total = safe_divide(long_term_borrowings, total_assets) * 100
        
        # Equity to Assets Ratio
        equity_to_assets = safe_divide(total_equity, total_assets) * 100
        
        # Retained Earnings Ratio = Reserves / Total Equity
        retained_earnings_ratio = safe_divide(reserves, total_equity) * 100
        
        # Store values
        features['current_ratio'].append(current_ratio)
        features['quick_ratio'].append(quick_ratio)
        features['debt_to_asset_ratio'].append(debt_to_asset)
        features['debt_to_equity_ratio'].append(debt_to_equity)
        features['roe'].append(roe)
        features['roa'].append(roa)
        features['net_profit_margin'].append(net_profit_margin)
        features['operating_profit_margin'].append(operating_profit_margin)
        features['asset_turnover'].append(asset_turnover)
        features['revenue_growth_rate'].append(revenue_growth)
        features['profit_growth_rate'].append(profit_growth)
        features['pe_ratio'].append(pe_ratio)
        features['working_capital'].append(working_capital)
        features['total_assets'].append(total_assets)
        features['total_liabilities'].append(total_liabilities)
        features['operating_cashflow_to_sales'].append(operating_cashflow_to_sales)
        features['free_cashflow'].append(free_cashflow)
        features['inventory_turnover'].append(inventory_turnover)
        features['current_assets_to_total_assets'].append(current_assets_to_total)
        features['long_term_debt_to_total_assets'].append(long_term_debt_to_total)
        features['equity_to_assets_ratio'].append(equity_to_assets)
        features['eps_growth_rate'].append(eps_growth)
        features['price_to_book_ratio'].append(price_to_book)
        features['dividend_yield'].append(dividend_yield)
        features['retained_earnings_ratio'].append(retained_earnings_ratio)

print("\n3. Creating feature DataFrames...")

# Reshape data: Convert from list of values to DataFrame format
# Each feature needs to be reshaped: (num_companies * num_periods) -> (num_companies, num_periods)

for feature_name, values in features.items():
    # Reshape the flat list into a matrix
    values_array = np.array(values).reshape(len(company_names), len(time_periods))
    
    # Create DataFrame
    df_feature = pd.DataFrame(values_array, columns=time_periods)
    df_feature.insert(0, 'Company Name', company_names)
    
    # Replace inf and very large values with NaN, then fill with 0
    df_feature = df_feature.replace([np.inf, -np.inf], np.nan)
    df_feature = df_feature.fillna(0)
    
    # Clip extreme outliers (beyond 3 standard deviations for each column)
    for col in time_periods:
        col_data = df_feature[col]
        if len(col_data[col_data != 0]) > 0:
            mean = col_data[col_data != 0].mean()
            std = col_data[col_data != 0].std()
            if std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df_feature[col] = col_data.clip(lower=lower_bound, upper=upper_bound)
    
    # Save to CSV
    output_file = OTHER_DATA_PATH / f"{feature_name}.csv"
    df_feature.to_csv(output_file, index=False)
    print(f"   ✓ Saved {feature_name}.csv ({df_feature.shape[0]} companies, {df_feature.shape[1]-1} periods)")

print("\n4. Creating summary statistics...")

# Create a summary file with statistics for each feature
summary_data = []
for feature_name in features.keys():
    file_path = OTHER_DATA_PATH / f"{feature_name}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        numeric_cols = [col for col in df.columns if col != 'Company Name']
        all_values = df[numeric_cols].values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        all_values = all_values[all_values != 0]
        
        if len(all_values) > 0:
            summary_data.append({
                'Feature': feature_name,
                'Mean': np.mean(all_values),
                'Median': np.median(all_values),
                'Std': np.std(all_values),
                'Min': np.min(all_values),
                'Max': np.max(all_values),
                'Non_Zero_Count': len(all_values),
                'Total_Values': len(df[numeric_cols].values.flatten())
            })

summary_df = pd.DataFrame(summary_data)
summary_file = OTHER_DATA_PATH / "feature_summary_statistics.csv"
summary_df.to_csv(summary_file, index=False)
print(f"   ✓ Saved feature_summary_statistics.csv")

print("\n" + "=" * 80)
print("Feature Engineering Complete!")
print("=" * 80)
print(f"\nGenerated {len(features)} financial features:")
print("\nLiquidity Ratios:")
print("  - current_ratio")
print("  - quick_ratio")
print("\nLeverage Ratios:")
print("  - debt_to_asset_ratio")
print("  - debt_to_equity_ratio")
print("\nProfitability Ratios:")
print("  - roe (Return on Equity)")
print("  - roa (Return on Assets)")
print("  - net_profit_margin")
print("  - operating_profit_margin")
print("\nEfficiency Ratios:")
print("  - asset_turnover")
print("  - inventory_turnover")
print("\nGrowth Ratios:")
print("  - revenue_growth_rate")
print("  - profit_growth_rate")
print("  - eps_growth_rate")
print("\nMarket Ratios:")
print("  - pe_ratio")
print("  - price_to_book_ratio")
print("  - dividend_yield")
print("\nOther Metrics:")
print("  - working_capital")
print("  - total_assets")
print("  - total_liabilities")
print("  - operating_cashflow_to_sales")
print("  - free_cashflow")
print("  - current_assets_to_total_assets")
print("  - long_term_debt_to_total_assets")
print("  - equity_to_assets_ratio")
print("  - retained_earnings_ratio")
print(f"\nAll files saved to: {OTHER_DATA_PATH}")
print("=" * 80)


