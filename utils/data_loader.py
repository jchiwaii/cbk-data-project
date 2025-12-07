"""
Data loading and preprocessing utilities for CBK Economic Dashboard
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Get the base directory
BASE_DIR = Path(__file__).parent.parent


def clean_numeric(value):
    """Clean numeric values by removing commas and converting to float"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(',', '').replace('"', '').strip()
        if cleaned == '' or cleaned == '-':
            return np.nan
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return np.nan


def load_gdp_data():
    """Load and process Annual GDP data"""
    df = pd.read_csv(BASE_DIR / 'Annual GDP.csv')
    df.columns = ['Year', 'Nominal_GDP', 'GDP_Growth', 'Real_GDP']
    
    for col in ['Nominal_GDP', 'GDP_Growth', 'Real_GDP']:
        df[col] = df[col].apply(clean_numeric)
    
    df['Year'] = df['Year'].apply(clean_numeric).astype(int)
    df = df.sort_values('Year').reset_index(drop=True)
    
    return df


def load_inflation_data():
    """Load and process Inflation Rates data"""
    df = pd.read_csv(BASE_DIR / 'Inflation Rates.csv')
    df.columns = ['Year', 'Month', 'Annual_Avg_Inflation', '12_Month_Inflation']
    
    # Clean numeric columns
    df['Year'] = df['Year'].apply(clean_numeric).astype(int)
    df['Annual_Avg_Inflation'] = df['Annual_Avg_Inflation'].apply(clean_numeric)
    df['12_Month_Inflation'] = df['12_Month_Inflation'].apply(clean_numeric)
    
    # Create date column
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['Month_Num'] = df['Month'].map(month_map)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month_Num'].astype(str) + '-01')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def load_public_debt_data():
    """Load and process Public Debt data"""
    df = pd.read_csv(BASE_DIR / 'Public Debt.csv', skiprows=3)
    df.columns = ['Year', 'Month', 'Domestic_Debt', 'External_Debt', 'Total_Debt']
    
    # Remove empty rows
    df = df.dropna(subset=['Year', 'Month'])
    
    # Clean numeric columns
    for col in ['Year', 'Month', 'Domestic_Debt', 'External_Debt', 'Total_Debt']:
        df[col] = df[col].apply(clean_numeric)
    
    df = df.dropna(subset=['Year', 'Month'])
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    
    # Create date column
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def load_domestic_debt_data():
    """Load and process Domestic Debt by Instrument data"""
    df = pd.read_csv(BASE_DIR / 'Domestic Debt by Instrument.csv', skiprows=2)
    
    # Clean column names
    df.columns = ['Fiscal_Year', 'Treasury_Bills', 'Treasury_Bonds', 'Government_Stocks', 
                  'Overdraft_CBK', 'Advances_Commercial', 'Other_Debt', 'Total_Domestic_Debt']
    
    # Remove empty rows and footnotes
    df = df.dropna(subset=['Fiscal_Year'])
    df = df[~df['Fiscal_Year'].astype(str).str.contains(r'\*|Source|shillings', na=False, regex=True)]
    
    # Parse fiscal year to date
    def parse_fiscal_date(val):
        if pd.isna(val):
            return pd.NaT
        val = str(val).strip()
        if '-' in val:
            # Format like "Sep-99", "Dec-99"
            try:
                return pd.to_datetime(val, format='%b-%y')
            except:
                pass
        if val.startswith(('1-', '2-', '3-', '4-', '5-', '6-', '7-', '8-', '9-')):
            # Format like "1-Jan", "2-Feb" etc (year prefix)
            parts = val.split('-')
            if len(parts) == 2:
                year_suffix = parts[0]
                month = parts[1]
                try:
                    year = 2000 + int(year_suffix) if int(year_suffix) < 50 else 1900 + int(year_suffix)
                    return pd.to_datetime(f"{month}-{year}", format='%b-%Y')
                except:
                    pass
        # Try parsing formats like "24-Jul", "25-Feb"
        if '-' in val:
            parts = val.split('-')
            if len(parts) == 2:
                try:
                    year_suffix = int(parts[0])
                    month = parts[1]
                    year = 2000 + year_suffix if year_suffix < 50 else 1900 + year_suffix
                    return pd.to_datetime(f"{month}-{year}", format='%b-%Y')
                except:
                    pass
        return pd.NaT
    
    df['Date'] = df['Fiscal_Year'].apply(parse_fiscal_date)
    df = df.dropna(subset=['Date'])
    
    # Clean numeric columns
    for col in ['Treasury_Bills', 'Treasury_Bonds', 'Government_Stocks', 
                'Overdraft_CBK', 'Advances_Commercial', 'Other_Debt', 'Total_Domestic_Debt']:
        df[col] = df[col].apply(clean_numeric)
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def load_revenue_expenditure_data():
    """Load and process Revenue and Expenditure data"""
    df = pd.read_csv(BASE_DIR / 'Revenue and Expenditure.csv', skiprows=6)
    
    # Set column names based on the structure
    df.columns = ['Fiscal_Year', 'Month', 'Import_Duty', 'Excise_Duty', 'Income_Tax', 
                  'VAT', 'Other_Tax', 'Total_Tax_Revenue', 'Non_Tax_Revenue', 'Total_Revenue',
                  'Programme_Grants', 'Project_Grants', 'Total_Grants',
                  'Domestic_Interest', 'Foreign_Interest', 'Wages_Salaries', 'Pensions',
                  'Other_Recurrent', 'Total_Recurrent', 'County_Transfer', 'Dev_Expenditure', 'Total_Expenditure']
    
    # Remove rows with NaN in key columns
    df = df.dropna(subset=['Fiscal_Year', 'Month'])
    
    # Clean fiscal year and month
    df['Year'] = df['Fiscal_Year'].apply(clean_numeric)
    df['Month'] = df['Month'].apply(clean_numeric)
    
    df = df.dropna(subset=['Year', 'Month'])
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    
    # Create date
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
    
    # Clean all numeric columns
    numeric_cols = ['Import_Duty', 'Excise_Duty', 'Income_Tax', 'VAT', 'Other_Tax',
                    'Total_Tax_Revenue', 'Non_Tax_Revenue', 'Total_Revenue',
                    'Programme_Grants', 'Project_Grants', 'Total_Grants',
                    'Domestic_Interest', 'Foreign_Interest', 'Wages_Salaries', 'Pensions',
                    'Other_Recurrent', 'Total_Recurrent', 'County_Transfer', 'Dev_Expenditure', 'Total_Expenditure']
    
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def get_latest_metrics():
    """Get latest values for key metrics"""
    gdp = load_gdp_data()
    inflation = load_inflation_data()
    debt = load_public_debt_data()
    fiscal = load_revenue_expenditure_data()
    
    latest_gdp = gdp.iloc[-1]
    latest_inflation = inflation.iloc[0]  # Data is sorted desc by date
    latest_debt = debt.iloc[-1]
    
    # Get latest fiscal year data (June is fiscal year end)
    latest_fiscal = fiscal[fiscal['Month'] == 6].iloc[-1] if len(fiscal[fiscal['Month'] == 6]) > 0 else fiscal.iloc[-1]
    
    return {
        'gdp': {
            'year': int(latest_gdp['Year']),
            'nominal': latest_gdp['Nominal_GDP'],
            'real': latest_gdp['Real_GDP'],
            'growth': latest_gdp['GDP_Growth']
        },
        'inflation': {
            'date': latest_inflation['Date'],
            'rate': latest_inflation['12_Month_Inflation'],
            'annual_avg': latest_inflation['Annual_Avg_Inflation']
        },
        'debt': {
            'date': latest_debt['Date'],
            'total': latest_debt['Total_Debt'],
            'domestic': latest_debt['Domestic_Debt'],
            'external': latest_debt['External_Debt']
        },
        'fiscal': {
            'date': latest_fiscal['Date'],
            'revenue': latest_fiscal['Total_Revenue'],
            'expenditure': latest_fiscal['Total_Expenditure'],
            'balance': (latest_fiscal['Total_Revenue'] or 0) - (latest_fiscal['Total_Expenditure'] or 0)
        }
    }


def get_annual_fiscal_summary():
    """Aggregate fiscal data by fiscal year (July-June)"""
    df = load_revenue_expenditure_data()
    
    # Get June data for each year (cumulative fiscal year totals)
    june_data = df[df['Month'] == 6].copy()
    june_data['Fiscal_Year_End'] = june_data['Year']
    
    return june_data[['Fiscal_Year_End', 'Total_Revenue', 'Total_Expenditure', 
                      'Import_Duty', 'Excise_Duty', 'Income_Tax', 'VAT', 
                      'Total_Tax_Revenue', 'Non_Tax_Revenue']].dropna()


def calculate_debt_to_gdp():
    """Calculate debt to GDP ratio over time"""
    gdp = load_gdp_data()
    debt = load_public_debt_data()
    
    # Get year-end debt figures (December or latest available)
    debt['Year'] = debt['Date'].dt.year
    yearly_debt = debt.groupby('Year').last().reset_index()
    
    # Merge with GDP
    merged = yearly_debt.merge(gdp[['Year', 'Nominal_GDP']], on='Year', how='inner')
    merged['Debt_to_GDP'] = (merged['Total_Debt'] / merged['Nominal_GDP']) * 100
    
    return merged[['Year', 'Total_Debt', 'Nominal_GDP', 'Debt_to_GDP']]


if __name__ == '__main__':
    # Test the data loaders
    print("Testing data loaders...")
    
    print("\n1. GDP Data:")
    gdp = load_gdp_data()
    print(gdp.head())
    
    print("\n2. Inflation Data:")
    inflation = load_inflation_data()
    print(inflation.head())
    
    print("\n3. Public Debt Data:")
    debt = load_public_debt_data()
    print(debt.head())
    
    print("\n4. Domestic Debt Data:")
    dom_debt = load_domestic_debt_data()
    print(dom_debt.head())
    
    print("\n5. Revenue & Expenditure Data:")
    fiscal = load_revenue_expenditure_data()
    print(fiscal.head())
    
    print("\n6. Latest Metrics:")
    metrics = get_latest_metrics()
    print(metrics)

