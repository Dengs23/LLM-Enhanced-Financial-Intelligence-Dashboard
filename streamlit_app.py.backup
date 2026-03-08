# streamlit_app.py - CFO Insurance Risk Intelligence Platform v3.0
# Streamlit version for deployment

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import uuid
import re
import traceback
import warnings
import hashlib
import time
import json
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import openai
from openai import OpenAI

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page config
st.set_page_config(
    page_title="CFO Insurance Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1a237e;
        margin-bottom: 1rem;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a237e;
        margin-bottom: 0.25rem;
    }
    .kpi-subtitle {
        font-size: 0.85rem;
        color: #999;
    }
    
    /* Scenario cards */
    .scenario-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1a237e;
        margin: 0.5rem 0;
    }
    .scenario-card h4 {
        margin: 0 0 0.5rem 0;
        color: #1a237e;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .user-message {
        background: #1a237e;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    .assistant-message {
        background: #f0f2f5;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 3px dashed #1a237e;
        padding: 2rem;
        text-align: center;
        border-radius: 1rem;
        background: #f8f9fa;
        margin-bottom: 1rem;
    }
    
    /* Metrics badges */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #e8eaf6;
        color: #1a237e;
        border-radius: 2rem;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = 'insurance_risk.db'

def init_database():
    """Initialize database with complete schema"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS dashboards (
            id TEXT PRIMARY KEY,
            html TEXT,
            data TEXT,
            created TEXT,
            updated TEXT,
            currency TEXT,
            company_name TEXT,
            analysis_year INTEGER,
            version INTEGER DEFAULT 3
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            dashboard_id TEXT,
            timestamp TEXT,
            user_message TEXT,
            agent_response TEXT,
            FOREIGN KEY (dashboard_id) REFERENCES dashboards(id)
        )''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

init_database()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar"},
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "ZAR": {"symbol": "R", "name": "South African Rand"},
    "KES": {"symbol": "KSh", "name": "Kenyan Shilling"},
    "NGN": {"symbol": "₦", "name": "Nigerian Naira"}
}

def generate_id():
    """Generate unique ID"""
    return hashlib.md5(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:12]

def get_currency_symbol(currency_code):
    """Get currency symbol"""
    return CURRENCIES.get(currency_code, {}).get('symbol', '$')

def format_currency(value, currency_code='ZAR', decimals=0):
    """Format currency value"""
    if value is None or value == 0:
        return f"{get_currency_symbol(currency_code)}0"
    symbol = get_currency_symbol(currency_code)
    if abs(value) >= 1e6:
        return f"{symbol}{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:.{decimals}f}K"
    return f"{symbol}{value:.{decimals}f}"

def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers"""
    try:
        if denominator and denominator != 0:
            return numerator / denominator
        return default
    except:
        return default

def safe_abs(value):
    """Safely get absolute value"""
    try:
        if value is None or pd.isna(value):
            return 0
        return abs(float(value))
    except (ValueError, TypeError):
        return 0

# ============================================================================
# ENHANCED DATA PARSER WITH MULTI-YEAR SUPPORT
# ============================================================================

class InsuranceDataParser:
    """Parser for insurance financial statements with multi-year support"""
    
    def __init__(self):
        self.column_mappings = {
            # Premium mappings
            'gross_written_premium': [
                'gross written premium', 'gross premiums written', 'total premiums',
                'gross premium', 'premiums written', 'direct premiums',
                'gross premium written', 'premium revenue', 'gross written contributions'
            ],
            'net_written_premium': [
                'net written premium', 'net premiums written', 'net premium',
                'premiums written net'
            ],
            'earned_premium': [
                'earned premiums', 'premiums earned', 'net earned premium'
            ],
            
            # Claims mappings
            'claims_incurred': [
                'claims incurred', 'claims paid', 'total claims', 'losses incurred',
                'claims expense', 'underwriting claims', 'insurance claims',
                'claims cost', 'benefits paid', 'gross claims incurred'
            ],
            'outstanding_claims': [
                'outstanding claims', 'claims outstanding', 'claims reserves',
                'loss reserves', 'claims provision'
            ],
            
            # Expense mappings
            'underwriting_expenses': [
                'underwriting expenses', 'operating expenses', 'total expenses',
                'administrative expenses', 'management expenses', 'operating costs'
            ],
            'commission_expenses': [
                'commission', 'commissions', 'commission expense', 'acquisition costs',
                'brokerage', 'agent commissions'
            ],
            
            # Investment income
            'investment_income': [
                'investment income', 'investment return', 'investment revenue',
                'income from investments', 'interest income', 'dividend income'
            ],
            
            # Balance sheet items
            'total_assets': [
                'total assets', 'assets', 'total assets', 'balance sheet total',
                'sum of assets'
            ],
            'total_liabilities': [
                'total liabilities', 'liabilities', 'total liabilities',
                'total creditors', 'sum of liabilities'
            ],
            'total_equity': [
                'total equity', 'shareholders funds', 'shareholders equity',
                'capital and reserves', 'net assets', 'policyholders surplus'
            ],
            'cash_equivalents': [
                'cash and cash equivalents', 'cash', 'bank balances',
                'cash at bank', 'cash on hand'
            ],
            'total_investments': [
                'total investments', 'investments', 'investment portfolio',
                'financial assets', 'invested assets'
            ],
            
            # Technical provisions
            'unearned_premium': [
                'unearned premiums', 'unearned premium reserve', 'upr'
            ],
            'life_fund': [
                'life fund', 'policyholder liabilities', 'life assurance fund'
            ],
            
            # Profit and loss
            'net_profit': [
                'net profit', 'net income', 'profit after tax', 'net earnings',
                'profit for the year', 'net surplus', 'net result', 'bottom line'
            ],
            'profit_before_tax': [
                'profit before tax', 'profit before taxation', 'pbt', 'pre-tax profit',
                'income before tax'
            ],
            'tax_expense': [
                'tax expense', 'income tax', 'taxation', 'tax charge'
            ],
            
            # Reinsurance
            'reinsurance_premiums_ceded': [
                'reinsurance premiums', 'premiums ceded', 'reinsurance expense',
                'ceded premiums'
            ]
        }
    
    def parse_excel(self, uploaded_file):
        """Parse Excel file and extract multi-year data"""
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            
            # Extract all years and metrics across sheets
            all_years_data = {}
            
            for sheet_name in excel_file.sheet_names:
                # Try different header positions
                for header_row in range(5):
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
                        
                        # Extract years from columns
                        years = self._extract_years(df)
                        if years:
                            # Extract data for each year
                            year_data = self._extract_metrics_by_year(df, years)
                            
                            for year, metrics in year_data.items():
                                if year not in all_years_data:
                                    all_years_data[year] = {}
                                all_years_data[year].update(metrics)
                            
                            break
                    except:
                        continue
            
            if not all_years_data:
                return self._handle_single_period(excel_file)
            
            # Find latest year
            latest_year = max(all_years_data.keys())
            
            # Use latest year data as current metrics
            current_metrics = all_years_data[latest_year]
            
            # Calculate derived metrics
            current_metrics = self._calculate_metrics(current_metrics)
            current_metrics['analysis_year'] = latest_year
            current_metrics['years_available'] = sorted(list(all_years_data.keys()))
            current_metrics['historical_data'] = {
                year: self._calculate_metrics(data) 
                for year, data in all_years_data.items()
            }
            
            # Extract trend data for key metrics
            current_metrics['trends'] = self._extract_trends(all_years_data)
            
            return {
                'success': True,
                'metrics': current_metrics,
                'latest_year': latest_year,
                'years_available': sorted(list(all_years_data.keys()))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_years(self, df):
        """Extract years from dataframe columns"""
        years = []
        
        for col in df.columns:
            col_str = str(col).lower()
            match = re.search(r'20\d{2}', col_str)
            if match:
                year = int(match.group())
                if year not in years:
                    years.append(year)
        
        if not years and len(df) > 0:
            first_row = df.iloc[0]
            for val in first_row:
                val_str = str(val).lower()
                match = re.search(r'20\d{2}', val_str)
                if match:
                    year = int(match.group())
                    if year not in years:
                        years.append(year)
        
        return sorted(years)
    
    def _extract_metrics_by_year(self, df, years):
        """Extract metrics for each year"""
        year_data = {year: {} for year in years}
        
        # Find column indices for each year
        year_cols = {}
        for i, col in enumerate(df.columns):
            col_str = str(col).lower()
            for year in years:
                if str(year) in col_str:
                    year_cols[year] = i
                    break
        
        # If no year columns found, use first numeric column for latest year
        if not year_cols and years:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                year_cols[years[0]] = df.columns.get_loc(numeric_cols[0])
        
        # Extract metrics
        desc_col = df.columns[0]
        
        for idx, row in df.iterrows():
            description = str(row[desc_col]).lower().strip()
            if pd.isna(description) or description == 'nan':
                continue
            
            for metric_key, search_terms in self.column_mappings.items():
                for term in search_terms:
                    if term.lower() in description:
                        for year, col_idx in year_cols.items():
                            try:
                                value = row[col_idx]
                                if pd.notna(value) and value != '':
                                    clean_value = self._clean_numeric_value(value)
                                    if clean_value is not None:
                                        year_data[year][metric_key] = clean_value
                            except:
                                pass
                        break
        
        return year_data
    
    def _clean_numeric_value(self, value):
        """Clean and convert numeric values"""
        try:
            if pd.isna(value):
                return None
            
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                value = re.sub(r'[R\$€£¥\s,]', '', value)
                if '(' in value and ')' in value:
                    value = '-' + value.replace('(', '').replace(')', '')
                value = re.sub(r'[^\d.-]', '', value)
                if value and value != '-':
                    return float(value)
            
            return None
        except:
            return None
    
    def _handle_single_period(self, excel_file):
        """Handle file with no years (single period)"""
        all_metrics = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            desc_col = df.columns[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                value_col = numeric_cols[0]
                
                for idx, row in df.iterrows():
                    description = str(row[desc_col]).lower().strip()
                    if pd.isna(description) or description == 'nan':
                        continue
                    
                    for metric_key, search_terms in self.column_mappings.items():
                        if metric_key in all_metrics:
                            continue
                            
                        for term in search_terms:
                            if term.lower() in description:
                                try:
                                    value = row[value_col]
                                    if pd.notna(value):
                                        clean_value = self._clean_numeric_value(value)
                                        if clean_value is not None:
                                            all_metrics[metric_key] = clean_value
                                            break
                                except:
                                    continue
        
        all_metrics = self._calculate_metrics(all_metrics)
        all_metrics['analysis_year'] = 'Current'
        
        return {
            'success': True,
            'metrics': all_metrics,
            'latest_year': 'Current',
            'years_available': ['Current']
        }
    
    def _calculate_metrics(self, metrics):
        """Calculate all derived metrics"""
        calculated = metrics.copy()
        
        claims = safe_abs(metrics.get('claims_incurred', 0))
        premium = safe_abs(metrics.get('gross_written_premium', 0))
        expenses = safe_abs(metrics.get('underwriting_expenses', 0))
        
        if claims > 0 and premium > 0:
            calculated['loss_ratio'] = (claims / premium) * 100
        
        if expenses > 0 and premium > 0:
            calculated['expense_ratio'] = (expenses / premium) * 100
        
        if 'loss_ratio' in calculated and 'expense_ratio' in calculated:
            calculated['combined_ratio'] = calculated['loss_ratio'] + calculated['expense_ratio']
        
        equity = safe_abs(metrics.get('total_equity', 0))
        liabilities = safe_abs(metrics.get('total_liabilities', 0))
        if liabilities > 0:
            calculated['capital_adequacy_ratio'] = equity / liabilities
        
        profit = safe_abs(metrics.get('net_profit', 0))
        if profit > 0 and equity > 0:
            calculated['roe'] = (profit / equity) * 100
        
        if profit > 0 and premium > 0:
            calculated['net_margin'] = (profit / premium) * 100
        
        calculated['parsed_fields'] = [k for k in metrics.keys() 
                                      if k not in ['loss_ratio', 'expense_ratio', 'combined_ratio', 
                                                  'capital_adequacy_ratio', 'roe', 'net_margin']]
        
        return calculated
    
    def _extract_trends(self, years_data):
        """Extract trends for key metrics"""
        trends = {}
        
        for metric in ['loss_ratio', 'combined_ratio', 'gross_written_premium', 'net_profit']:
            values = []
            years = sorted(years_data.keys())
            
            for year in years:
                if metric in years_data[year]:
                    if metric in ['gross_written_premium', 'net_profit']:
                        values.append(safe_abs(years_data[year][metric]))
                    else:
                        values.append(years_data[year][metric])
                elif metric == 'loss_ratio' and 'claims_incurred' in years_data[year] and 'gross_written_premium' in years_data[year]:
                    claims = safe_abs(years_data[year]['claims_incurred'])
                    premium = safe_abs(years_data[year]['gross_written_premium'])
                    if premium > 0:
                        lr = (claims / premium) * 100
                        values.append(lr)
                    else:
                        values.append(None)
                else:
                    values.append(None)
            
            valid_values = [v for v in values if v is not None]
            if len(valid_values) >= 2:
                direction = 'increasing' if valid_values[-1] > valid_values[0] else 'decreasing'
                change_pct = ((valid_values[-1] - valid_values[0]) / valid_values[0]) * 100 if valid_values[0] != 0 else 0
                
                trends[metric] = {
                    'direction': direction,
                    'change_percent': change_pct,
                    'values': dict(zip(years, values))
                }
        
        return trends

# ============================================================================
# DYNAMIC MACROECONOMIC MODEL
# ============================================================================

class DynamicMacroeconomicModel:
    """Macroeconomic model that adapts to the insurance data"""
    
    def __init__(self, metrics, country='ZA'):
        self.metrics = metrics
        self.country = country
        
        self.base_rates = {
            'ZA': {
                'inflation': 5.2,
                'repo_rate': 7.5,
                'prime_rate': 11.0,
                'gdp_growth': 0.8,
                'unemployment': 32.1,
                'usd_zar': 18.75
            },
            'US': {
                'inflation': 3.2,
                'fed_rate': 5.25,
                'prime_rate': 8.5,
                'gdp_growth': 2.1,
                'unemployment': 3.8,
                'eur_usd': 1.08
            },
            'UK': {
                'inflation': 4.0,
                'bank_rate': 5.25,
                'prime_rate': 8.75,
                'gdp_growth': 0.5,
                'unemployment': 4.2,
                'gbp_usd': 1.27
            }
        }
        
    def generate_comprehensive_data(self):
        """Generate macroeconomic data tailored to the insurance metrics"""
        
        rates = self.base_rates.get(self.country, self.base_rates['ZA'])
        
        premium = safe_abs(self.metrics.get('gross_written_premium', 50000000))
        claims = safe_abs(self.metrics.get('claims_incurred', 25000000))
        investments = safe_abs(self.metrics.get('total_investments', premium * 1.5))
        
        loss_ratio = self.metrics.get('loss_ratio', 50)
        claims_inflation_sensitivity = min(1.2, max(0.6, loss_ratio / 50))
        
        return {
            'inflation': {
                'cpi_current': rates['inflation'],
                'cpi_forecast_12m': round(rates['inflation'] * 0.95, 1),
                'trend': 'stable' if rates['inflation'] < 6 else 'elevated',
                'impact_on_claims': round(claims_inflation_sensitivity, 2),
                'monetary_loss_impact': round(claims * (rates['inflation'] / 100) * claims_inflation_sensitivity, 0)
            },
            'interest_rates': {
                'policy_rate': rates.get('repo_rate', rates.get('fed_rate', 5.0)),
                'prime_rate': rates.get('prime_rate', 8.5),
                'forecast_12m': round(rates.get('repo_rate', 7.5) * 0.90, 2),
                'investment_income_impact': round(investments * (rates.get('repo_rate', 7.5) / 100) * 0.6, 0)
            },
            'gdp_growth': {
                'current': rates['gdp_growth'],
                'forecast_12m': round(rates['gdp_growth'] * 1.5, 1),
                'recession_probability': 0.15 if rates['gdp_growth'] < 1 else 0.08,
                'expected_premium_impact': round(premium * 0.7 * (rates['gdp_growth'] / 100), 0)
            },
            'unemployment': {
                'current': rates['unemployment'],
                'forecast_12m': round(rates['unemployment'] * 0.97, 1)
            },
            'stress_scenarios': {
                'mild': {
                    'claims_impact': round(claims * 0.15, 0)
                },
                'moderate': {
                    'claims_impact': round(claims * 0.30, 0)
                },
                'severe': {
                    'claims_impact': round(claims * 0.50, 0)
                }
            }
        }

# ============================================================================
# SCENARIO ANALYSIS MODEL (SAM)
# ============================================================================

class ScenarioAnalysisModel:
    """Strategic scenario analysis for CFO decision-making"""
    
    def __init__(self, metrics, macro_model):
        self.metrics = metrics
        self.macro = macro_model.generate_comprehensive_data()
        
    def generate_strategic_scenarios(self):
        """Generate CFO-focused strategic scenarios"""
        
        premium = safe_abs(self.metrics.get('gross_written_premium', 50000000))
        claims = safe_abs(self.metrics.get('claims_incurred', 25000000))
        equity = safe_abs(self.metrics.get('total_equity', 30000000))
        
        loss_ratio = self.metrics.get('loss_ratio', 50)
        combined_ratio = self.metrics.get('combined_ratio', 90)
        
        scenarios = {
            'base': {
                'name': 'Base Case',
                'description': 'Current trajectory with stable conditions',
                'probability': 0.50,
                'financial_metrics': {
                    'loss_ratio': loss_ratio,
                    'combined_ratio': combined_ratio,
                    'net_profit': self.metrics.get('net_profit', 8000000),
                    'roe': self.metrics.get('roe', 25)
                }
            },
            'growth': {
                'name': 'Growth & Expansion',
                'description': 'Aggressive market expansion',
                'probability': 0.25,
                'financial_metrics': {
                    'loss_ratio': round(loss_ratio * 1.02, 1),
                    'combined_ratio': round(combined_ratio * 0.98, 1),
                    'net_profit': round(self.metrics.get('net_profit', 8000000) * 1.25, 0),
                    'roe': round(self.metrics.get('roe', 25) * 1.15, 1)
                },
                'strategic_actions': [
                    'Enter 2 new geographic markets',
                    'Increase marketing spend by 20%',
                    'Launch 3 new products'
                ],
                'capital_required': round(equity * 0.25, 0)
            },
            'efficiency': {
                'name': 'Operational Excellence',
                'description': 'Focus on expense reduction',
                'probability': 0.15,
                'financial_metrics': {
                    'loss_ratio': round(loss_ratio * 0.96, 1),
                    'combined_ratio': round(combined_ratio * 0.92, 1),
                    'net_profit': round(self.metrics.get('net_profit', 8000000) * 1.20, 0),
                    'roe': round(self.metrics.get('roe', 25) * 1.18, 1)
                },
                'strategic_actions': [
                    'Implement AI claims processing',
                    'Consolidate regional offices',
                    'Automate policy administration'
                ],
                'implementation_cost': round(equity * 0.12, 0)
            },
            'defensive': {
                'name': 'Capital Preservation',
                'description': 'Conservative approach',
                'probability': 0.10,
                'financial_metrics': {
                    'loss_ratio': round(loss_ratio * 0.94, 1),
                    'combined_ratio': round(combined_ratio * 0.95, 1),
                    'net_profit': round(self.metrics.get('net_profit', 8000000) * 0.85, 0),
                    'roe': round(self.metrics.get('roe', 25) * 0.80, 1)
                },
                'strategic_actions': [
                    'Reduce exposure in volatile lines',
                    'Increase reinsurance coverage',
                    'Build liquidity reserves'
                ]
            },
            'stress': {
                'name': 'Stress Scenario',
                'description': 'Severe economic downturn',
                'probability': 0.05,
                'financial_metrics': {
                    'loss_ratio': round(min(loss_ratio * 1.35, 95), 1),
                    'combined_ratio': round(min(combined_ratio * 1.25, 115), 1),
                    'net_profit': round(self.metrics.get('net_profit', 8000000) * 0.30, 0),
                    'roe': round(self.metrics.get('roe', 25) * 0.40, 1)
                },
                'strategic_actions': [
                    'Draw on contingent capital',
                    'File for premium rate increases',
                    'Emergency expense freeze'
                ]
            }
        }
        
        return scenarios

# ============================================================================
# DIRECT LLM INTEGRATION
# ============================================================================

class CFOLlmAssistant:
    """CFO Assistant using direct OpenAI API"""
    
    def __init__(self, metrics=None, macro_data=None, scenarios=None):
        self.metrics = metrics or {}
        self.macro_data = macro_data or {}
        self.scenarios = scenarios or {}
        
        self.system_prompt = """You are a CFO-focused Insurance Risk Intelligence Assistant.

Your role:
- Provide concise, professional, actionable insights
- Focus on key metrics: claims, loss ratio, combined ratio, assets, capital adequacy
- Use the provided data to answer questions
- Structure responses with clear sections and bullet points
- Be specific with numbers and amounts
- Highlight risks and opportunities

Be helpful but concise - CFOs are busy."""
    
    def update_data(self, metrics, macro_data, scenarios):
        """Update the assistant with latest data"""
        self.metrics = metrics
        self.macro_data = macro_data
        self.scenarios = scenarios
    
    def _format_context(self):
        """Format the context data for the LLM"""
        context = "CURRENT INSURANCE DATA:\n"
        
        metrics_dict = {
            'Gross Written Premium': format_currency(self.metrics.get('gross_written_premium', 0), 'ZAR'),
            'Claims Incurred': format_currency(self.metrics.get('claims_incurred', 0), 'ZAR'),
            'Loss Ratio': f"{self.metrics.get('loss_ratio', 0):.1f}%" if self.metrics.get('loss_ratio') else 'N/A',
            'Combined Ratio': f"{self.metrics.get('combined_ratio', 0):.1f}%" if self.metrics.get('combined_ratio') else 'N/A',
            'Total Assets': format_currency(self.metrics.get('total_assets', 0), 'ZAR'),
            'Total Equity': format_currency(self.metrics.get('total_equity', 0), 'ZAR'),
            'Capital Adequacy Ratio': f"{self.metrics.get('capital_adequacy_ratio', 0):.2f}x" if self.metrics.get('capital_adequacy_ratio') else 'N/A',
            'ROE': f"{self.metrics.get('roe', 0):.1f}%" if self.metrics.get('roe') else 'N/A'
        }
        
        for key, value in metrics_dict.items():
            if value != 'N/A' and value != 'R0':
                context += f"- {key}: {value}\n"
        
        if self.macro_data:
            context += "\nMACROECONOMIC ENVIRONMENT:\n"
            inf = self.macro_data.get('inflation', {})
            rates = self.macro_data.get('interest_rates', {})
            
            if inf:
                context += f"- Inflation: {inf.get('cpi_current', 'N/A')}%\n"
            if rates:
                context += f"- Interest Rate: {rates.get('policy_rate', 'N/A')}%\n"
        
        return context
    
    def get_response(self, query):
        """Get response from OpenAI"""
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            context = self._format_context()
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Here is the current data:\n\n{context}\n\nUser question: {query}\n\nProvide a helpful, concise response:"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return self._get_fallback_response(query)
    
    def _get_fallback_response(self, query):
        """Fallback when OpenAI is unavailable"""
        query_lower = query.lower()
        
        if 'claim' in query_lower:
            claims = self.metrics.get('claims_incurred', 0)
            lr = self.metrics.get('loss_ratio', 0)
            return f"""📋 **Claims Analysis**

Current Claims: {format_currency(claims, 'ZAR')}
Loss Ratio: {lr:.1f}%

Key Insights:
• {'Claims are within target' if lr < 60 else 'Claims above target - investigate'}"""
        
        elif 'loss ratio' in query_lower:
            lr = self.metrics.get('loss_ratio', 0)
            status = "Excellent" if lr < 55 else "Good" if lr < 65 else "Warning" if lr < 75 else "Critical"
            return f"""📊 **Loss Ratio: {lr:.1f}% ({status})**

Target range: 55-65%
Variance: {(lr - 60):+.1f}%

Action: {'Maintain current approach' if lr < 55 else 'Review pricing' if lr < 65 else 'Implement rate increases' if lr < 75 else 'Emergency review needed'}"""
        
        elif 'combined' in query_lower:
            cr = self.metrics.get('combined_ratio', 0)
            result = "Profit" if cr < 100 else "Loss"
            return f"""💰 **Combined Ratio: {cr:.1f}%**

Underwriting Result: {result} of {abs(100-cr):.1f}%"""
        
        else:
            return f"""📈 **CFO Dashboard Summary**

Loss Ratio: {self.metrics.get('loss_ratio', 0):.1f}%
Combined Ratio: {self.metrics.get('combined_ratio', 0):.1f}%
Assets: {format_currency(self.metrics.get('total_assets', 0), 'ZAR')}

Ask me about claims, loss ratio, combined ratio, or assets."""

# ============================================================================
# CHART GENERATION
# ============================================================================

class ChartGenerator:
    """Generate CFO-focused charts"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1a237e',
            'secondary': '#5c6bc0',
            'success': '#2e7d32',
            'warning': '#ed6c02',
            'danger': '#c62828'
        }
    
    def create_claims_trend(self, metrics):
        """Create claims trend chart"""
        trends = metrics.get('trends', {})
        
        fig = go.Figure()
        
        if 'loss_ratio' in trends and trends['loss_ratio'].get('values'):
            values = trends['loss_ratio']['values']
            years = list(values.keys())
            ratios = list(values.values())
            
            fig.add_trace(go.Scatter(
                x=years,
                y=ratios,
                mode='lines+markers',
                name='Loss Ratio',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=10)
            ))
            
            fig.add_hrect(y0=0, y1=55, line_width=0, fillcolor=self.colors['success'], opacity=0.1)
            fig.add_hrect(y0=55, y1=65, line_width=0, fillcolor=self.colors['secondary'], opacity=0.1)
            fig.add_hrect(y0=65, y1=75, line_width=0, fillcolor=self.colors['warning'], opacity=0.1)
            fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor=self.colors['danger'], opacity=0.1)
        
        fig.update_layout(
            title='Loss Ratio Trend',
            xaxis_title='Year',
            yaxis_title='Loss Ratio %',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_combined_ratio_chart(self, metrics):
        """Create combined ratio breakdown"""
        lr = metrics.get('loss_ratio', 0)
        er = metrics.get('expense_ratio', 0)
        cr = metrics.get('combined_ratio', 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Loss Ratio', 'Expense Ratio', 'Combined Ratio'],
            y=[lr, er, cr],
            marker_color=[self.colors['primary'], self.colors['secondary'], 
                         self.colors['danger'] if cr > 100 else self.colors['success']],
            text=[f'{lr:.1f}%', f'{er:.1f}%', f'{cr:.1f}%'],
            textposition='auto'
        ))
        
        fig.add_hline(y=100, line_dash="dash", line_color=self.colors['danger'])
        
        fig.update_layout(
            title='Combined Ratio Components',
            yaxis_title='Percentage (%)',
            height=400
        )
        
        return fig
    
    def create_asset_composition(self, metrics):
        """Create asset composition pie chart"""
        cash = safe_abs(metrics.get('cash_equivalents', 0))
        investments = safe_abs(metrics.get('total_investments', 0))
        other = safe_abs(metrics.get('total_assets', 0)) - cash - investments
        
        if other < 0:
            other = 0
        
        fig = go.Figure(data=[go.Pie(
            labels=['Cash & Equivalents', 'Investments', 'Other Assets'],
            values=[cash, investments, other],
            marker_colors=[self.colors['success'], self.colors['primary'], self.colors['secondary']],
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Asset Composition',
            height=400
        )
        
        return fig

# ============================================================================
# STREAMLIT APP
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'macro_data' not in st.session_state:
        st.session_state.macro_data = None
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = None
    if 'assistant' not in st.session_state:
        st.session_state.assistant = CFOLlmAssistant()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'dashboard_id' not in st.session_state:
        st.session_state.dashboard_id = None
    if 'company_name' not in st.session_state:
        st.session_state.company_name = "Insurance Company"
    if 'currency' not in st.session_state:
        st.session_state.currency = "ZAR"
    if 'country' not in st.session_state:
        st.session_state.country = "ZA"
    if 'latest_year' not in st.session_state:
        st.session_state.latest_year = "Current"

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏢 CFO Insurance Intelligence Platform</h1>
        <p>Direct LLM Integration | Multi-Year Analysis | Strategic Scenarios</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        st.session_state.company_name = st.text_input("Company Name", st.session_state.company_name)
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=3)
        with col2:
            st.session_state.country = st.selectbox("Country", ["ZA", "US", "UK"], index=0)
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📤 Upload Financial Data")
        uploaded_file = st.file_uploader(
            "Upload Excel file (Income Statement, Balance Sheet)",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file:
            with st.spinner("Parsing financial data..."):
                parser = InsuranceDataParser()
                result = parser.parse_excel(uploaded_file)
                
                if result['success']:
                    st.session_state.metrics = result['metrics']
                    st.session_state.latest_year = result['latest_year']
                    
                    st.success(f"✅ Successfully parsed data")
                    st.info(f"📅 Latest year: {result['latest_year']}")
                    st.info(f"📊 Years available: {', '.join(map(str, result['years_available']))}")
                    
                    # Generate macro and scenarios
                    macro_model = DynamicMacroeconomicModel(
                        st.session_state.metrics, 
                        st.session_state.country
                    )
                    st.session_state.macro_data = macro_model.generate_comprehensive_data()
                    
                    sam = ScenarioAnalysisModel(
                        st.session_state.metrics, 
                        macro_model
                    )
                    st.session_state.scenarios = sam.generate_strategic_scenarios()
                    
                    # Update assistant
                    st.session_state.assistant.update_data(
                        st.session_state.metrics,
                        st.session_state.macro_data,
                        st.session_state.scenarios
                    )
                    
                    # Generate dashboard ID
                    st.session_state.dashboard_id = generate_id()
                    
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # API Key status
        st.markdown("---")
        st.markdown("### 🔑 OpenAI Status")
        try:
            if st.secrets["OPENAI_API_KEY"]:
                st.success("✅ OpenAI configured")
        except:
            st.warning("⚠️ Using fallback responses (no API key)")
    
    # Main content area
    if st.session_state.metrics:
        display_dashboard()
    else:
        display_welcome()

def display_welcome():
    """Display welcome screen when no data is loaded"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">📊 Step 1</div>
            <div class="kpi-value">Upload Data</div>
            <div class="kpi-subtitle">Excel files with financials</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🤖 Step 2</div>
            <div class="kpi-value">AI Analysis</div>
            <div class="kpi-subtitle">LLM-powered insights</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">🎯 Step 3</div>
            <div class="kpi-value">Strategic View</div>
            <div class="kpi-subtitle">Scenarios & recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: #f8f9fa; border-radius: 1rem;">
        <h3>📁 Upload your financial statements to begin</h3>
        <p style="color: #666;">Supports multi-year data with auto-detection of latest year</p>
        <p style="color: #999; font-size: 0.9rem; margin-top: 1rem;">
            Key metrics: Premiums, Claims, Loss Ratio, Combined Ratio, Assets, Equity
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard():
    """Display the main dashboard"""
    
    # Dashboard header
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 1rem; margin-bottom: 1rem; border-left: 4px solid #1a237e;">
        <h2 style="margin: 0; color: #1a237e;">{st.session_state.company_name}</h2>
        <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
            <span class="metric-badge">📅 Analysis Year: {st.session_state.latest_year}</span>
            <span class="metric-badge">💰 Currency: {st.session_state.currency}</span>
            <span class="metric-badge">🌍 Country: {st.session_state.country}</span>
            <span class="metric-badge">🆔 {st.session_state.dashboard_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = st.session_state.metrics
    currency_sym = get_currency_symbol(st.session_state.currency)
    
    with col1:
        if 'gross_written_premium' in metrics:
            premium = safe_abs(metrics['gross_written_premium'])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📈 Gross Written Premium</div>
                <div class="kpi-value">{currency_sym}{premium/1e6:.1f}M</div>
                <div class="kpi-subtitle">Annual premium</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'claims_incurred' in metrics:
            claims = safe_abs(metrics['claims_incurred'])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">💰 Claims Incurred</div>
                <div class="kpi-value">{currency_sym}{claims/1e6:.1f}M</div>
                <div class="kpi-subtitle">Total claims</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'loss_ratio' in metrics:
            lr = metrics['loss_ratio']
            color = "#2e7d32" if lr < 55 else "#ed6c02" if lr < 75 else "#c62828"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📊 Loss Ratio</div>
                <div class="kpi-value" style="color: {color};">{lr:.1f}%</div>
                <div class="kpi-subtitle">Claims / Premium</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'combined_ratio' in metrics:
            cr = metrics['combined_ratio']
            color = "#2e7d32" if cr < 100 else "#c62828"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">⚖️ Combined Ratio</div>
                <div class="kpi-value" style="color: {color};">{cr:.1f}%</div>
                <div class="kpi-subtitle">Underwriting result</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Second KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'total_assets' in metrics:
            assets = safe_abs(metrics['total_assets'])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🏦 Total Assets</div>
                <div class="kpi-value">{currency_sym}{assets/1e6:.1f}M</div>
                <div class="kpi-subtitle">Balance sheet strength</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'total_equity' in metrics:
            equity = safe_abs(metrics['total_equity'])
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📊 Total Equity</div>
                <div class="kpi-value">{currency_sym}{equity/1e6:.1f}M</div>
                <div class="kpi-subtitle">Shareholder funds</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'capital_adequacy_ratio' in metrics:
            car = metrics['capital_adequacy_ratio']
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🛡️ Capital Adequacy</div>
                <div class="kpi-value">{car:.2f}x</div>
                <div class="kpi-subtitle">Equity / Liabilities</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'roe' in metrics:
            roe = metrics['roe']
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📈 Return on Equity</div>
                <div class="kpi-value">{roe:.1f}%</div>
                <div class="kpi-subtitle">Net profit / Equity</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### 📊 Visual Analytics")
    chart_gen = ChartGenerator()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if metrics.get('trends', {}).get('loss_ratio'):
            st.plotly_chart(chart_gen.create_claims_trend(metrics), use_container_width=True)
    
    with col2:
        if 'combined_ratio' in metrics:
            st.plotly_chart(chart_gen.create_combined_ratio_chart(metrics), use_container_width=True)
    
    # Asset composition chart
    if 'total_assets' in metrics:
        st.plotly_chart(chart_gen.create_asset_composition(metrics), use_container_width=True)
    
    # Scenarios
    if st.session_state.scenarios:
        st.markdown("### 🎯 Strategic Scenarios")
        
        cols = st.columns(len(st.session_state.scenarios))
        for idx, (key, scenario) in enumerate(st.session_state.scenarios.items()):
            if key != 'expected':
                with cols[idx]:
                    fm = scenario.get('financial_metrics', {})
                    st.markdown(f"""
                    <div class="scenario-card">
                        <h4>{scenario.get('name', key)}</h4>
                        <p style="color: #666; font-size: 0.9rem;">Prob: {scenario.get('probability', 0)*100:.0f}%</p>
                        <p style="margin: 0.5rem 0;">Combined: {fm.get('combined_ratio', 0):.1f}%</p>
                        <p style="margin: 0;">ROE: {fm.get('roe', 0):.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown("### 🤖 CFO LLM Assistant")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', 
                          unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input("Ask your CFO assistant...", key="chat_input", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Send")
        
        if submitted and query:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.get_response(query)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            st.rerun()
    
    # Suggested questions
    st.markdown("""
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
        <span class="metric-badge" onclick="navigator.clipboard.writeText('Claims analysis')">Claims analysis</span>
        <span class="metric-badge" onclick="navigator.clipboard.writeText('Loss ratio trend')">Loss ratio trend</span>
        <span class="metric-badge" onclick="navigator.clipboard.writeText('Combined ratio breakdown')">Combined ratio</span>
        <span class="metric-badge" onclick="navigator.clipboard.writeText('Asset composition')">Asset composition</span>
        <span class="metric-badge" onclick="navigator.clipboard.writeText('Compare scenarios')">Compare scenarios</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Export options
    with st.expander("📥 Export Options"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as PDF"):
                st.info("PDF export coming soon")
        with col2:
            if st.button("Export as HTML"):
                st.info("HTML export coming soon")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
