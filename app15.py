# app15.py - Insurance Risk Intelligence Platform v3.0
# CFO-Focused Conversational AI with Direct LLM Integration
# Run on: http://localhost:5014

from flask import Flask, request, jsonify, render_template, url_for, render_template_string, session
from flask_cors import CORS
from flask_socketio import SocketIO
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import uuid
import secrets
from scipy import stats
import sqlite3
import re
import traceback
import warnings
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple

# OpenAI direct import (no LangChain)
import openai
from openai import OpenAI

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)
CORS(app)
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create directories
os.makedirs('templates', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('exports', exist_ok=True)

# OpenAI API Key (set this in environment variables)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
client = OpenAI(api_key=OPENAI_API_KEY)

# Currency configuration
CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar"},
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "ZAR": {"symbol": "R", "name": "South African Rand"},
    "KES": {"symbol": "KSh", "name": "Kenyan Shilling"},
    "NGN": {"symbol": "₦", "name": "Nigerian Naira"}
}

# ============================================================================
# DATABASE SETUP
# ============================================================================

DB_PATH = os.path.join(os.path.dirname(__file__), 'insurance_risk_v3.db')

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
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database error: {e}")

init_database()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_id():
    """Generate unique ID"""
    return hashlib.md5(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:12]

def get_currency_symbol(currency_code):
    """Get currency symbol"""
    return CURRENCIES.get(currency_code, {}).get('symbol', '$')

def format_currency(value, currency_code='ZAR', decimals=0):
    """Format currency value"""
    if value is None or value == 0:
        return "R0"
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
            float_val = float(value)
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
    def _calculate_metrics(self, metrics):
        """Calculate all derived metrics"""
        calculated = metrics.copy()
        
        # Ensure claims incurred is positive for ratio calculations
        claims = safe_abs(metrics.get('claims_incurred', 0))
        premium = safe_abs(metrics.get('gross_written_premium', 0))
        expenses = safe_abs(metrics.get('underwriting_expenses', 0))
        
        # Loss ratio = Claims / Premium (using absolute values)
        if claims > 0 and premium > 0:
            calculated['loss_ratio'] = (claims / premium) * 100
        
        # Expense ratio = Expenses / Premium (using absolute values)
        if expenses > 0 and premium > 0:
            calculated['expense_ratio'] = (expenses / premium) * 100
        
        # Combined ratio = Loss ratio + Expense ratio
        if 'loss_ratio' in calculated and 'expense_ratio' in calculated:
            calculated['combined_ratio'] = calculated['loss_ratio'] + calculated['expense_ratio']
        
        # Capital adequacy ratio
        equity = safe_abs(metrics.get('total_equity', 0))
        liabilities = safe_abs(metrics.get('total_liabilities', 0))
        if liabilities > 0:
            calculated['capital_adequacy_ratio'] = equity / liabilities
        
        # Return on Equity (ROE)
        profit = safe_abs(metrics.get('net_profit', 0))
        if profit > 0 and equity > 0:
            calculated['roe'] = (profit / equity) * 100
        
        # Net profit margin
        if profit > 0 and premium > 0:
            calculated['net_margin'] = (profit / premium) * 100
        
        # Track parsed fields
        calculated['parsed_fields'] = [k for k in metrics.keys() 
                                      if k not in ['loss_ratio', 'expense_ratio', 'combined_ratio', 
                                                  'capital_adequacy_ratio', 'roe', 'net_margin']]
        
        return calculated    
    def parse_excel(self, file_path):
        """Parse Excel file and extract multi-year data"""
        try:
            excel_file = pd.ExcelFile(file_path)
            print(f"\n📊 Sheets found: {excel_file.sheet_names}")
            
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
                            print(f"  📅 Sheet '{sheet_name}' - Years found: {years}")
                            
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
                print("⚠️ No years found - treating as single period")
                return self._handle_single_period(excel_file)
            
            # Find latest year
            latest_year = max(all_years_data.keys())
            print(f"\n✅ Latest year detected: {latest_year}")
            
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
            
            print(f"✅ Parsed {len(current_metrics.get('parsed_fields', []))} metrics from {latest_year}")
            
            return {
                'success': True,
                'metrics': current_metrics,
                'latest_year': latest_year,
                'years_available': sorted(list(all_years_data.keys()))
            }
            
        except Exception as e:
            print(f"❌ Parse error: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _extract_years(self, df):
        """Extract years from dataframe columns"""
        years = []
        
        for col in df.columns:
            col_str = str(col).lower()
            # Look for years in column names
            match = re.search(r'20\d{2}', col_str)
            if match:
                year = int(match.group())
                if year not in years:
                    years.append(year)
        
        # Also check first row for years
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
        desc_col = df.columns[0]  # Assume first column is description
        
        for idx, row in df.iterrows():
            description = str(row[desc_col]).lower().strip()
            if pd.isna(description) or description == 'nan':
                continue
            
            for metric_key, search_terms in self.column_mappings.items():
                for term in search_terms:
                    if term.lower() in description:
                        # Get value for each year
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
                # Remove currency symbols, commas, and spaces
                value = re.sub(r'[R\$€£¥\s,]', '', value)
                # Handle negative in brackets
                if '(' in value and ')' in value:
                    value = '-' + value.replace('(', '').replace(')', '')
                # Remove any remaining non-numeric except decimal and minus
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
            
            # Use first column as description, find numeric column
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
    
    # Ensure claims incurred is positive for ratio calculations
            claims = safe_abs(metrics.get('claims_incurred', 0))
            premium = safe_abs(metrics.get('gross_written_premium', 0))
            expenses = safe_abs(metrics.get('underwriting_expenses', 0))
    
    # Loss ratio = Claims / Premium (using absolute values)
            if claims > 0 and premium > 0:
               calculated['loss_ratio'] = (claims / premium) * 100
    
    # Expense ratio = Expenses / Premium (using absolute values)
            if expenses > 0 and premium > 0:
               calculated['expense_ratio'] = (expenses / premium) * 100
    
    # Combined ratio = Loss ratio + Expense ratio
            if 'loss_ratio' in calculated and 'expense_ratio' in calculated:
                calculated['combined_ratio'] = calculated['loss_ratio'] + calculated['expense_ratio']
    
    # Capital adequacy ratio
            equity = safe_abs(metrics.get('total_equity', 0))
            liabilities = safe_abs(metrics.get('total_liabilities', 0))
            if liabilities > 0:
               calculated['capital_adequacy_ratio'] = equity / liabilities
    
    # Return on Equity (ROE)
            profit = safe_abs(metrics.get('net_profit', 0))
            if profit > 0 and equity > 0:
               calculated['roe'] = (profit / equity) * 100
    
    # Net profit margin
            if profit > 0 and premium > 0:
               calculated['net_margin'] = (profit / premium) * 100
    
    # Track parsed fields
            calculated['parsed_fields'] = [k for k in metrics.keys() 
            if k not in ['loss_ratio', 'expense_ratio', 'combined_ratio', 'capital_adequacy_ratio', 'roe', 'net_margin']]
    
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
            # Calculate trend direction
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
        
        # Base rates (could be enhanced with API calls)
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
        
        # Get base rates for country
        rates = self.base_rates.get(self.country, self.base_rates['ZA'])
        
        # Calculate insurance-specific impacts based on actual data
        premium = safe_abs(self.metrics.get('gross_written_premium', 50000000))
        claims = safe_abs(self.metrics.get('claims_incurred', 25000000))
        investments = safe_abs(self.metrics.get('total_investments', premium * 1.5))
        
        # Dynamic impact factors
        loss_ratio = self.metrics.get('loss_ratio', 50)
        claims_inflation_sensitivity = min(1.2, max(0.6, loss_ratio / 50))
        investment_sensitivity = 0.3 if investments > premium else 0.5
        
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
# DIRECT LLM INTEGRATION (NO LANGCHAIN)
# ============================================================================

class CFOLlmAssistant:
    """CFO Assistant using direct OpenAI API (no LangChain)"""
    
    def __init__(self, metrics=None, macro_data=None, scenarios=None):
        self.metrics = metrics or {}
        self.macro_data = macro_data or {}
        self.scenarios = scenarios or {}
        
        self.system_prompt = """You are a CFO-focused Insurance Risk Intelligence Assistant for South African insurers.

Your role:
- Provide concise, professional, actionable insights
- Focus on key metrics: claims, loss ratio, combined ratio, assets, capital adequacy
- Use the provided data to answer questions
- Structure responses with clear sections and bullet points
- Be specific with numbers and amounts (use Rands)
- Highlight risks and opportunities

Format numbers with R (e.g., R50M, R2.5K)
Be helpful but concise - CFOs are busy."""
    
    def update_data(self, metrics, macro_data, scenarios):
        """Update the assistant with latest data"""
        self.metrics = metrics
        self.macro_data = macro_data
        self.scenarios = scenarios
    
    def _format_context(self):
        """Format the context data for the LLM"""
        context = "CURRENT INSURANCE DATA:\n"
        
        # Key metrics
        metrics_dict = {
            'Gross Written Premium': format_currency(self.metrics.get('gross_written_premium', 0), 'ZAR'),
            'Claims Incurred': format_currency(self.metrics.get('claims_incurred', 0), 'ZAR'),
            'Loss Ratio': f"{self.metrics.get('loss_ratio', 0):.1f}%" if self.metrics.get('loss_ratio') else 'N/A',
            'Combined Ratio': f"{self.metrics.get('combined_ratio', 0):.1f}%" if self.metrics.get('combined_ratio') else 'N/A',
            'Total Assets': format_currency(self.metrics.get('total_assets', 0), 'ZAR'),
            'Total Equity': format_currency(self.metrics.get('total_equity', 0), 'ZAR'),
            'Capital Adequacy Ratio': f"{self.metrics.get('capital_adequacy_ratio', 0):.2f}x" if self.metrics.get('capital_adequacy_ratio') else 'N/A',
            'ROE': f"{self.metrics.get('roe', 0):.1f}%" if self.metrics.get('roe') else 'N/A',
            'Net Profit': format_currency(self.metrics.get('net_profit', 0), 'ZAR')
        }
        
        for key, value in metrics_dict.items():
            if value != 'N/A' and value != 'R0':
                context += f"- {key}: {value}\n"
        
        # Macro data
        if self.macro_data:
            context += "\nMACROECONOMIC ENVIRONMENT:\n"
            inf = self.macro_data.get('inflation', {})
            rates = self.macro_data.get('interest_rates', {})
            gdp = self.macro_data.get('gdp_growth', {})
            
            if inf:
                context += f"- Inflation: {inf.get('cpi_current', 'N/A')}%\n"
                if inf.get('monetary_loss_impact'):
                    context += f"- Inflation impact on claims: {format_currency(inf['monetary_loss_impact'], 'ZAR')}\n"
            
            if rates:
                context += f"- Interest Rate: {rates.get('policy_rate', 'N/A')}%\n"
                if rates.get('investment_income_impact'):
                    context += f"- Expected investment income: {format_currency(rates['investment_income_impact'], 'ZAR')}\n"
            
            if gdp:
                context += f"- GDP Growth: {gdp.get('current', 'N/A')}%\n"
        
        # Scenarios
        if self.scenarios:
            context += "\nSTRATEGIC SCENARIOS:\n"
            for key, scenario in list(self.scenarios.items())[:3]:  # Show top 3
                if key != 'expected':
                    fm = scenario.get('financial_metrics', {})
                    context += f"- {scenario.get('name')}: Combined {fm.get('combined_ratio', 0):.1f}%, ROE {fm.get('roe', 0):.1f}%\n"
        
        return context
    
    def get_response(self, query):
        """Get response from OpenAI"""
        try:
            context = self._format_context()
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Here is the current data:\n\n{context}\n\nUser question: {query}\n\nProvide a helpful, concise response:"}
            ]
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if available
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback response
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
• {'Claims are within target' if lr < 60 else 'Claims above target - investigate'}

Recommendations:
1. Review large loss activity
2. Analyze claims by product line"""
        
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

Underwriting Result: {result} of {abs(100-cr):.1f}%

Components:
• Loss Ratio: {self.metrics.get('loss_ratio', 0):.1f}%
• Expense Ratio: {self.metrics.get('expense_ratio', 0):.1f}%"""
        
        elif 'asset' in query_lower:
            assets = self.metrics.get('total_assets', 0)
            equity = self.metrics.get('total_equity', 0)
            return f"""🏦 **Asset Position**

Total Assets: {format_currency(assets, 'ZAR')}
Total Equity: {format_currency(equity, 'ZAR')}
Leverage: {(assets/equity if equity>0 else 0):.1f}x

Balance sheet is {'strong' if equity > assets*0.3 else 'adequate'}."""
        
        else:
            return f"""📈 **CFO Dashboard Summary**

Loss Ratio: {self.metrics.get('loss_ratio', 0):.1f}%
Combined Ratio: {self.metrics.get('combined_ratio', 0):.1f}%
Assets: {format_currency(self.metrics.get('total_assets', 0), 'ZAR')}

Ask me about:
• Claims analysis
• Loss ratio trends
• Combined ratio breakdown
• Asset composition
• Strategic scenarios
• Macroeconomic impacts"""

# Global assistant instance
cfo_assistant = CFOLlmAssistant()

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
            'danger': '#c62828',
            'background': '#f5f5f5',
            'card_bg': '#ffffff'
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
            
            # Add target bands
            fig.add_hrect(y0=0, y1=55, line_width=0, fillcolor=self.colors['success'], opacity=0.1)
            fig.add_hrect(y0=55, y1=65, line_width=0, fillcolor=self.colors['secondary'], opacity=0.1)
            fig.add_hrect(y0=65, y1=75, line_width=0, fillcolor=self.colors['warning'], opacity=0.1)
            fig.add_hrect(y0=75, y1=100, line_width=0, fillcolor=self.colors['danger'], opacity=0.1)
        
        fig.update_layout(
            title='Loss Ratio Trend',
            xaxis_title='Year',
            yaxis_title='Loss Ratio %',
            height=400,
            paper_bgcolor=self.colors['card_bg']
        )
        
        return fig.to_html(full_html=False)
    
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
            height=400,
            paper_bgcolor=self.colors['card_bg']
        )
        
        return fig.to_html(full_html=False)
    
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
            height=400,
            paper_bgcolor=self.colors['card_bg']
        )
        
        return fig.to_html(full_html=False)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>CFO Insurance Intelligence v3.0</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .hero {
            background: white;
            border-radius: 30px;
            padding: 60px;
            max-width: 900px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        }
        h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #1a237e, #5c6bc0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .subtitle {
            color: #666;
            font-size: 1.3em;
            margin-bottom: 30px;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 40px 0;
        }
        .feature {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 4px solid #1a237e;
        }
        .feature h3 { color: #1a237e; margin-bottom: 10px; }
        .feature p { color: #666; font-size: 0.9em; }
        .cta {
            display: inline-block;
            background: linear-gradient(135deg, #1a237e, #5c6bc0);
            color: white;
            padding: 20px 50px;
            border-radius: 50px;
            text-decoration: none;
            font-size: 1.2em;
            font-weight: bold;
            transition: transform 0.3s;
        }
        .cta:hover { transform: translateY(-3px); }
        .version {
            position: absolute;
            top: 20px;
            right: 20px;
            color: rgba(255,255,255,0.8);
            font-size: 0.9em;
        }
        .badge {
            background: #5c6bc0;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="version">v3.0 | Direct LLM | Port 5014</div>
    <div class="hero">
        <span class="badge">Direct OpenAI Integration</span>
        <h1>🏢 CFO Intelligence Platform</h1>
        <p class="subtitle">Insurance Analytics with Direct LLM Assistant</p>
        
        <div class="features">
            <div class="feature">
                <h3>📊 Smart Analytics</h3>
                <p>Claims, loss ratio, combined ratio, assets - all from YOUR data</p>
            </div>
            <div class="feature">
                <h3>🤖 Direct LLM</h3>
                <p>OpenAI-powered assistant (no LangChain overhead)</p>
            </div>
            <div class="feature">
                <h3>🎯 Strategic Scenarios</h3>
                <p>Growth, efficiency, defensive, and stress scenarios</p>
            </div>
            <div class="feature">
                <h3>🌍 Dynamic Macro</h3>
                <p>Economic impacts tailored to your portfolio</p>
            </div>
        </div>
        
        <div style="text-align: center;">
            <a href="/generator" class="cta">🚀 Launch Platform</a>
        </div>
    </div>
</body>
</html>
''')

@app.route('/generator')
def generator():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>CFO Dashboard Generator</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { color: #1a237e; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 30px; }
        
        .upload-area {
            border: 3px dashed #1a237e;
            padding: 40px;
            text-align: center;
            border-radius: 15px;
            cursor: pointer;
            background: #f8f9fa;
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        .upload-area:hover { background: #e8eaf6; }
        .upload-area.dragover { background: #c5cae9; }
        
        .config-row {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .config-item {
            flex: 1;
            min-width: 200px;
        }
        .config-item label {
            display: block;
            margin-bottom: 8px;
            color: #1a237e;
            font-weight: bold;
        }
        .config-item select, .config-item input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
        }
        
        .generate-btn {
            width: 100%;
            padding: 20px;
            background: linear-gradient(135deg, #1a237e, #5c6bc0);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px 0;
        }
        .generate-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1a237e;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .ai-assistant {
            background: linear-gradient(135deg, #1a237e, #283593);
            color: white;
            padding: 25px;
            border-radius: 20px;
            margin: 30px 0;
        }
        
        .chat-container {
            background: white;
            border-radius: 15px;
            height: 400px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            color: #333;
        }
        .message {
            margin: 10px 0;
            padding: 12px 18px;
            border-radius: 20px;
            max-width: 80%;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user-message {
            background: #1a237e;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        .assistant-message {
            background: #f0f2f5;
            color: #333;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        .chat-input-area {
            display: flex;
            gap: 10px;
            padding: 15px;
            border-top: 1px solid #e0e0e0;
        }
        .chat-input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        .chat-input:focus { border-color: #1a237e; }
        .send-btn {
            padding: 12px 25px;
            background: #1a237e;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .suggestions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 10px 0;
            padding: 0 15px;
        }
        .suggestion-chip {
            padding: 8px 15px;
            background: #e8eaf6;
            color: #1a237e;
            border-radius: 20px;
            font-size: 0.85em;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion-chip:hover { background: #c5cae9; }
        
        .dashboard-preview {
            margin-top: 30px;
            display: none;
        }
        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .typing-indicator {
            display: flex;
            gap: 5px;
            padding: 15px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏢 CFO Dashboard Generator</h1>
        <p class="subtitle">Upload your financial statements - we'll auto-detect the latest year</p>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()" 
             ondrop="dropHandler(event)" ondragover="dragOverHandler(event)">
            <div style="font-size: 48px; margin-bottom: 15px;">📊</div>
            <h3>Drop Excel file here or click to upload</h3>
            <p style="color: #666; margin-top: 10px;">Income Statement, Balance Sheet, or Combined</p>
            <input type="file" id="fileInput" accept=".xlsx,.xls" style="display: none;" onchange="handleFile(this.files[0])">
            <div id="fileName" style="margin-top: 15px; color: #1a237e; font-weight: bold;"></div>
        </div>
        
        <div class="config-row">
            <div class="config-item">
                <label>Company Name</label>
                <input type="text" id="companyName" value="Insurance Company" placeholder="Enter company name">
            </div>
            <div class="config-item">
                <label>Currency</label>
                <select id="currency">
                    <option value="ZAR" selected>ZAR (R) - South Africa</option>
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="GBP">GBP (£)</option>
                </select>
            </div>
            <div class="config-item">
                <label>Country (for macro)</label>
                <select id="country">
                    <option value="ZA" selected>South Africa</option>
                    <option value="US">United States</option>
                    <option value="UK">United Kingdom</option>
                </select>
            </div>
        </div>
        
        <button class="generate-btn" onclick="generateDashboard()" id="generateBtn">
            🚀 Generate CFO Dashboard
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>Analyzing Your Data...</h3>
            <p id="loadingText">Detecting latest year and calculating metrics...</p>
        </div>
        
        <div class="ai-assistant">
            <h3>🤖 CFO LLM Assistant</h3>
            <p style="margin-bottom: 15px; opacity: 0.9;">Powered by OpenAI (direct integration, no LangChain)</p>
            
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant-message">
                        👋 Hello CFO! I'm your AI assistant. Ask me about:
                        <ul style="margin-top: 10px; padding-left: 20px;">
                            <li><strong>Claims:</strong> "Claims analysis"</li>
                            <li><strong>Loss Ratio:</strong> "Loss ratio trend"</li>
                            <li><strong>Combined Ratio:</strong> "Combined ratio breakdown"</li>
                            <li><strong>Assets:</strong> "Asset composition"</li>
                            <li><strong>Scenarios:</strong> "Compare growth vs defensive"</li>
                            <li><strong>Macro:</strong> "Inflation impact on claims"</li>
                        </ul>
                    </div>
                </div>
                
                <div class="suggestions" id="suggestions">
                    <span class="suggestion-chip" onclick="sendQuick('Claims analysis')">Claims analysis</span>
                    <span class="suggestion-chip" onclick="sendQuick('Loss ratio trend')">Loss ratio</span>
                    <span class="suggestion-chip" onclick="sendQuick('Combined ratio breakdown')">Combined ratio</span>
                    <span class="suggestion-chip" onclick="sendQuick('Asset composition')">Assets</span>
                    <span class="suggestion-chip" onclick="sendQuick('Compare growth vs defensive scenarios')">Scenarios</span>
                    <span class="suggestion-chip" onclick="sendQuick('Inflation impact')">Inflation</span>
                </div>
                
                <div class="chat-input-area">
                    <input type="text" class="chat-input" id="chatInput" 
                           placeholder="Ask your CFO assistant..." 
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button class="send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
        
        <div class="dashboard-preview" id="dashboardPreview">
            <div class="preview-header">
                <h3>📊 Your Dashboard</h3>
            </div>
            <iframe id="previewFrame"></iframe>
        </div>
    </div>
    
    <script>
        let currentDashboardId = null;
        let uploadedData = null;
        
        function dragOverHandler(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }
        
        function dropHandler(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        }
        
        async function handleFile(file) {
            if (!file) return;
            document.getElementById('fileName').textContent = '📄 ' + file.name;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const res = await fetch('/upload_excel', { method: 'POST', body: formData });
                const result = await res.json();
                
                if (result.success) {
                    uploadedData = result;
                    showNotification(`✅ Found data for year: ${result.latest_year}`);
                } else {
                    showNotification('❌ Upload failed: ' + result.error);
                }
            } catch (e) {
                showNotification('❌ Error: ' + e.message);
            }
        }
        
        function showNotification(msg) {
            const div = document.createElement('div');
            div.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #333; color: white; padding: 15px 25px; border-radius: 10px; z-index: 1000;';
            div.textContent = msg;
            document.body.appendChild(div);
            setTimeout(() => div.remove(), 3000);
        }
        
        async function generateDashboard() {
            if (!uploadedData) {
                showNotification('❌ Please upload a file first');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const loadingText = document.getElementById('loadingText');
            
            btn.disabled = true;
            loading.style.display = 'block';
            
            const steps = [
                'Processing your financial data...',
                'Detecting latest year...',
                'Calculating key metrics...',
                'Running scenario analysis...',
                'Building macroeconomic model...',
                'Generating dashboard...'
            ];
            
            let stepIndex = 0;
            const interval = setInterval(() => {
                if (stepIndex < steps.length) {
                    loadingText.textContent = steps[stepIndex];
                    stepIndex++;
                }
            }, 800);
            
            try {
                const res = await fetch('/generate_dashboard', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        uploaded_data: uploadedData.metrics,
                        latest_year: uploadedData.latest_year,
                        company_name: document.getElementById('companyName').value,
                        currency: document.getElementById('currency').value,
                        country: document.getElementById('country').value
                    })
                });
                
                clearInterval(interval);
                const result = await res.json();
                
                loading.style.display = 'none';
                btn.disabled = false;
                
                if (result.success) {
                    currentDashboardId = result.dashboard_id;
                    document.getElementById('dashboardPreview').style.display = 'block';
                    document.getElementById('previewFrame').src = result.share_link;
                    
                    addMessage('assistant', '✅ Dashboard generated! Using data from **' + uploadedData.latest_year + '**. Ask me anything about your metrics.');
                    
                    document.getElementById('dashboardPreview').scrollIntoView({ behavior: 'smooth' });
                } else {
                    showNotification('❌ Error: ' + result.error);
                }
            } catch (e) {
                clearInterval(interval);
                loading.style.display = 'none';
                btn.disabled = false;
                showNotification('❌ Error: ' + e.message);
            }
        }
        
        function sendQuick(text) {
            document.getElementById('chatInput').value = text;
            sendMessage();
        }
        
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            const typingId = 'typing-' + Date.now();
            showTyping(typingId);
            
            try {
                const res = await fetch('/ask_agent', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: message,
                        dashboard_id: currentDashboardId
                    })
                });
                
                removeTyping(typingId);
                const result = await res.json();
                
                if (result.success) {
                    addMessage('assistant', result.response);
                } else {
                    addMessage('assistant', '❌ Error: ' + result.error);
                }
            } catch (e) {
                removeTyping(typingId);
                addMessage('assistant', '❌ Error connecting to assistant');
            }
        }
        
        function addMessage(role, content) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role + '-message';
            messageDiv.innerHTML = content.replace(/\\n/g, '<br>');
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function showTyping(id) {
            const messagesDiv = document.getElementById('chatMessages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.id = id;
            typingDiv.innerHTML = '<span></span><span></span><span></span>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function removeTyping(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }
    </script>
</body>
</html>
''')

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    temp_path = os.path.join('uploads', f"temp_{uuid.uuid4()}.xlsx")
    file.save(temp_path)
    
    parser = InsuranceDataParser()
    result = parser.parse_excel(temp_path)
    
    os.remove(temp_path)
    
    if result['success']:
        return jsonify({
            'success': True,
            'metrics': result['metrics'],
            'latest_year': result['latest_year'],
            'years_available': result['years_available']
        })
    else:
        return jsonify({'success': False, 'error': result.get('error', 'Parse failed')})

@app.route('/generate_dashboard', methods=['POST'])
def generate_dashboard():
    try:
        data = request.json
        uploaded_metrics = data.get('uploaded_data', {})
        latest_year = data.get('latest_year', 'Current')
        company_name = data.get('company_name', 'Insurance Company')
        currency = data.get('currency', 'ZAR')
        country = data.get('country', 'ZA')
        
        # Generate dashboard ID
        dashboard_id = generate_id()
        
        # Run dynamic macroeconomic model
        macro_model = DynamicMacroeconomicModel(uploaded_metrics, country)
        macro_data = macro_model.generate_comprehensive_data()
        
        # Run scenario analysis
        sam = ScenarioAnalysisModel(uploaded_metrics, macro_model)
        scenarios = sam.generate_strategic_scenarios()
        
        # Update assistant with data
        cfo_assistant.update_data(uploaded_metrics, macro_data, scenarios)
        
        # Store in session
        session['current_dashboard_data'] = {'metrics': uploaded_metrics}
        session['macroeconomic_data'] = macro_data
        session['scenario_analysis'] = scenarios
        
        # Generate charts
        chart_gen = ChartGenerator()
        
        # Build dashboard HTML
        html = build_dashboard_html(
            dashboard_id, company_name, currency, latest_year,
            uploaded_metrics, chart_gen, scenarios, macro_data
        )
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO dashboards 
                     (id, html, data, created, currency, company_name, analysis_year) 
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (dashboard_id, html, json.dumps({'metrics': uploaded_metrics}),
                   datetime.now().isoformat(), currency, company_name, 
                   latest_year if latest_year != 'Current' else 0))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'dashboard_id': dashboard_id,
            'share_link': url_for('view_dashboard', dashboard_id=dashboard_id, _external=True)
        })
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def build_dashboard_html(dashboard_id, company, currency, latest_year,
                        metrics, chart_gen, scenarios, macro_data):
    """Build CFO-focused dashboard HTML"""
    
    currency_symbol = get_currency_symbol(currency)
    
    # KPI cards
    def kpi_card(title, value, subtitle, color_class="primary"):
        if value is None or value == '' or value == 'R0':
            return ''
        return f'''
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        '''
    
    kpi_cards = []
    
    # Premium
    if 'gross_written_premium' in metrics:
        kpi_cards.append(kpi_card('Gross Written Premium', 
                                  f"{currency_symbol}{safe_abs(metrics['gross_written_premium'])/1e6:.1f}M", 
                                  f'Annual premium'))
    
    # Claims
    if 'claims_incurred' in metrics:
        kpi_cards.append(kpi_card('Claims Incurred', 
                                  f"{currency_symbol}{safe_abs(metrics['claims_incurred'])/1e6:.1f}M", 
                                  f'Total claims'))
    
    # Loss Ratio
    if 'loss_ratio' in metrics:
        lr = metrics['loss_ratio']
        kpi_cards.append(kpi_card('Loss Ratio', 
                                  f"{lr:.1f}%", 
                                  f'Claims / Premium'))
    
    # Combined Ratio
    if 'combined_ratio' in metrics:
        cr = metrics['combined_ratio']
        kpi_cards.append(kpi_card('Combined Ratio', 
                                  f"{cr:.1f}%", 
                                  f'Underwriting result'))
    
    # Total Assets
    if 'total_assets' in metrics:
        kpi_cards.append(kpi_card('Total Assets', 
                                  f"{currency_symbol}{safe_abs(metrics['total_assets'])/1e6:.1f}M", 
                                  f'Balance sheet strength'))
    
    # Capital Adequacy
    if 'capital_adequacy_ratio' in metrics:
        car = metrics['capital_adequacy_ratio']
        kpi_cards.append(kpi_card('Capital Adequacy', 
                                  f"{car:.2f}x", 
                                  f'Equity / Liabilities'))
    
    # ROE
    if 'roe' in metrics:
        kpi_cards.append(kpi_card('Return on Equity', 
                                  f"{metrics['roe']:.1f}%", 
                                  f'Net profit / Equity'))
    
    # Generate charts
    charts_html = ""
    if metrics.get('trends', {}).get('loss_ratio'):
        charts_html += f'<div class="chart-box">{chart_gen.create_claims_trend(metrics)}</div>'
    
    if 'combined_ratio' in metrics:
        charts_html += f'<div class="chart-box">{chart_gen.create_combined_ratio_chart(metrics)}</div>'
    
    if 'total_assets' in metrics:
        charts_html += f'<div class="chart-box">{chart_gen.create_asset_composition(metrics)}</div>'
    
    # Scenario summary
    scenario_summary = ""
    if scenarios:
        scenario_summary = "<div class='scenario-summary'><h3>Strategic Scenarios</h3><div class='scenario-grid'>"
        for key, s in list(scenarios.items())[:4]:
            if key != 'expected':
                fm = s.get('financial_metrics', {})
                scenario_summary += f"""
                <div class="scenario-card">
                    <h4>{s.get('name', key)}</h4>
                    <p class="probability">Prob: {s.get('probability', 0)*100:.0f}%</p>
                    <p>Combined: {fm.get('combined_ratio', 0):.1f}%</p>
                    <p>ROE: {fm.get('roe', 0):.1f}%</p>
                </div>
                """
        scenario_summary += "</div></div>"
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>{company} - CFO Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #212121;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 6px solid #1a237e;
        }}
        .header h1 {{ margin: 0 0 10px 0; color: #1a237e; }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            background: #1a237e;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-right: 10px;
        }}
        .badge.secondary {{ background: #5c6bc0; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-top: 4px solid #1a237e;
        }}
        .kpi-card .kpi-label {{ font-size: 0.9em; color: #666; margin-bottom: 8px; }}
        .kpi-card .kpi-value {{ font-size: 2.2em; font-weight: bold; margin-bottom: 5px; }}
        .kpi-card .kpi-subtitle {{ font-size: 0.85em; color: #666; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .chart-box {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .scenario-summary {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
        }}
        .scenario-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .scenario-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 3px solid #1a237e;
        }}
        .scenario-card h4 {{ margin: 0 0 8px 0; color: #1a237e; }}
        .scenario-card .probability {{ color: #666; font-size: 0.9em; margin-bottom: 5px; }}
        
        .footer {{
            margin-top: 30px;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{company}</h1>
        <span class="badge">Analysis Year: {latest_year}</span>
        <span class="badge secondary">Currency: {currency} ({currency_symbol})</span>
        <span class="badge secondary">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
    
    <div class="kpi-grid">
        {''.join(kpi_cards)}
    </div>
    
    <div class="charts-grid">
        {charts_html}
    </div>
    
    {scenario_summary}
    
    <div class="footer">
        <p>CFO Insurance Risk Intelligence v3.0 | Direct LLM Integration | Dashboard ID: {dashboard_id}</p>
        <p>Ask the AI assistant for detailed analysis</p>
    </div>
</body>
</html>'''

@app.route('/dashboard/<dashboard_id>')
def view_dashboard(dashboard_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT html FROM dashboards WHERE id=?", (dashboard_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return result[0]
        return "Dashboard not found", 404
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    try:
        data = request.json
        query = data.get('query', '')
        dashboard_id = data.get('dashboard_id')
        
        # Get data from session
        metrics = session.get('current_dashboard_data', {}).get('metrics', {})
        macro = session.get('macroeconomic_data', {})
        scenarios = session.get('scenario_analysis', {})
        
        # Update assistant with latest data
        cfo_assistant.update_data(metrics, macro, scenarios)
        
        # Get response from LLM
        response = cfo_assistant.get_response(query)
        
        # Store conversation (optional)
        if dashboard_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('''INSERT INTO conversations (id, dashboard_id, timestamp, user_message, agent_response)
                             VALUES (?, ?, ?, ?, ?)''',
                          (generate_id(), dashboard_id, datetime.now().isoformat(), query, response))
                conn.commit()
                conn.close()
            except:
                pass
        
        return jsonify({'success': True, 'response': response})
        
    except Exception as e:
        print(f"Agent error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("="*70)
    print("🏢 INSURANCE RISK INTELLIGENCE v3.0 - DIRECT LLM EDITION")
    print("="*70)
    print("Key Features:")
    print("• Dynamic year detection - always uses latest year")
    print("• CFO-focused metrics: claims, loss ratio, combined ratio, assets")
    print("• Direct OpenAI integration (NO LangChain)")
    print("• Strategic Scenario Analysis (SAM)")
    print("• Dynamic macroeconomic model tailored to your data")
    print("• Clean dashboard - no metrics listing")
    print("="*70)
    print("🌐 http://localhost:5014")
    print("="*70)
    print("⚠️  Make sure to set OPENAI_API_KEY environment variable")
    print("="*70)
    
    socketio.run(app, debug=True, port=5014, host='localhost')
