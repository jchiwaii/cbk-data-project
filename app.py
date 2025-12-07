"""
Kenya CBK Economic Data Dashboard
Interactive dashboard for analyzing Kenya's macroeconomic indicators
"""
import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_gdp_data, load_inflation_data, load_public_debt_data,
    load_domestic_debt_data, load_revenue_expenditure_data,
    get_latest_metrics, get_annual_fiscal_summary, calculate_debt_to_gdp
)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)
app.title = "Kenya Economic Dashboard | CBK Data"

# Color palette
COLORS = {
    'primary': '#00A86B',      # CBK Green
    'secondary': '#FFD700',    # Gold accent
    'tertiary': '#1E90FF',     # Blue
    'quaternary': '#FF6B6B',   # Coral red
    'background': '#1a1a2e',
    'card': '#16213e',
    'text': '#e8e8e8',
    'muted': '#8892b0'
}

CHART_COLORS = ['#00A86B', '#FFD700', '#1E90FF', '#FF6B6B', '#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C']

# Chart template
CHART_TEMPLATE = 'plotly_dark'


def format_currency(value, prefix='KSh ', suffix='M'):
    """Format large numbers as currency"""
    if pd.isna(value) or value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"{prefix}{value/1e6:,.0f}{suffix}"
    elif abs(value) >= 1e6:
        return f"{prefix}{value/1e6:,.1f}{suffix}"
    else:
        return f"{prefix}{value:,.0f}"


def format_trillion(value):
    """Format as trillions"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"KSh {value/1e6:,.2f}T"


def create_kpi_card(title, value, subtitle="", icon="", color=COLORS['primary']):
    """Create a KPI card component"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x", style={'color': color}),
            ], className="mb-2"),
            html.H6(title, className="text-muted mb-1", style={'fontSize': '0.85rem'}),
            html.H3(value, className="mb-1", style={'color': color, 'fontWeight': '700'}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '0.75rem'})
        ], className="text-center py-3")
    ], className="h-100 border-0", style={'backgroundColor': COLORS['card']})


# ============== OVERVIEW TAB ==============
def create_overview_tab():
    """Create the Overview tab content"""
    try:
        metrics = get_latest_metrics()
        gdp_data = load_gdp_data()
        debt_data = load_public_debt_data()
        inflation_data = load_inflation_data()
    except Exception as e:
        return html.Div(f"Error loading data: {str(e)}")
    
    # KPI Cards
    kpi_row = dbc.Row([
        dbc.Col([
            create_kpi_card(
                f"GDP ({metrics['gdp']['year']})",
                format_trillion(metrics['gdp']['nominal']),
                f"Growth: {metrics['gdp']['growth']:.1f}%",
                "fa-chart-line",
                COLORS['primary']
            )
        ], md=3, sm=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "Total Public Debt",
                format_trillion(metrics['debt']['total']),
                f"As of {metrics['debt']['date'].strftime('%b %Y')}",
                "fa-landmark",
                COLORS['tertiary']
            )
        ], md=3, sm=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "Inflation Rate",
                f"{metrics['inflation']['rate']:.1f}%",
                f"12-Month ({metrics['inflation']['date'].strftime('%b %Y')})",
                "fa-percentage",
                COLORS['secondary']
            )
        ], md=3, sm=6, className="mb-3"),
        dbc.Col([
            create_kpi_card(
                "Budget Balance",
                format_currency(metrics['fiscal']['balance']),
                "Latest Fiscal Year",
                "fa-balance-scale",
                COLORS['quaternary'] if metrics['fiscal']['balance'] < 0 else COLORS['primary']
            )
        ], md=3, sm=6, className="mb-3"),
    ], className="mb-4")
    
    # Mini charts
    # GDP Trend mini chart
    gdp_fig = go.Figure()
    gdp_fig.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['Nominal_GDP'],
        mode='lines+markers', name='Nominal GDP',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4)
    ))
    gdp_fig.update_layout(
        template=CHART_TEMPLATE,
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='GDP Trend', font=dict(size=14)),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    # Debt composition mini chart
    latest_debt = debt_data.iloc[-1]
    debt_pie = go.Figure(data=[go.Pie(
        labels=['Domestic', 'External'],
        values=[latest_debt['Domestic_Debt'], latest_debt['External_Debt']],
        hole=0.6,
        marker_colors=[COLORS['primary'], COLORS['tertiary']]
    )])
    debt_pie.update_layout(
        template=CHART_TEMPLATE,
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=f'Debt Composition ({latest_debt["Date"].strftime("%b %Y")})', font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Inflation trend mini chart - sort by date descending and take last 36 months
    inflation_sorted = inflation_data.sort_values('Date', ascending=False)
    inflation_recent = inflation_sorted.head(36).sort_values('Date')  # Last 3 years, then re-sort for chart
    inflation_fig = go.Figure()
    inflation_fig.add_trace(go.Scatter(
        x=inflation_recent['Date'], y=inflation_recent['12_Month_Inflation'],
        mode='lines', name='Inflation',
        line=dict(color=COLORS['secondary'], width=2),
        fill='tozeroy', fillcolor=f"rgba(255, 215, 0, 0.1)"
    ))
    # Add target band
    inflation_fig.add_hline(y=5, line_dash="dash", line_color=COLORS['primary'], 
                           annotation_text="Target", annotation_position="right")
    inflation_fig.update_layout(
        template=CHART_TEMPLATE,
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='Inflation Trend (3Y)', font=dict(size=14)),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='%')
    )
    
    # GDP Growth bar chart
    growth_fig = go.Figure()
    colors = [COLORS['primary'] if x >= 0 else COLORS['quaternary'] for x in gdp_data['GDP_Growth']]
    growth_fig.add_trace(go.Bar(
        x=gdp_data['Year'], y=gdp_data['GDP_Growth'],
        marker_color=colors
    ))
    growth_fig.update_layout(
        template=CHART_TEMPLATE,
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='GDP Growth Rate (%)', font=dict(size=14)),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    charts_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=gdp_fig, config={'displayModeBar': False})])
            ], className="border-0", style={'backgroundColor': COLORS['card']})
        ], md=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=debt_pie, config={'displayModeBar': False})])
            ], className="border-0", style={'backgroundColor': COLORS['card']})
        ], md=6, className="mb-3"),
    ])
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=inflation_fig, config={'displayModeBar': False})])
            ], className="border-0", style={'backgroundColor': COLORS['card']})
        ], md=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=growth_fig, config={'displayModeBar': False})])
            ], className="border-0", style={'backgroundColor': COLORS['card']})
        ], md=6, className="mb-3"),
    ])
    
    return html.Div([
        html.H4("Economic Overview", className="mb-4", style={'color': COLORS['text']}),
        kpi_row,
        charts_row,
        charts_row2
    ])


# ============== GDP TAB ==============
def create_gdp_tab():
    """Create the GDP Analysis tab content"""
    gdp_data = load_gdp_data()
    
    # Main GDP chart - Nominal vs Real
    gdp_comparison = go.Figure()
    gdp_comparison.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['Nominal_GDP'],
        mode='lines+markers', name='Nominal GDP',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8)
    ))
    gdp_comparison.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['Real_GDP'],
        mode='lines+markers', name='Real GDP',
        line=dict(color=COLORS['tertiary'], width=3),
        marker=dict(size=8)
    ))
    gdp_comparison.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Nominal vs Real GDP (KSh Million)', font=dict(size=16)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million'),
        hovermode='x unified'
    )
    
    # GDP Growth Rate chart
    growth_colors = [COLORS['primary'] if x >= 0 else COLORS['quaternary'] for x in gdp_data['GDP_Growth']]
    growth_chart = go.Figure()
    growth_chart.add_trace(go.Bar(
        x=gdp_data['Year'], y=gdp_data['GDP_Growth'],
        marker_color=growth_colors,
        text=[f"{x:.1f}%" for x in gdp_data['GDP_Growth']],
        textposition='outside'
    ))
    growth_chart.add_hline(y=0, line_color='white', line_width=1)
    growth_chart.add_hline(y=gdp_data['GDP_Growth'].mean(), line_dash="dash", 
                          line_color=COLORS['secondary'],
                          annotation_text=f"Avg: {gdp_data['GDP_Growth'].mean():.1f}%",
                          annotation_position="right")
    growth_chart.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Annual GDP Growth Rate (%)', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Growth %')
    )
    
    # YoY Change calculation
    gdp_data['YoY_Change'] = gdp_data['Nominal_GDP'].pct_change() * 100
    yoy_chart = go.Figure()
    yoy_chart.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['YoY_Change'],
        mode='lines+markers', name='YoY Change',
        line=dict(color=COLORS['secondary'], width=2),
        fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.2)'
    ))
    yoy_chart.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Year-over-Year Nominal GDP Change (%)', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='% Change')
    )
    
    return html.Div([
        html.H4("GDP Analysis", className="mb-4", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=gdp_comparison, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=growth_chart, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=yoy_chart, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=6),
        ])
    ])


# ============== DEBT TAB ==============
def create_debt_tab():
    """Create the Debt Analysis tab content"""
    debt_data = load_public_debt_data()
    dom_debt = load_domestic_debt_data()
    debt_gdp = calculate_debt_to_gdp()
    
    # Stacked area - Domestic vs External
    debt_area = go.Figure()
    debt_area.add_trace(go.Scatter(
        x=debt_data['Date'], y=debt_data['Domestic_Debt'],
        mode='lines', name='Domestic Debt',
        stackgroup='one', fillcolor=f"rgba(0, 168, 107, 0.7)",
        line=dict(color=COLORS['primary'], width=0.5)
    ))
    debt_area.add_trace(go.Scatter(
        x=debt_data['Date'], y=debt_data['External_Debt'],
        mode='lines', name='External Debt',
        stackgroup='one', fillcolor=f"rgba(30, 144, 255, 0.7)",
        line=dict(color=COLORS['tertiary'], width=0.5)
    ))
    debt_area.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Public Debt Composition Over Time (KSh Million)', font=dict(size=16)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million'),
        hovermode='x unified'
    )
    
    # Latest debt composition by instrument
    latest_dom = dom_debt.iloc[-1]
    instruments = ['Treasury_Bills', 'Treasury_Bonds', 'Overdraft_CBK', 'Advances_Commercial', 'Other_Debt']
    labels = ['Treasury Bills', 'Treasury Bonds', 'CBK Overdraft', 'Commercial Advances', 'Other']
    values = [latest_dom[col] for col in instruments]
    
    # Filter out NaN and zero values
    filtered = [(l, v) for l, v in zip(labels, values) if pd.notna(v) and v > 0]
    if filtered:
        labels, values = zip(*filtered)
    
    instrument_pie = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=CHART_COLORS[:len(labels)]
    )])
    instrument_pie.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f'Domestic Debt by Instrument ({latest_dom["Date"].strftime("%b %Y")})', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05)
    )
    
    # Debt to GDP ratio
    debt_gdp_chart = go.Figure()
    debt_gdp_chart.add_trace(go.Scatter(
        x=debt_gdp['Year'], y=debt_gdp['Debt_to_GDP'],
        mode='lines+markers', name='Debt/GDP',
        line=dict(color=COLORS['quaternary'], width=3),
        marker=dict(size=8),
        fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    debt_gdp_chart.add_hline(y=50, line_dash="dash", line_color=COLORS['secondary'],
                            annotation_text="50% threshold", annotation_position="right")
    debt_gdp_chart.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Debt-to-GDP Ratio (%)', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='%')
    )
    
    # Total debt trend
    total_debt_chart = go.Figure()
    total_debt_chart.add_trace(go.Scatter(
        x=debt_data['Date'], y=debt_data['Total_Debt'],
        mode='lines', name='Total Debt',
        line=dict(color=COLORS['secondary'], width=2)
    ))
    total_debt_chart.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Total Public Debt Trend', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million')
    )
    
    return html.Div([
        html.H4("Debt Analysis", className="mb-4", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=debt_area, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=instrument_pie, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=debt_gdp_chart, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=total_debt_chart, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ])
    ])


# ============== INFLATION TAB ==============
def create_inflation_tab():
    """Create the Inflation tab content"""
    inflation_data = load_inflation_data()
    
    # Main inflation trend
    inflation_trend = go.Figure()
    inflation_trend.add_trace(go.Scatter(
        x=inflation_data['Date'], y=inflation_data['12_Month_Inflation'],
        mode='lines', name='12-Month Inflation',
        line=dict(color=COLORS['secondary'], width=2)
    ))
    inflation_trend.add_trace(go.Scatter(
        x=inflation_data['Date'], y=inflation_data['Annual_Avg_Inflation'],
        mode='lines', name='Annual Average',
        line=dict(color=COLORS['tertiary'], width=2, dash='dot')
    ))
    # Target band
    inflation_trend.add_hrect(y0=2.5, y1=7.5, line_width=0, 
                             fillcolor="rgba(0, 168, 107, 0.1)",
                             annotation_text="CBK Target (2.5-7.5%)", 
                             annotation_position="top right")
    inflation_trend.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Inflation Rate Over Time', font=dict(size=16)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title=''),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Inflation Rate (%)'),
        hovermode='x unified'
    )
    
    # Inflation heatmap by month/year
    inflation_pivot = inflation_data.pivot_table(
        index='Month', columns='Year', values='12_Month_Inflation', aggfunc='mean'
    )
    
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
    inflation_pivot = inflation_pivot.reindex(months_order)
    
    heatmap = go.Figure(data=go.Heatmap(
        z=inflation_pivot.values,
        x=inflation_pivot.columns,
        y=inflation_pivot.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title='%')
    ))
    heatmap.update_layout(
        template=CHART_TEMPLATE,
        height=450,
        margin=dict(l=100, r=40, t=60, b=60),
        title=dict(text='Monthly Inflation Heatmap by Year', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Month', autorange='reversed')
    )
    
    # Recent inflation stats - get last 12 months by date
    inflation_sorted = inflation_data.sort_values('Date', ascending=False)
    recent_inflation = inflation_sorted.head(12).sort_values('Date')
    recent_bar = go.Figure()
    recent_bar.add_trace(go.Bar(
        x=recent_inflation['Date'], y=recent_inflation['12_Month_Inflation'],
        marker_color=[COLORS['primary'] if v <= 7.5 else COLORS['quaternary'] 
                     for v in recent_inflation['12_Month_Inflation']],
        text=[f"{v:.1f}%" for v in recent_inflation['12_Month_Inflation']],
        textposition='outside'
    ))
    recent_bar.add_hline(y=7.5, line_dash="dash", line_color=COLORS['quaternary'],
                        annotation_text="Upper Target (7.5%)", annotation_position="right")
    recent_bar.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Last 12 Months Inflation', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='%')
    )
    
    return html.Div([
        html.H4("Inflation Analysis", className="mb-4", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=inflation_trend, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=heatmap, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=recent_bar, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=5),
        ])
    ])


# ============== FISCAL TAB ==============
def create_fiscal_tab():
    """Create the Fiscal Analysis tab content"""
    fiscal_summary = get_annual_fiscal_summary()
    fiscal_data = load_revenue_expenditure_data()
    
    # Revenue vs Expenditure grouped bar
    rev_exp = go.Figure()
    rev_exp.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'], 
        y=fiscal_summary['Total_Revenue'],
        name='Revenue',
        marker_color=COLORS['primary']
    ))
    rev_exp.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'], 
        y=fiscal_summary['Total_Expenditure'],
        name='Expenditure',
        marker_color=COLORS['quaternary']
    ))
    rev_exp.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Annual Revenue vs Expenditure (Fiscal Year End - June)', font=dict(size=16)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Fiscal Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million'),
        barmode='group'
    )
    
    # Revenue breakdown stacked bar
    revenue_breakdown = go.Figure()
    for i, (col, label) in enumerate([
        ('Import_Duty', 'Import Duty'),
        ('Excise_Duty', 'Excise Duty'),
        ('Income_Tax', 'Income Tax'),
        ('VAT', 'VAT'),
        ('Non_Tax_Revenue', 'Non-Tax Revenue')
    ]):
        revenue_breakdown.add_trace(go.Bar(
            x=fiscal_summary['Fiscal_Year_End'],
            y=fiscal_summary[col],
            name=label,
            marker_color=CHART_COLORS[i]
        ))
    revenue_breakdown.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Revenue Breakdown by Source', font=dict(size=16)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Fiscal Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million'),
        barmode='stack'
    )
    
    # Budget deficit/surplus trend
    fiscal_summary['Budget_Balance'] = fiscal_summary['Total_Revenue'] - fiscal_summary['Total_Expenditure']
    balance_colors = [COLORS['primary'] if x >= 0 else COLORS['quaternary'] for x in fiscal_summary['Budget_Balance']]
    
    balance_chart = go.Figure()
    balance_chart.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'],
        y=fiscal_summary['Budget_Balance'],
        marker_color=balance_colors,
        text=[f"{x/1e6:.0f}T" for x in fiscal_summary['Budget_Balance']],
        textposition='outside'
    ))
    balance_chart.add_hline(y=0, line_color='white', line_width=1)
    balance_chart.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        title=dict(text='Budget Balance (Surplus/Deficit)', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Fiscal Year'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='KSh Million')
    )
    
    # Tax revenue composition pie
    latest_fiscal = fiscal_summary.iloc[-1]
    tax_components = ['Import_Duty', 'Excise_Duty', 'Income_Tax', 'VAT']
    tax_labels = ['Import Duty', 'Excise Duty', 'Income Tax', 'VAT']
    tax_values = [latest_fiscal[col] for col in tax_components if pd.notna(latest_fiscal[col])]
    tax_labels = [l for l, col in zip(tax_labels, tax_components) if pd.notna(latest_fiscal[col])]
    
    tax_pie = go.Figure(data=[go.Pie(
        labels=tax_labels, values=tax_values,
        hole=0.5,
        marker_colors=CHART_COLORS[:len(tax_labels)]
    )])
    tax_pie.update_layout(
        template=CHART_TEMPLATE,
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f'Tax Revenue Composition (FY{int(latest_fiscal["Fiscal_Year_End"])})', font=dict(size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return html.Div([
        html.H4("Fiscal Analysis", className="mb-4", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=rev_exp, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=revenue_breakdown, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=balance_chart, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=tax_pie, config={'displayModeBar': True})])
                ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
            ], md=5),
        ])
    ])


# ============== CORRELATIONS TAB ==============
def create_correlations_tab():
    """Create the Correlations tab content"""
    try:
        gdp_data = load_gdp_data()
        inflation_data = load_inflation_data()
        debt_data = load_public_debt_data()
        
        # Get annual averages for inflation
        inflation_annual = inflation_data.groupby('Year')['12_Month_Inflation'].mean().reset_index()
        inflation_annual.columns = ['Year', 'Avg_Inflation']
        
        # Merge datasets
        merged = gdp_data.merge(inflation_annual, on='Year', how='inner')
        
        # Get year-end debt
        debt_yearly = debt_data.copy()
        debt_yearly['Year'] = debt_yearly['Date'].dt.year
        yearly_debt = debt_yearly.groupby('Year')['Total_Debt'].last().reset_index()
        merged = merged.merge(yearly_debt, on='Year', how='inner')
        
        # Handle negative GDP_Growth for size parameter (must be positive)
        merged['GDP_Growth_Size'] = merged['GDP_Growth'].abs() + 1
        
        # GDP vs Inflation scatter (without trendline to avoid statsmodels issues)
        gdp_inflation = go.Figure()
        gdp_inflation.add_trace(go.Scatter(
            x=merged['Nominal_GDP'], 
            y=merged['Avg_Inflation'],
            mode='markers',
            marker=dict(
                size=merged['GDP_Growth_Size'] * 3,
                color=merged['Year'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title='Year')
            ),
            text=merged.apply(lambda r: f"Year: {r['Year']}<br>GDP: {r['Nominal_GDP']:,.0f}<br>Inflation: {r['Avg_Inflation']:.1f}%<br>Growth: {r['GDP_Growth']:.1f}%", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        # Add trendline manually using numpy
        z = np.polyfit(merged['Nominal_GDP'], merged['Avg_Inflation'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['Nominal_GDP'].min(), merged['Nominal_GDP'].max(), 100)
        gdp_inflation.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.5)', dash='dash'),
            name='Trend',
            showlegend=False
        ))
        gdp_inflation.update_layout(
            template=CHART_TEMPLATE,
            height=400,
            margin=dict(l=60, r=40, t=60, b=60),
            title=dict(text='GDP vs Inflation (sized by growth rate)', font=dict(size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Nominal GDP (KSh M)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Avg Inflation (%)')
        )
        
        # Debt vs GDP scatter
        debt_gdp_scatter = go.Figure()
        debt_gdp_scatter.add_trace(go.Scatter(
            x=merged['Nominal_GDP'], 
            y=merged['Total_Debt'],
            mode='markers',
            marker=dict(color=COLORS['tertiary'], size=12),
            text=merged.apply(lambda r: f"Year: {r['Year']}<br>GDP: {r['Nominal_GDP']:,.0f}<br>Debt: {r['Total_Debt']:,.0f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        # Add trendline
        z2 = np.polyfit(merged['Nominal_GDP'], merged['Total_Debt'], 1)
        p2 = np.poly1d(z2)
        debt_gdp_scatter.add_trace(go.Scatter(
            x=x_line, y=p2(x_line),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.5)', dash='dash'),
            name='Trend',
            showlegend=False
        ))
        debt_gdp_scatter.update_layout(
            template=CHART_TEMPLATE,
            height=400,
            margin=dict(l=60, r=40, t=60, b=60),
            title=dict(text='GDP vs Total Public Debt', font=dict(size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Nominal GDP (KSh M)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Total Debt (KSh M)')
        )
    
        # Correlation matrix
        corr_data = merged[['Nominal_GDP', 'Real_GDP', 'GDP_Growth', 'Avg_Inflation', 'Total_Debt']].corr()
        
        corr_heatmap = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=['Nominal GDP', 'Real GDP', 'GDP Growth', 'Inflation', 'Total Debt'],
            y=['Nominal GDP', 'Real GDP', 'GDP Growth', 'Inflation', 'Total Debt'],
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=12),
            colorbar=dict(title='Correlation')
        ))
        corr_heatmap.update_layout(
            template=CHART_TEMPLATE,
            height=450,
            margin=dict(l=100, r=40, t=60, b=100),
            title=dict(text='Correlation Matrix', font=dict(size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Growth vs Inflation over time
        dual_axis = make_subplots(specs=[[{"secondary_y": True}]])
        dual_axis.add_trace(
            go.Scatter(x=merged['Year'], y=merged['GDP_Growth'], 
                      name='GDP Growth', line=dict(color=COLORS['primary'], width=2)),
            secondary_y=False
        )
        dual_axis.add_trace(
            go.Scatter(x=merged['Year'], y=merged['Avg_Inflation'],
                      name='Inflation', line=dict(color=COLORS['quaternary'], width=2)),
            secondary_y=True
        )
        dual_axis.update_layout(
            template=CHART_TEMPLATE,
            height=350,
            margin=dict(l=60, r=60, t=60, b=60),
            title=dict(text='GDP Growth vs Inflation Over Time', font=dict(size=16)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        dual_axis.update_xaxes(showgrid=False)
        dual_axis.update_yaxes(title_text="GDP Growth (%)", secondary_y=False, 
                              showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        dual_axis.update_yaxes(title_text="Inflation (%)", secondary_y=True)
        
        return html.Div([
            html.H4("Correlations & Relationships", className="mb-4", style={'color': COLORS['text']}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=gdp_inflation, config={'displayModeBar': True})])
                    ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=debt_gdp_scatter, config={'displayModeBar': True})])
                    ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
                ], md=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=corr_heatmap, config={'displayModeBar': True})])
                    ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=dual_axis, config={'displayModeBar': True})])
                    ], className="border-0 mb-3", style={'backgroundColor': COLORS['card']})
                ], md=6),
            ])
        ])
    except Exception as e:
        return html.Div([
            html.H4("Correlations & Relationships", className="mb-4", style={'color': COLORS['text']}),
            dbc.Alert(f"Error loading correlation data: {str(e)}", color="danger")
        ])


# ============== MAIN LAYOUT ==============
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H2("Kenya Economic Dashboard", 
                       className="mb-0", 
                       style={'color': COLORS['primary'], 'fontWeight': '700'}),
                html.P("Central Bank of Kenya Data Analysis", 
                      className="text-muted mb-0",
                      style={'fontSize': '0.9rem'})
            ], className="py-3")
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("Data Source: ", className="text-muted"),
                html.Span("CBK", style={'color': COLORS['primary'], 'fontWeight': '600'})
            ], className="text-end py-3")
        ], md=4)
    ], className="border-bottom mb-4", style={'borderColor': f'{COLORS["primary"]}40 !important'}),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Overview", tab_id="overview", 
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
        dbc.Tab(label="GDP Analysis", tab_id="gdp",
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
        dbc.Tab(label="Debt Analysis", tab_id="debt",
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
        dbc.Tab(label="Inflation", tab_id="inflation",
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
        dbc.Tab(label="Fiscal", tab_id="fiscal",
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
        dbc.Tab(label="Correlations", tab_id="correlations",
               label_style={'color': COLORS['muted']},
               active_label_style={'color': COLORS['primary'], 'fontWeight': '600'}),
    ], id="tabs", active_tab="overview", className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': f'{COLORS["primary"]}40'}),
            html.P([
                "Built with ",
                html.Span("Dash", style={'color': COLORS['primary']}),
                " & ",
                html.Span("Plotly", style={'color': COLORS['tertiary']}),
                " | Data: Central Bank of Kenya"
            ], className="text-center text-muted", style={'fontSize': '0.8rem'})
        ])
    ], className="mt-4")
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})


@callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab"""
    if active_tab == "overview":
        return create_overview_tab()
    elif active_tab == "gdp":
        return create_gdp_tab()
    elif active_tab == "debt":
        return create_debt_tab()
    elif active_tab == "inflation":
        return create_inflation_tab()
    elif active_tab == "fiscal":
        return create_fiscal_tab()
    elif active_tab == "correlations":
        return create_correlations_tab()
    return html.Div("Select a tab")


if __name__ == '__main__':
    app.run(debug=True, port=8050)

