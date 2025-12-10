"""
Kenya CBK Economic Data Dashboard
Interactive dashboard for analyzing Kenya's macroeconomic indicators
"""
import os
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

# Initialize the Dash app with Bootstrap theme and FontAwesome
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)
app.title = "Kenya Economic Dashboard | CBK Data"

# Color palette - Matching CSS variables
COLORS = {
    'primary': '#00F2EA',      # Cyan
    'secondary': '#FF0055',    # Neon Pink
    'tertiary': '#FFE600',     # Electric Yellow
    'quaternary': '#7000FF',   # Deep Violet
    'background': '#05050A',   # Deep dark
    'card': 'rgba(18, 18, 28, 0.6)', # Glass
    'text': '#FFFFFF',
    'muted': '#A0A0B0',
    'success': '#00F2EA',
    'danger': '#FF0055'
}

CHART_COLORS = ['#00F2EA', '#FF0055', '#FFE600', '#7000FF', '#00B8FF', '#FF9F00', '#00FF9D', '#D900FF']

# Chart template configuration
CHART_FONT = "JetBrains Mono"
TITLE_FONT = "Space Grotesk"

def get_chart_layout(height=400):
    """Get common chart layout configuration"""
    return dict(
        template='plotly_dark',
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=CHART_FONT, size=12, color=COLORS['muted']),
        title=dict(font=dict(family=TITLE_FONT, size=18, color=COLORS['text'])),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            showgrid=False, 
            zeroline=False,
            title_font=dict(family=TITLE_FONT),
            tickfont=dict(family=CHART_FONT)
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.08)', 
            zeroline=False,
            title_font=dict(family=TITLE_FONT),
            tickfont=dict(family=CHART_FONT)
        ),
        hoverlabel=dict(
            bgcolor=COLORS['background'],
            bordercolor=COLORS['primary'],
            font=dict(family=CHART_FONT)
        )
    )

CHART_TEMPLATE = 'plotly_dark'

# High-resolution PNG download configuration
CHART_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'cbk_chart',
        'height': 1080,
        'width': 1920,
        'scale': 3  # 3x resolution for high quality
    },
    'modeBarButtonsToAdd': ['downloadImage'],
}

# Mini chart config (smaller default size)
MINI_CHART_CONFIG = {
    'displayModeBar': True,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'cbk_chart',
        'height': 720,
        'width': 1280,
        'scale': 3
    },
}


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
            ], className="kpi-card-icon mb-3"),
            html.H6(title, className="text-muted mb-2", style={'fontSize': '0.85rem', 'fontFamily': 'Space Grotesk'}),
            html.H3(value, className="mb-2 kpi-value", style={'color': color, 'fontWeight': '700'}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '0.75rem', 'fontFamily': 'Inter'})
        ], className="text-center py-4")
    ], className="h-100 border-0 glass-card")


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
    # GDP Trend mini chart (convert to trillions)
    gdp_fig = go.Figure()
    gdp_fig.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['Nominal_GDP'] / 1e6,
        mode='lines+markers', name='Nominal GDP',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=6, color=COLORS['background'], line=dict(color=COLORS['primary'], width=2)),
        hovertemplate='Year: %{x}<br>GDP: %{y:.2f}T<extra></extra>'
    ))
    
    layout = get_chart_layout(250)
    layout.update(
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='GDP Trend (KSh T)', font=dict(size=14)),
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title=None)
    )
    gdp_fig.update_layout(**layout)
    
    # Debt composition mini chart
    latest_debt = debt_data.iloc[-1]
    debt_pie = go.Figure(data=[go.Pie(
        labels=['Domestic', 'External'],
        values=[latest_debt['Domestic_Debt'], latest_debt['External_Debt']],
        hole=0.7,
        marker_colors=[COLORS['primary'], COLORS['tertiary']],
        textfont=dict(family=CHART_FONT)
    )])
    
    pie_layout = get_chart_layout(250)
    pie_layout.update(
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text=f'Debt Composition', font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        xaxis=None, yaxis=None
    )
    debt_pie.update_layout(**pie_layout)
    
    # Inflation trend mini chart - sort by date descending and take last 36 months
    inflation_sorted = inflation_data.sort_values('Date', ascending=False)
    inflation_recent = inflation_sorted.head(36).sort_values('Date')  # Last 3 years, then re-sort for chart
    inflation_fig = go.Figure()
    inflation_fig.add_trace(go.Scatter(
        x=inflation_recent['Date'], y=inflation_recent['12_Month_Inflation'],
        mode='lines', name='Inflation',
        line=dict(color=COLORS['secondary'], width=3),
        fill='tozeroy', fillcolor=f"rgba(255, 0, 85, 0.1)"
    ))
    # Add target band
    inflation_fig.add_hline(y=5, line_dash="dash", line_color=COLORS['success'], 
                           annotation_text="Target", annotation_position="right",
                           annotation_font=dict(color=COLORS['success'], family=CHART_FONT))
    
    inf_layout = get_chart_layout(250)
    inf_layout.update(
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='Inflation Trend (3Y)', font=dict(size=14)),
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.08)', title='%')
    )
    inflation_fig.update_layout(**inf_layout)
    
    # GDP Growth bar chart
    growth_fig = go.Figure()
    colors = [COLORS['primary'] if x >= 0 else COLORS['danger'] for x in gdp_data['GDP_Growth']]
    growth_fig.add_trace(go.Bar(
        x=gdp_data['Year'], y=gdp_data['GDP_Growth'],
        marker_color=colors,
        marker_line_width=0
    ))
    
    growth_layout = get_chart_layout(250)
    growth_layout.update(
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text='GDP Growth Rate (%)', font=dict(size=14)),
        showlegend=False
    )
    growth_fig.update_layout(**growth_layout)
    
    charts_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=gdp_fig, config=MINI_CHART_CONFIG)])
            ], className="border-0 glass-card")
        ], md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=debt_pie, config=MINI_CHART_CONFIG)])
            ], className="border-0 glass-card")
        ], md=6, className="mb-4"),
    ])
    
    charts_row2 = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=inflation_fig, config=MINI_CHART_CONFIG)])
            ], className="border-0 glass-card")
        ], md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=growth_fig, config=MINI_CHART_CONFIG)])
            ], className="border-0 glass-card")
        ], md=6, className="mb-4"),
    ])
    
    return html.Div([
        html.H4("Economic Overview", className="mb-4 gradient-text", style={'color': COLORS['text']}),
        kpi_row,
        charts_row,
        charts_row2
    ])


# ============== GDP TAB ==============
def create_gdp_tab():
    """Create the GDP Analysis tab content"""
    gdp_data = load_gdp_data()
    
    # Convert from millions to trillions for display
    gdp_nominal_t = gdp_data['Nominal_GDP'] / 1e6
    gdp_real_t = gdp_data['Real_GDP'] / 1e6
    
    # Main GDP chart - Nominal vs Real
    gdp_comparison = go.Figure()
    gdp_comparison.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_nominal_t,
        mode='lines+markers', name='Nominal GDP',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8, color=COLORS['background'], line=dict(color=COLORS['primary'], width=2)),
        hovertemplate='%{y:.2f}T<extra>Nominal GDP</extra>'
    ))
    gdp_comparison.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_real_t,
        mode='lines+markers', name='Real GDP',
        line=dict(color=COLORS['tertiary'], width=3),
        marker=dict(size=8, color=COLORS['background'], line=dict(color=COLORS['tertiary'], width=2)),
        hovertemplate='%{y:.2f}T<extra>Real GDP</extra>'
    ))
    
    layout = get_chart_layout(400)
    layout.update(
        title=dict(text='Nominal vs Real GDP (KSh Trillion)', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Year',
        yaxis_title='KSh Trillion',
        hovermode='x unified'
    )
    gdp_comparison.update_layout(**layout)
    
    # GDP Growth Rate chart
    growth_colors = [COLORS['primary'] if x >= 0 else COLORS['danger'] for x in gdp_data['GDP_Growth']]
    growth_chart = go.Figure()
    growth_chart.add_trace(go.Bar(
        x=gdp_data['Year'], y=gdp_data['GDP_Growth'],
        marker_color=growth_colors,
        text=[f"{x:.1f}%" for x in gdp_data['GDP_Growth']],
        textposition='outside',
        textfont=dict(family=CHART_FONT)
    ))
    growth_chart.add_hline(y=0, line_color='white', line_width=1)
    growth_chart.add_hline(y=gdp_data['GDP_Growth'].mean(), line_dash="dash", 
                          line_color=COLORS['secondary'],
                          annotation_text=f"Avg: {gdp_data['GDP_Growth'].mean():.1f}%",
                          annotation_position="right",
                          annotation_font=dict(color=COLORS['secondary'], family=CHART_FONT))
    
    growth_layout = get_chart_layout(400)
    growth_layout.update(
        margin=dict(l=40, r=80, t=40, b=40),
        title=dict(text='Annual GDP Growth Rate (%)', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title='Year',
        yaxis_title='Growth %'
    )
    growth_chart.update_layout(**growth_layout)
    
    # YoY Change calculation
    gdp_data['YoY_Change'] = gdp_data['Nominal_GDP'].pct_change() * 100
    yoy_chart = go.Figure()
    yoy_chart.add_trace(go.Scatter(
        x=gdp_data['Year'], y=gdp_data['YoY_Change'],
        mode='lines+markers', name='YoY Change',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(size=6, color=COLORS['background'], line=dict(color=COLORS['secondary'], width=2)),
        fill='tozeroy', fillcolor='rgba(255, 0, 85, 0.1)',
        hovertemplate='Year: %{x}<br>Change: %{y:.1f}%<extra></extra>'
    ))
    
    yoy_layout = get_chart_layout(400)
    yoy_layout.update(
        title=dict(text='Year-over-Year Nominal GDP Change (%)', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title='Year',
        yaxis_title='% Change'
    )
    yoy_chart.update_layout(**yoy_layout)
    
    return html.Div([
        html.H4("GDP Analysis", className="mb-4 gradient-text", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=gdp_comparison, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=growth_chart, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=yoy_chart, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=6),
        ])
    ])


# ============== DEBT TAB ==============
def create_debt_tab():
    """Create the Debt Analysis tab content"""
    debt_data = load_public_debt_data()
    dom_debt = load_domestic_debt_data()
    debt_gdp = calculate_debt_to_gdp()
    
    # Convert from millions to trillions for display
    domestic_t = debt_data['Domestic_Debt'] / 1e6
    external_t = debt_data['External_Debt'] / 1e6
    total_t = debt_data['Total_Debt'] / 1e6
    
    # Stacked area - Domestic vs External
    debt_area = go.Figure()
    debt_area.add_trace(go.Scatter(
        x=debt_data['Date'], y=domestic_t,
        mode='lines', name='Domestic Debt',
        stackgroup='one', fillcolor=f"rgba(0, 242, 234, 0.1)",
        line=dict(color=COLORS['primary'], width=1),
        hovertemplate='%{y:.2f}T<extra>Domestic</extra>'
    ))
    debt_area.add_trace(go.Scatter(
        x=debt_data['Date'], y=external_t,
        mode='lines', name='External Debt',
        stackgroup='one', fillcolor=f"rgba(255, 230, 0, 0.1)",
        line=dict(color=COLORS['tertiary'], width=1),
        hovertemplate='%{y:.2f}T<extra>External</extra>'
    ))
    
    layout = get_chart_layout(400)
    layout.update(
        title=dict(text='Public Debt Composition Over Time (KSh Trillion)', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title=None,
        yaxis_title='KSh Trillion',
        hovermode='x unified'
    )
    debt_area.update_layout(**layout)
    
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
        marker_colors=CHART_COLORS[:len(labels)],
        textfont=dict(family=CHART_FONT)
    )])
    
    pie_layout = get_chart_layout(400)
    pie_layout.update(
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f'Domestic Debt by Instrument ({latest_dom["Date"].strftime("%b %Y")})', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05),
        xaxis=None, yaxis=None
    )
    instrument_pie.update_layout(**pie_layout)
    
    # Debt to GDP ratio
    debt_gdp_chart = go.Figure()
    debt_gdp_chart.add_trace(go.Scatter(
        x=debt_gdp['Year'], y=debt_gdp['Debt_to_GDP'],
        mode='lines+markers', name='Debt/GDP',
        line=dict(color=COLORS['quaternary'], width=3),
        marker=dict(size=8, color=COLORS['background'], line=dict(color=COLORS['quaternary'], width=2)),
        fill='tozeroy', fillcolor='rgba(112, 0, 255, 0.1)',
        hovertemplate='Year: %{x}<br>Debt/GDP: %{y:.1f}%<extra></extra>'
    ))
    debt_gdp_chart.add_hline(y=50, line_dash="dash", line_color=COLORS['secondary'],
                            annotation_text="50% threshold", annotation_position="right",
                            annotation_font=dict(color=COLORS['secondary'], family=CHART_FONT))
    
    dg_layout = get_chart_layout(400)
    dg_layout.update(
        margin=dict(l=60, r=80, t=60, b=60),
        title=dict(text='Debt-to-GDP Ratio (%)', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title='Year',
        yaxis_title='%'
    )
    debt_gdp_chart.update_layout(**dg_layout)
    
    # Total debt trend
    total_debt_chart = go.Figure()
    total_debt_chart.add_trace(go.Scatter(
        x=debt_data['Date'], y=total_t,
        mode='lines', name='Total Debt',
        line=dict(color=COLORS['secondary'], width=3),
        fill='tozeroy', fillcolor='rgba(255, 0, 85, 0.05)',
        hovertemplate='%{x|%b %Y}<br>Total: %{y:.2f}T<extra></extra>'
    ))
    
    td_layout = get_chart_layout(350)
    td_layout.update(
        title=dict(text='Total Public Debt Trend', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title=None,
        yaxis_title='KSh Trillion'
    )
    total_debt_chart.update_layout(**td_layout)
    
    return html.Div([
        html.H4("Debt Analysis", className="mb-4 gradient-text", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=debt_area, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=instrument_pie, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=debt_gdp_chart, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=total_debt_chart, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
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
                             fillcolor="rgba(0, 242, 234, 0.05)",
                             annotation_text="CBK Target (2.5-7.5%)", 
                             annotation_position="top right",
                             annotation_font=dict(color=COLORS['success'], family=CHART_FONT))
    
    it_layout = get_chart_layout(400)
    it_layout.update(
        margin=dict(l=60, r=100, t=60, b=60),
        title=dict(text='Inflation Rate Over Time', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title=None,
        yaxis_title='Inflation Rate (%)',
        hovermode='x unified'
    )
    inflation_trend.update_layout(**it_layout)
    
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
        colorscale='Viridis',
        colorbar=dict(title='%', titlefont=dict(family=CHART_FONT), tickfont=dict(family=CHART_FONT))
    ))
    
    hm_layout = get_chart_layout(450)
    hm_layout.update(
        margin=dict(l=100, r=40, t=60, b=60),
        title=dict(text='Monthly Inflation Heatmap by Year', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title='Year',
        yaxis_title=None,
        yaxis=dict(autorange='reversed')
    )
    heatmap.update_layout(**hm_layout)
    
    # Recent inflation stats - get last 12 months by date
    inflation_sorted = inflation_data.sort_values('Date', ascending=False)
    recent_inflation = inflation_sorted.head(12).sort_values('Date')
    recent_bar = go.Figure()
    recent_bar.add_trace(go.Bar(
        x=recent_inflation['Date'], y=recent_inflation['12_Month_Inflation'],
        marker_color=[COLORS['primary'] if v <= 7.5 else COLORS['danger'] 
                     for v in recent_inflation['12_Month_Inflation']],
        text=[f"{v:.1f}%" for v in recent_inflation['12_Month_Inflation']],
        textposition='outside',
        textfont=dict(family=CHART_FONT)
    ))
    recent_bar.add_hline(y=7.5, line_dash="dash", line_color=COLORS['danger'],
                        annotation_text="Upper Target (7.5%)", annotation_position="right",
                        annotation_font=dict(color=COLORS['danger'], family=CHART_FONT))
    
    rb_layout = get_chart_layout(450)
    rb_layout.update(
        margin=dict(l=60, r=80, t=60, b=60),
        title=dict(text='Last 12 Months Inflation', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title=None,
        xaxis=dict(tickangle=-45),
        yaxis_title='%'
    )
    recent_bar.update_layout(**rb_layout)
    
    return html.Div([
        html.H4("Inflation Analysis", className="mb-4 gradient-text", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=inflation_trend, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=heatmap, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=recent_bar, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=5),
        ])
    ])


# ============== FISCAL TAB ==============
def create_fiscal_tab():
    """Create the Fiscal Analysis tab content"""
    fiscal_summary = get_annual_fiscal_summary()
    fiscal_data = load_revenue_expenditure_data()
    
    # Convert from millions to trillions for display
    revenue_t = fiscal_summary['Total_Revenue'] / 1e6
    expenditure_t = fiscal_summary['Total_Expenditure'] / 1e6
    
    # Revenue vs Expenditure grouped bar
    rev_exp = go.Figure()
    rev_exp.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'], 
        y=revenue_t,
        name='Revenue',
        marker_color=COLORS['primary'],
        hovertemplate='FY %{x}<br>Revenue: %{y:.2f}T<extra></extra>'
    ))
    rev_exp.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'], 
        y=expenditure_t,
        name='Expenditure',
        marker_color=COLORS['danger'],
        hovertemplate='FY %{x}<br>Expenditure: %{y:.2f}T<extra></extra>'
    ))
    
    re_layout = get_chart_layout(400)
    re_layout.update(
        title=dict(text='Annual Revenue vs Expenditure (Fiscal Year End - June)', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Fiscal Year',
        yaxis_title='KSh Trillion',
        barmode='group'
    )
    rev_exp.update_layout(**re_layout)
    
    # Revenue breakdown stacked bar (convert to trillions)
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
            y=fiscal_summary[col] / 1e6,
            name=label,
            marker_color=CHART_COLORS[i],
            hovertemplate=f'{label}: %{{y:.2f}}T<extra></extra>'
        ))
    
    rb_layout = get_chart_layout(400)
    rb_layout.update(
        title=dict(text='Revenue Breakdown by Source', font=dict(size=16, family=TITLE_FONT)),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis_title='Fiscal Year',
        yaxis_title='KSh Trillion',
        barmode='stack'
    )
    revenue_breakdown.update_layout(**rb_layout)
    
    # Budget deficit/surplus trend (convert to trillions)
    fiscal_summary['Budget_Balance'] = fiscal_summary['Total_Revenue'] - fiscal_summary['Total_Expenditure']
    balance_t = fiscal_summary['Budget_Balance'] / 1e6
    balance_colors = [COLORS['primary'] if x >= 0 else COLORS['danger'] for x in balance_t]
    
    balance_chart = go.Figure()
    balance_chart.add_trace(go.Bar(
        x=fiscal_summary['Fiscal_Year_End'],
        y=balance_t,
        marker_color=balance_colors,
        text=[f"{x:.2f}T" for x in balance_t],
        textposition='outside',
        textfont=dict(family=CHART_FONT)
    ))
    balance_chart.add_hline(y=0, line_color='white', line_width=1)
    
    bc_layout = get_chart_layout(500)
    bc_layout.update(
        title=dict(text='Budget Balance (Surplus/Deficit)', font=dict(size=16, family=TITLE_FONT)),
        xaxis_title='Fiscal Year',
        yaxis_title='KSh Trillion'
    )
    balance_chart.update_layout(**bc_layout)
    
    # Tax revenue composition pie
    latest_fiscal = fiscal_summary.iloc[-1]
    tax_components = ['Import_Duty', 'Excise_Duty', 'Income_Tax', 'VAT']
    tax_labels = ['Import Duty', 'Excise Duty', 'Income Tax', 'VAT']
    tax_values = [latest_fiscal[col] for col in tax_components if pd.notna(latest_fiscal[col])]
    tax_labels = [l for l, col in zip(tax_labels, tax_components) if pd.notna(latest_fiscal[col])]
    
    tax_pie = go.Figure(data=[go.Pie(
        labels=tax_labels, values=tax_values,
        hole=0.5,
        marker_colors=CHART_COLORS[:len(tax_labels)],
        textfont=dict(family=CHART_FONT)
    )])
    
    tp_layout = get_chart_layout(500)
    tp_layout.update(
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f'Tax Revenue Composition (FY{int(latest_fiscal["Fiscal_Year_End"])})', font=dict(size=16, family=TITLE_FONT)),
        xaxis=None, yaxis=None
    )
    tax_pie.update_layout(**tp_layout)
    
    return html.Div([
        html.H4("Fiscal Analysis", className="mb-4 gradient-text", style={'color': COLORS['text']}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=rev_exp, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=revenue_breakdown, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=balance_chart, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
            ], md=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([dcc.Graph(figure=tax_pie, config=CHART_CONFIG)])
                ], className="border-0 mb-4 glass-card")
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
        
        # Convert to trillions for display
        merged['GDP_T'] = merged['Nominal_GDP'] / 1e6
        merged['Debt_T'] = merged['Total_Debt'] / 1e6
        
        # GDP vs Inflation scatter (without trendline to avoid statsmodels issues)
        gdp_inflation = go.Figure()
        gdp_inflation.add_trace(go.Scatter(
            x=merged['GDP_T'], 
            y=merged['Avg_Inflation'],
            mode='markers',
            marker=dict(
                size=merged['GDP_Growth_Size'] * 3,
                color=merged['Year'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Year', titlefont=dict(family=CHART_FONT), tickfont=dict(family=CHART_FONT))
            ),
            text=merged.apply(lambda r: f"Year: {r['Year']}<br>GDP: {r['GDP_T']:.2f}T<br>Inflation: {r['Avg_Inflation']:.1f}%<br>Growth: {r['GDP_Growth']:.1f}%", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        # Add trendline manually using numpy
        z = np.polyfit(merged['GDP_T'], merged['Avg_Inflation'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['GDP_T'].min(), merged['GDP_T'].max(), 100)
        gdp_inflation.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
            name='Trend',
            showlegend=False
        ))
        
        gi_layout = get_chart_layout(400)
        gi_layout.update(
            title=dict(text='GDP vs Inflation (sized by growth rate)', font=dict(size=16, family=TITLE_FONT)),
            xaxis_title='Nominal GDP (KSh Trillion)',
            yaxis_title='Avg Inflation (%)'
        )
        gdp_inflation.update_layout(**gi_layout)
        
        # Debt vs GDP scatter
        debt_gdp_scatter = go.Figure()
        debt_gdp_scatter.add_trace(go.Scatter(
            x=merged['GDP_T'], 
            y=merged['Debt_T'],
            mode='markers',
            marker=dict(color=COLORS['tertiary'], size=12, line=dict(color=COLORS['background'], width=1)),
            text=merged.apply(lambda r: f"Year: {r['Year']}<br>GDP: {r['GDP_T']:.2f}T<br>Debt: {r['Debt_T']:.2f}T", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        # Add trendline
        z2 = np.polyfit(merged['GDP_T'], merged['Debt_T'], 1)
        p2 = np.poly1d(z2)
        debt_gdp_scatter.add_trace(go.Scatter(
            x=x_line, y=p2(x_line),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
            name='Trend',
            showlegend=False
        ))
        
        dg_layout = get_chart_layout(400)
        dg_layout.update(
            title=dict(text='GDP vs Total Public Debt', font=dict(size=16, family=TITLE_FONT)),
            xaxis_title='Nominal GDP (KSh Trillion)',
            yaxis_title='Total Debt (KSh Trillion)'
        )
        debt_gdp_scatter.update_layout(**dg_layout)
    
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
            textfont=dict(size=12, family=CHART_FONT),
            colorbar=dict(title='Correlation', titlefont=dict(family=CHART_FONT), tickfont=dict(family=CHART_FONT))
        ))
        
        ch_layout = get_chart_layout(450)
        ch_layout.update(
            margin=dict(l=100, r=40, t=60, b=100),
            title=dict(text='Correlation Matrix', font=dict(size=16, family=TITLE_FONT))
        )
        corr_heatmap.update_layout(**ch_layout)
        
        # Growth vs Inflation over time
        dual_axis = make_subplots(specs=[[{"secondary_y": True}]])
        dual_axis.add_trace(
            go.Scatter(x=merged['Year'], y=merged['GDP_Growth'], 
                      name='GDP Growth', line=dict(color=COLORS['primary'], width=3)),
            secondary_y=False
        )
        dual_axis.add_trace(
            go.Scatter(x=merged['Year'], y=merged['Avg_Inflation'],
                      name='Inflation', line=dict(color=COLORS['secondary'], width=3)),
            secondary_y=True
        )
        
        da_layout = get_chart_layout(450)
        da_layout.update(
            margin=dict(l=60, r=60, t=60, b=60),
            title=dict(text='GDP Growth vs Inflation Over Time', font=dict(size=16, family=TITLE_FONT)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )
        dual_axis.update_layout(**da_layout)
        
        dual_axis.update_xaxes(showgrid=False, title_font=dict(family=TITLE_FONT), tickfont=dict(family=CHART_FONT))
        dual_axis.update_yaxes(title_text="GDP Growth (%)", secondary_y=False, 
                              showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                              title_font=dict(family=TITLE_FONT), tickfont=dict(family=CHART_FONT))
        dual_axis.update_yaxes(title_text="Inflation (%)", secondary_y=True,
                              title_font=dict(family=TITLE_FONT), tickfont=dict(family=CHART_FONT))
        
        return html.Div([
            html.H4("Correlations & Relationships", className="mb-4 gradient-text", style={'color': COLORS['text']}),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=gdp_inflation, config=CHART_CONFIG)])
                    ], className="border-0 mb-4 glass-card")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=debt_gdp_scatter, config=CHART_CONFIG)])
                    ], className="border-0 mb-4 glass-card")
                ], md=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=corr_heatmap, config=CHART_CONFIG)])
                    ], className="border-0 mb-4 glass-card")
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([dcc.Graph(figure=dual_axis, config=CHART_CONFIG)])
                    ], className="border-0 mb-4 glass-card")
                ], md=6),
            ])
        ])
    except Exception as e:
        return html.Div([
            html.H4("Correlations & Relationships", className="mb-4 gradient-text", style={'color': COLORS['text']}),
            dbc.Alert(f"Error loading correlation data: {str(e)}", color="danger")
        ])


# ============== MAIN LAYOUT ==============
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Kenya Economic Dashboard", 
                       className="mb-1 header-title", 
                       style={'fontWeight': '700'}),
                html.P("Central Bank of Kenya Data Analysis", 
                      className="text-muted mb-0 font-mono",
                      style={'fontSize': '0.9rem'})
            ], className="py-4")
        ], md=8),
        dbc.Col([
            html.Div([
                html.Span("Data Source: ", className="text-muted"),
                html.Span("CBK", style={'color': COLORS['primary'], 'fontWeight': '600'})
            ], className="text-end py-4 font-mono")
        ], md=4)
    ], className="border-bottom mb-4", style={'borderColor': 'rgba(255,255,255,0.1)'}),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Overview", tab_id="overview"),
        dbc.Tab(label="GDP Analysis", tab_id="gdp"),
        dbc.Tab(label="Debt Analysis", tab_id="debt"),
        dbc.Tab(label="Inflation", tab_id="inflation"),
        dbc.Tab(label="Fiscal", tab_id="fiscal"),
        dbc.Tab(label="Correlations", tab_id="correlations"),
    ], id="tabs", active_tab="overview", className="mb-4 custom-tabs"),
    
    # Tab content
    html.Div(id="tab-content"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
            html.P([
                "Built with ",
                html.Span("Dash", style={'color': COLORS['primary']}),
                " & ",
                html.Span("Plotly", style={'color': COLORS['tertiary']}),
                " | Data: Central Bank of Kenya | ",
                html.A("yours sincerely", href="https://www.linkedin.com/in/john-chiwai/", target="_blank", style={'color': COLORS['secondary'], 'textDecoration': 'none', 'fontWeight': '600'})
            ], className="text-center text-muted font-mono", style={'fontSize': '0.8rem'})
        ])
    ], className="mt-5 mb-3")
    
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
    port = int(os.getenv("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)

