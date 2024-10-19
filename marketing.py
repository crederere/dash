import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import functools
import pytz

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit 설정
st.set_page_config(page_title="Wayer 대시보드", layout="wide")
st.title('Wayer 대시보드')

# 안전한 데이터프레임 접근을 위한 함수
def safe_get(df, col, default=0):
    return df[col].sum() if col in df.columns else default

# 데이터 로딩 함수
@st.cache_data
def load_data(uploaded_file):
    logging.info('엑셀 파일에서 데이터를 로드합니다.')
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheets = xls.sheet_names
        data = {}
        required_sheets = ['Meta_RD', 'abr_raw_01', 'abr_raw_02', 'retention_group']
        
        for sheet in required_sheets:
            if sheet in sheets:
                data[sheet] = pd.read_excel(xls, sheet_name=sheet)
            else:
                st.warning(f"{sheet} 시트가 엑셀 파일에 없습니다.")
                data[sheet] = pd.DataFrame()  # 빈 데이터프레임 생성
        
        logging.info('데이터 로드 완료.')
        return data['Meta_RD'], data['abr_raw_01'], data['abr_raw_02'], data['retention_group']
    except Exception as e:
        logging.error(f'데이터 로드 중 오류 발생: {e}')
        st.error(f'데이터 로드 중 오류 발생: {e}')
        return None, None, None, None

# 데이터 전처리 함수
def preprocess_data(meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df):
    logging.info('데이터 전처리를 시작합니다.')

    # Meta_RD 시트 전처리
    if not meta_rd_df.empty:
        meta_rd_df['Start Date'] = pd.to_datetime(meta_rd_df['기간'].str.split(' - ').str[0], errors='coerce').dt.tz_localize(KST)
        meta_rd_df['End Date'] = pd.to_datetime(meta_rd_df['기간'].str.split(' - ').str[1], errors='coerce').dt.tz_localize(KST)
        meta_rd_df['Event Date'] = pd.to_datetime(meta_rd_df['일'], errors='coerce').dt.tz_localize(KST)
        
        numeric_columns = ['지출 금액 (KRW)', '노출', '클릭(전체)', '결과', '구매', '구매 전환값', '고유 구매', '앱 설치', '앱 활성화', '고유 앱 활성화', '장바구니에 담기', '장바구니에 담기 전환값']
        for col in numeric_columns:
            if col in meta_rd_df.columns:
                meta_rd_df[col] = pd.to_numeric(meta_rd_df[col], errors='coerce').fillna(0)

    # abr_raw_01 시트 전처리
    if not abr_raw_01_df.empty:
        abr_raw_01_df['Event Date'] = pd.to_datetime(abr_raw_01_df['Event Date'], errors='coerce').dt.tz_localize(KST)
        numeric_columns_abr_01 = ['Total Install', 'Install', 'Total Reinstall', 'Signup(Unique)(App)', 'Signup(Unique)(Web)', 'Total Purchase(Total)', 'Revenue(App)', 'Total Revenue']
        for col in numeric_columns_abr_01:
            if col in abr_raw_01_df.columns:
                abr_raw_01_df[col] = pd.to_numeric(abr_raw_01_df[col], errors='coerce').fillna(0)

    # abr_raw_02 시트 전처리
    if not abr_raw_02_df.empty:
        abr_raw_02_df['Event Date'] = pd.to_datetime(abr_raw_02_df['Event Date'], errors='coerce').dt.tz_localize(KST)
        numeric_columns_abr_02 = ['Tutorial Complete', 'PGHD Total', 'StartTrial', 'Add to Cart', '튜토리얼 완료 유저 수 (App)', 'SKAN 튜토리얼 완료 유저 수 (App)', '튜토리얼 완료 유저 수(Web)', 'Complete_add_Workout 유저 수 (App)', 'Condition_details 유저 수 (App)', 'Fasting_Watch_Start 유저 수 (App)', 'Health_Connect 유저 수 (App)', 'Andys Feedback 유저 수 (App)', 'Routine_Check 유저 수 (App)', 'Steps_details 유저 수 (App)', 'Sugar_Details 유저 수 (App)', 'Water_Details 유저 수 (App)', 'Weight_details 유저 수 (App)', 'addToNutrients 유저 수 (App)', '체험판 시작 유저 수 (App)', 'SKAN 체험판 시작 유저 수 (App)', '상품리스트 조회 유저 수 (App)', '상품리스트 조회 유저 수 (Web)', 'SKAN 상품리스트 조회 유저 수 (App)', 'Reward 유저 수 (App)']
        for col in numeric_columns_abr_02:
            if col in abr_raw_02_df.columns:
                abr_raw_02_df[col] = pd.to_numeric(abr_raw_02_df[col], errors='coerce').fillna(0)

    # retention_group 시트 전처리
    if not retention_group_df.empty:
        retention_group_df['Start Date'] = pd.to_datetime(retention_group_df['Start Date'], errors='coerce').dt.tz_localize(KST)
        retention_columns = ['Total Start Events'] + [f'Day {i}' for i in range(31)]  # Day 0 to Day 30
        for col in retention_columns:
            if col in retention_group_df.columns:
                retention_group_df[col] = pd.to_numeric(retention_group_df[col], errors='coerce').fillna(0)

    logging.info('데이터 전처리 완료.')
    return meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df

# 요약 지표 계산 함수
def generate_metrics(meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df):
    logging.info('요약 지표를 계산합니다.')

    metrics = {}

    # Meta_RD 데이터에서 계산
    metrics['총 지출 (KRW)'] = safe_get(meta_rd_df, '지출 금액 (KRW)')
    metrics['총 노출 수'] = safe_get(meta_rd_df, '노출')
    metrics['총 클릭 수'] = safe_get(meta_rd_df, '클릭(전체)')
    metrics['CTR (%)'] = (metrics['총 클릭 수'] / metrics['총 노출 수'] * 100) if metrics['총 노출 수'] > 0 else 0
    metrics['총 설치 수'] = safe_get(meta_rd_df, '앱 설치')
    metrics['CPI (KRW)'] = metrics['총 지출 (KRW)'] / metrics['총 설치 수'] if metrics['총 설치 수'] > 0 else 0
    metrics['총 구매 수'] = safe_get(meta_rd_df, '구매')
    metrics['CPA (KRW)'] = metrics['총 지출 (KRW)'] / metrics['총 구매 수'] if metrics['총 구매 수'] > 0 else 0
    metrics['총 매출 (KRW)'] = safe_get(meta_rd_df, '구매 전환값')
    metrics['ROAS (%)'] = (metrics['총 매출 (KRW)'] / metrics['총 지출 (KRW)'] * 100) if metrics['총 지출 (KRW)'] > 0 else 0

    # ABR Raw 01 데이터에서 계산
    metrics['총 회원가입 수'] = safe_get(abr_raw_01_df, 'Signup(Unique)(App)') + safe_get(abr_raw_01_df, 'Signup(Unique)(Web)')
    metrics['회원가입률 (%)'] = (metrics['총 회원가입 수'] / metrics['총 클릭 수'] * 100) if metrics['총 클릭 수'] > 0 else 0

    # Retention 데이터에서 계산
    metrics['Day 1 Retention (%)'] = retention_group_df['Day 1'].mean() if 'Day 1' in retention_group_df.columns else 0
    metrics['Day 7 Retention (%)'] = retention_group_df['Day 7'].mean() if 'Day 7' in retention_group_df.columns else 0

    # ABR Raw 02 데이터에서 계산
    total_tutorial_completions = safe_get(abr_raw_02_df, '튜토리얼 완료 유저 수 (App)') + safe_get(abr_raw_02_df, '튜토리얼 완료 유저 수(Web)')
    metrics['튜토리얼 완료율 (%)'] = (total_tutorial_completions / metrics['총 설치 수'] * 100) if metrics['총 설치 수'] > 0 else 0

    # 추가 지표
    metrics['구매전환율 (%)'] = (metrics['총 구매 수'] / metrics['총 클릭 수'] * 100) if metrics['총 클릭 수'] > 0 else 0
    metrics['평균 구매액 (KRW)'] = metrics['총 매출 (KRW)'] / metrics['총 구매 수'] if metrics['총 구매 수'] > 0 else 0

    metrics_df = pd.DataFrame([metrics])
    logging.info('요약 지표 계산 완료.')
    return metrics_df

# 시각화 함수들
def plot_spend_trend(meta_rd_df):
    if meta_rd_df.empty:
        return None
    daily_spend = meta_rd_df.groupby('Event Date')['지출 금액 (KRW)'].sum().reset_index()
    if daily_spend.empty or len(daily_spend) < 2:
        return None
    fig = px.line(daily_spend, x='Event Date', y='지출 금액 (KRW)', title='일별 지출 추이')
    fig.update_layout(
        xaxis_title='날짜',
        yaxis_title='지출 금액 (KRW)',
        font=dict(size=14),
        hoverlabel=dict(font_size=16)
    )
    fig.update_traces(
        hovertemplate='날짜: %{x}<br>지출 금액: %{y:,.0f} KRW'
    )
    return fig

def plot_performance_metrics(meta_rd_df):
    if meta_rd_df.empty:
        return None
    daily_metrics = meta_rd_df.groupby('Event Date').agg({
        '노출': 'sum',
        '클릭(전체)': 'sum',
        '지출 금액 (KRW)': 'sum',
        '구매': 'sum',
        '구매 전환값': 'sum'
    }).reset_index()
    if daily_metrics.empty or len(daily_metrics) < 2:
        return None
    daily_metrics['CTR'] = daily_metrics['클릭(전체)'] / daily_metrics['노출'] * 100
    daily_metrics['CPC'] = daily_metrics['지출 금액 (KRW)'] / daily_metrics['클릭(전체)']
    daily_metrics['ROAS'] = daily_metrics['구매 전환값'] / daily_metrics['지출 금액 (KRW)'] * 100
    
    fig = make_subplots(rows=3, cols=2, subplot_titles=('일별 노출 수', '일별 클릭 수', 'CTR (%)', 'CPC (KRW)', 'ROAS (%)', '일별 구매 수'))
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['노출'], mode='lines', name='노출 수'), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['클릭(전체)'], mode='lines', name='클릭 수'), row=1, col=2)
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['CTR'], mode='lines', name='CTR'), row=2, col=1)
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['CPC'], mode='lines', name='CPC'), row=2, col=2)
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['ROAS'], mode='lines', name='ROAS'), row=3, col=1)
    fig.add_trace(go.Scatter(x=daily_metrics['Event Date'], y=daily_metrics['구매'], mode='lines', name='구매 수'), row=3, col=2)
    fig.update_layout(height=1000, title_text="주요 성과 지표 추이", font=dict(size=14))
    
    # 한글 툴팁 설정
    for i in fig['data']:
        if i['name'] == '노출 수':
            i['hovertemplate'] = '날짜: %{x}<br>노출 수: %{y:,.0f}'
        elif i['name'] == '클릭 수':
            i['hovertemplate'] = '날짜: %{x}<br>클릭 수: %{y:,.0f}'
        elif i['name'] == 'CTR':
            i['hovertemplate'] = '날짜: %{x}<br>CTR: %{y:.2f}%'
        elif i['name'] == 'CPC':
            i['hovertemplate'] = '날짜: %{x}<br>CPC: %{y:,.0f} KRW'
        elif i['name'] == 'ROAS':
            i['hovertemplate'] = '날짜: %{x}<br>ROAS: %{y:.2f}%'
        elif i['name'] == '구매 수':
            i['hovertemplate'] = '날짜: %{x}<br>구매 수: %{y:,.0f}'

    return fig

def plot_funnel(meta_rd_df, abr_raw_01_df):
    total_impressions = safe_get(meta_rd_df, '노출')
    total_clicks = safe_get(meta_rd_df, '클릭(전체)')
    total_installs = safe_get(meta_rd_df, '앱 설치')
    total_signups = safe_get(abr_raw_01_df, 'Signup(Unique)(App)') + safe_get(abr_raw_01_df, 'Signup(Unique)(Web)')
    total_purchases = safe_get(meta_rd_df, '구매')

    funnel_data = [total_impressions, total_clicks, total_installs, total_signups, total_purchases]
    funnel_labels = ['노출', '클릭', '앱 설치', '회원가입', '구매']

    if any(value == 0 for value in funnel_data):
        return None

    fig = go.Figure(go.Funnel(y=funnel_labels, x=funnel_data))
    fig.update_layout(title_text="마케팅 퍼널", font=dict(size=14))
    fig.update_traces(hovertemplate='단계: %{y}<br>수: %{x:,.0f}')
    return fig

def plot_retention(retention_group_df):
    if retention_group_df.empty:
        return None
    retention_columns = [col for col in retention_group_df.columns if col.startswith('Day ')]
    retention_data = retention_group_df[retention_columns].mean()
    
    if retention_data.empty or len(retention_data) < 2:
        return None

    days = range(len(retention_data))
    fig = px.line(x=days, y=retention_data.values, title='리텐션 추이')
    fig.update_layout(
        xaxis_title='Day',
        yaxis_title='리텐션 비율 (%)',
        font=dict(size=14),
        hoverlabel=dict(font_size=16)
    )
    fig.update_traces(
        hovertemplate='Day %{x}<br>리텐션: %{y:.2f}%'
    )
    return fig

def plot_channel_performance(meta_rd_df):
    if meta_rd_df.empty or 'Channel' not in meta_rd_df.columns:
        return None
    channel_perf = meta_rd_df.groupby('Channel').agg({
        '지출 금액 (KRW)': 'sum',
        '노출': 'sum',
        '클릭(전체)': 'sum',
        '구매': 'sum',
        '구매 전환값': 'sum'
    }).reset_index()
    
    if channel_perf.empty:
        return None

    channel_perf['CTR'] = channel_perf['클릭(전체)'] / channel_perf['노출'] * 100
    channel_perf['CPC'] = channel_perf['지출 금액 (KRW)'] / channel_perf['클릭(전체)']
    channel_perf['ROAS'] = channel_perf['구매 전환값'] / channel_perf['지출 금액 (KRW)'] * 100
    
    fig = px.bar(channel_perf, x='Channel', y=['지출 금액 (KRW)', '구매 전환값'], title='채널별 지출 및 매출')
    fig.add_trace(go.Scatter(x=channel_perf['Channel'], y=channel_perf['ROAS'], mode='lines+markers', name='ROAS (%)', yaxis='y2'))
    fig.update_layout(
        yaxis2=dict(title='ROAS (%)', overlaying='y', side='right'),
        font=dict(size=14),
        hoverlabel=dict(font_size=16)
    )
    fig.update_traces(
        hovertemplate='채널: %{x}<br>지출: %{y:,.0f} KRW<br>매출: %{y:,.0f} KRW'
    )
    return fig

def plot_daily_installs(abr_raw_01_df):
    if abr_raw_01_df.empty:
        return None
    daily_installs = abr_raw_01_df.groupby('Event Date')['Install'].sum().reset_index()
    if daily_installs.empty or len(daily_installs) < 2:
        return None
    fig = px.line(daily_installs, x='Event Date', y='Install', title='일별 앱 설치 수')
    fig.update_layout(
        xaxis_title='날짜',
        yaxis_title='설치 수',
        font=dict(size=14),
        hoverlabel=dict(font_size=16)
    )
    fig.update_traces(
        hovertemplate='날짜: %{x}<br>설치 수: %{y:,.0f}'
    )
    return fig

def plot_event_funnel(abr_raw_02_df):
    event_data = [
        safe_get(abr_raw_02_df, '튜토리얼 완료 유저 수 (App)'),
        safe_get(abr_raw_02_df, 'Complete_add_Workout 유저 수 (App)'),
        safe_get(abr_raw_02_df, '체험판 시작 유저 수 (App)'),
        safe_get(abr_raw_02_df, 'Add to Cart'),
        safe_get(abr_raw_02_df, 'StartTrial')
    ]
    event_labels = ['튜토리얼 완료', 'Workout 추가', '체험판 시작', '장바구니 추가', '트라이얼 시작']

    if any(value == 0 for value in event_data):
        return None

    fig = go.Figure(go.Funnel(y=event_labels, x=event_data))
    fig.update_layout(title_text="이벤트 퍼널", font=dict(size=14))
    fig.update_traces(hovertemplate='단계: %{y}<br>수: %{x:,.0f}')
    return fig

def plot_daily_roas(meta_rd_df):
    if meta_rd_df.empty:
        return None
    daily_roas = meta_rd_df.groupby('Event Date').agg({
        '지출 금액 (KRW)': 'sum',
        '구매 전환값': 'sum'
    }).reset_index()
    if daily_roas.empty or len(daily_roas) < 2:
        return None
    daily_roas['ROAS'] = daily_roas['구매 전환값'] / daily_roas['지출 금액 (KRW)'] * 100
    
    fig = px.line(daily_roas, x='Event Date', y='ROAS', title='일별 ROAS 추이')
    fig.update_layout(
        xaxis_title='날짜',
        yaxis_title='ROAS (%)',
        font=dict(size=14),
        hoverlabel=dict(font_size=16)
    )
    fig.update_traces(
        hovertemplate='날짜: %{x}<br>ROAS: %{y:.2f}%'
    )
    return fig

# 멀티셀렉트에서 'All' 옵션을 처리하는 함수
def get_multiselect(label, options, default=None):
    options = [str(option) for option in options]  # 모든 옵션을 문자열로 변환
    options = sorted(set(options))  # 중복 제거 후 정렬
    options_with_all = ["전체"] + options
    if default is None:
        default = ["전체"]
    selected = st.sidebar.multiselect(label, options_with_all, default=default)
    if "전체" in selected:
        return options
    return selected

# 메인 함수
def main():
    st.sidebar.header('엑셀 파일 업로드')
    uploaded_file = st.sidebar.file_uploader("전처리된 엑셀 파일을 업로드하세요.", type=['xlsx'])

    if uploaded_file:
        meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df = load_data(uploaded_file)

        if any(df is None for df in [meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df]):
            st.error('필요한 데이터 시트 중 하나 이상을 로드하지 못했습니다.')
            return

        meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df = preprocess_data(
            meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df)

        # 필터링 옵션
        st.sidebar.header('데이터 필터링')
        
        # 날짜 범위 선택
        all_dates = pd.concat([
            meta_rd_df['Event Date'] if 'Event Date' in meta_rd_df.columns else pd.Series(),
            abr_raw_01_df['Event Date'] if 'Event Date' in abr_raw_01_df.columns else pd.Series(),
            abr_raw_02_df['Event Date'] if 'Event Date' in abr_raw_02_df.columns else pd.Series(),
            retention_group_df['Start Date'] if 'Start Date' in retention_group_df.columns else pd.Series()
        ])
        date_min = all_dates.min().date()
        date_max = all_dates.max().date()
        if pd.isnull(date_min) or pd.isnull(date_max):
            st.error("유효한 날짜 데이터가 없습니다.")
            return

        # 날짜 선택 방식 개선
        date_option = st.sidebar.radio("날짜 선택 방식", ["날짜 범위", "최근 N일"])
        
        if date_option == "날짜 범위":
            start_date, end_date = st.sidebar.date_input(
                "날짜 범위 선택",
                [date_min, date_max],
                min_value=date_min,
                max_value=date_max
            )
        else:
            days = st.sidebar.number_input("최근 일수 선택", min_value=1, max_value=365, value=30)
            end_date = date_max
            start_date = end_date - timedelta(days=days-1)

        start_date = pd.Timestamp(start_date).tz_localize(KST)
        end_date = pd.Timestamp(end_date).tz_localize(KST).replace(hour=23, minute=59, second=59)

        # 채널 선택
# 채널 선택
        all_channels = set()
        for df in [meta_rd_df, abr_raw_01_df, abr_raw_02_df, retention_group_df]:
            if 'Channel' in df.columns:
                all_channels.update(df['Channel'].astype(str).unique())
        channels = list(all_channels)
        selected_channels = get_multiselect("채널 선택", channels, default=channels)

        # 캠페인 목표 선택 (Meta_RD 시트에만 해당)
        campaign_objectives = meta_rd_df['목표 이벤트'].unique() if '목표 이벤트' in meta_rd_df.columns else []
        selected_objectives = get_multiselect("캠페인 목표 선택", campaign_objectives, default=campaign_objectives)

        # 캠페인 구분 선택 (Meta_RD 시트에만 해당)
        campaign_segmentations = meta_rd_df['캠페인 구분'].unique() if '캠페인 구분' in meta_rd_df.columns else []
        selected_segmentations = get_multiselect("캠페인 구분 선택", campaign_segmentations, default=campaign_segmentations)

       # 데이터 필터링
        meta_rd_filtered = meta_rd_df.loc[
            (meta_rd_df['Event Date'] >= start_date) & 
            (meta_rd_df['Event Date'] <= end_date) &
            (meta_rd_df['Channel'].astype(str).isin(selected_channels)) &
            (meta_rd_df['목표 이벤트'].isin(selected_objectives)) &
            (meta_rd_df['캠페인 구분'].isin(selected_segmentations))
        ].copy()

        abr_raw_01_filtered = abr_raw_01_df.loc[
            (abr_raw_01_df['Event Date'] >= start_date) & 
            (abr_raw_01_df['Event Date'] <= end_date) &
            (abr_raw_01_df['Channel'].astype(str).isin(selected_channels))
        ].copy()

        abr_raw_02_filtered = abr_raw_02_df.loc[
            (abr_raw_02_df['Event Date'] >= start_date) & 
            (abr_raw_02_df['Event Date'] <= end_date) &
            (abr_raw_02_df['Channel'].astype(str).isin(selected_channels))
        ].copy()

        retention_filtered = retention_group_df.loc[
            (retention_group_df['Start Date'] >= start_date) & 
            (retention_group_df['Start Date'] <= end_date) &
            (retention_group_df['Channel'].astype(str).isin(selected_channels))
        ].copy()

        # 요약 지표 계산
        metrics_df = generate_metrics(meta_rd_filtered, abr_raw_01_filtered, abr_raw_02_filtered, retention_filtered)

        # 대시보드 레이아웃
        st.header("주요 성과 지표")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 지출 (KRW)", f"{metrics_df['총 지출 (KRW)'].values[0]:,.0f}")
        col2.metric("총 매출 (KRW)", f"{metrics_df['총 매출 (KRW)'].values[0]:,.0f}")
        col3.metric("ROAS (%)", f"{metrics_df['ROAS (%)'].values[0]:.2f}%")
        col4.metric("총 설치 수", f"{metrics_df['총 설치 수'].values[0]:,.0f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CPI (KRW)", f"{metrics_df['CPI (KRW)'].values[0]:,.2f}")
        col2.metric("CPA (KRW)", f"{metrics_df['CPA (KRW)'].values[0]:,.2f}")
        col3.metric("CTR (%)", f"{metrics_df['CTR (%)'].values[0]:.2f}%")
        col4.metric("회원가입률 (%)", f"{metrics_df['회원가입률 (%)'].values[0]:.2f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("구매전환율 (%)", f"{metrics_df['구매전환율 (%)'].values[0]:.2f}%")
        col2.metric("평균 구매액 (KRW)", f"{metrics_df['평균 구매액 (KRW)'].values[0]:,.2f}")
        col3.metric("Day 1 Retention (%)", f"{metrics_df['Day 1 Retention (%)'].values[0]:.2f}%")
        col4.metric("Day 7 Retention (%)", f"{metrics_df['Day 7 Retention (%)'].values[0]:.2f}%")

        st.header("시각화")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "지출 추이", "성과 지표", "마케팅 퍼널", "리텐션", "채널별 성과", "이벤트 퍼널", "ROAS 추이", "일별 앱 설치 수"
        ])
        
        with tab1:
            fig = plot_spend_trend(meta_rd_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("지출 추이를 표시할 데이터가 충분하지 않습니다.")

        with tab2:
            fig = plot_performance_metrics(meta_rd_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("성과 지표를 표시할 데이터가 충분하지 않습니다.")

        with tab3:
            fig = plot_funnel(meta_rd_filtered, abr_raw_01_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("마케팅 퍼널을 표시할 데이터가 충분하지 않습니다.")

        with tab4:
            fig = plot_retention(retention_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("리텐션 데이터를 표시할 데이터가 충분하지 않습니다.")

        with tab5:
            fig = plot_channel_performance(meta_rd_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("채널별 성과를 표시할 데이터가 충분하지 않습니다.")

        with tab6:
            fig = plot_event_funnel(abr_raw_02_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("이벤트 퍼널을 표시할 데이터가 충분하지 않습니다.")

        with tab7:
            fig = plot_daily_roas(meta_rd_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("ROAS 추이를 표시할 데이터가 충분하지 않습니다.")

        with tab8:
            fig = plot_daily_installs(abr_raw_01_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("일별 앱 설치 수를 표시할 데이터가 충분하지 않습니다.")

        # 날짜별 상세 데이터 표 추가
        st.header("날짜별 상세 데이터")
        
        # 데이터 준비
        daily_data = meta_rd_filtered.groupby('Event Date').agg({
            '지출 금액 (KRW)': 'sum',
            '노출': 'sum',
            '클릭(전체)': 'sum',
            '앱 설치': 'sum',
            '구매': 'sum',
            '구매 전환값': 'sum'
        }).reset_index()

        daily_data = daily_data.merge(
            abr_raw_01_filtered.groupby('Event Date').agg({
                'Install': 'sum',
                'Signup(Unique)(App)': 'sum',
                'Signup(Unique)(Web)': 'sum',
                'Total Purchase(Total)': 'sum',
                'Total Revenue': 'sum'
            }),
            on='Event Date',
            how='left'
        )

        daily_data = daily_data.merge(
            retention_filtered.groupby('Start Date').agg({
                'Day 1': 'mean',
                'Day 7': 'mean'
            }),
            left_on='Event Date',
            right_on='Start Date',
            how='left'
        )

        # 계산된 지표 추가
        daily_data['CTR (%)'] = (daily_data['클릭(전체)'] / daily_data['노출']) * 100
        daily_data['CVR (%)'] = (daily_data['구매'] / daily_data['클릭(전체)']) * 100
        daily_data['ROAS (%)'] = (daily_data['구매 전환값'] / daily_data['지출 금액 (KRW)']) * 100

        # 날짜 형식 변경 및 칼럼 순서 조정
        daily_data['Event Date'] = daily_data['Event Date'].dt.strftime('%Y-%m-%d')
        columns_order = [
            'Event Date', '지출 금액 (KRW)', '노출', '클릭(전체)', 'CTR (%)', '앱 설치', 
            'Install', 'Signup(Unique)(App)', 'Signup(Unique)(Web)', '구매', 
            'Total Purchase(Total)', '구매 전환값', 'Total Revenue', 'CVR (%)', 'ROAS (%)', 
            'Day 1', 'Day 7'
        ]
        daily_data = daily_data.reindex(columns=columns_order, fill_value=0)

        # 데이터 표시
        st.dataframe(daily_data.style.format({
            '지출 금액 (KRW)': '{:,.0f}',
            '노출': '{:,.0f}',
            '클릭(전체)': '{:,.0f}',
            'CTR (%)': '{:.2f}%',
            '앱 설치': '{:,.0f}',
            'Install': '{:,.0f}',
            'Signup(Unique)(App)': '{:,.0f}',
            'Signup(Unique)(Web)': '{:,.0f}',
            '구매': '{:,.0f}',
            'Total Purchase(Total)': '{:,.0f}',
            '구매 전환값': '{:,.0f}',
            'Total Revenue': '{:,.0f}',
            'CVR (%)': '{:.2f}%',
            'ROAS (%)': '{:.2f}%',
            'Day 1': '{:.2f}%',
            'Day 7': '{:.2f}%'
        }))

        # 심층 분석 섹션
        st.header("심층 분석")

        # 채널별 성과 분석
        st.subheader("채널별 성과 분석")
        if not meta_rd_filtered.empty and 'Channel' in meta_rd_filtered.columns:
            channel_performance = meta_rd_filtered.groupby('Channel').agg({
                '지출 금액 (KRW)': 'sum',
                '노출': 'sum',
                '클릭(전체)': 'sum',
                '구매': 'sum',
                '구매 전환값': 'sum'
            }).reset_index()
            channel_performance['ROAS'] = channel_performance['구매 전환값'] / channel_performance['지출 금액 (KRW)'] * 100
            channel_performance['CTR'] = channel_performance['클릭(전체)'] / channel_performance['노출'] * 100
            channel_performance['CVR'] = channel_performance['구매'] / channel_performance['클릭(전체)'] * 100
            st.dataframe(channel_performance.style.format({
                '지출 금액 (KRW)': '{:,.0f}',
                '노출': '{:,.0f}',
                '클릭(전체)': '{:,.0f}',
                '구매': '{:,.0f}',
                '구매 전환값': '{:,.0f}',
                'ROAS': '{:.2f}%',
                'CTR': '{:.2f}%',
                'CVR': '{:.2f}%'
            }))
        else:
            st.warning("채널별 성과 분석을 위한 데이터가 없습니다.")

        # 시간대별 성과 분석
        st.subheader("시간대별 성과 분석")
        if not meta_rd_filtered.empty and 'Event Date' in meta_rd_filtered.columns:
            meta_rd_filtered['Hour'] = meta_rd_filtered['Event Date'].dt.hour
            hourly_performance = meta_rd_filtered.groupby('Hour').agg({
                '지출 금액 (KRW)': 'sum',
                '노출': 'sum',
                '클릭(전체)': 'sum',
                '구매': 'sum',
                '구매 전환값': 'sum'
            }).reset_index()
            if not hourly_performance.empty and len(hourly_performance) >= 2:
                hourly_performance['CTR'] = hourly_performance['클릭(전체)'] / hourly_performance['노출'] * 100
                hourly_performance['CVR'] = hourly_performance['구매'] / hourly_performance['클릭(전체)'] * 100
                hourly_performance['ROAS'] = hourly_performance['구매 전환값'] / hourly_performance['지출 금액 (KRW)'] * 100
                fig_hourly = px.line(hourly_performance, x='Hour', y=['CTR', 'CVR', 'ROAS'], title='시간대별 CTR, CVR, ROAS')
                fig_hourly.update_layout(xaxis_title='시간', yaxis_title='비율 (%)', font=dict(size=14))
                fig_hourly.update_traces(hovertemplate='시간: %{x}<br>%{y:.2f}%')
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.warning("시간대별 성과 분석을 위한 데이터가 충분하지 않습니다.")
        else:
            st.warning("시간대별 성과 분석을 위한 데이터가 없습니다.")

        # 리텐션 분석 부분 수정
        st.subheader("리텐션 분석")
        if not retention_filtered.empty:
            # 숫자 데이터만 선택
            numeric_columns = retention_filtered.select_dtypes(include=[np.number]).columns
            retention_data = retention_filtered[numeric_columns].iloc[:, 3:]  # 'Total Start Events' 열 제외
            
            if not retention_data.empty:
                retention_heatmap = retention_data.mean().reset_index()
                retention_heatmap.columns = ['Day', 'Retention']
                fig_retention = px.imshow(retention_heatmap['Retention'].values.reshape(1, -1),
                                        labels=dict(x="Day", y="", color="Retention Rate"),
                                        x=retention_heatmap['Day'],
                                        color_continuous_scale="YlOrRd")
                fig_retention.update_layout(title="리텐션 히트맵", font=dict(size=14))
                fig_retention.update_traces(hovertemplate='Day: %{x}<br>Retention: %{z:.2f}%')
                st.plotly_chart(fig_retention, use_container_width=True)
            else:
                st.warning("리텐션 분석을 위한 숫자 데이터가 충분하지 않습니다.")
        else:
            st.warning("리텐션 분석을 위한 데이터가 없습니다.")

        # 이벤트 분석
        st.subheader("주요 이벤트 분석")
        if not abr_raw_02_filtered.empty:
            event_columns = ['튜토리얼 완료 유저 수 (App)', 'Fasting_Watch_Start 유저 수 (App)', 'Health_Connect 유저 수 (App)']
            available_event_columns = [col for col in event_columns if col in abr_raw_02_filtered.columns]
            if available_event_columns:
                event_counts = abr_raw_02_filtered[available_event_columns].sum()
                fig_events = px.bar(event_counts, title="주요 이벤트 발생 횟수")
                fig_events.update_layout(font=dict(size=14))
                fig_events.update_traces(hovertemplate='이벤트: %{x}<br>발생 횟수: %{y:,.0f}')
                st.plotly_chart(fig_events, use_container_width=True)
            else:
                st.warning("주요 이벤트 분석을 위한 데이터가 충분하지 않습니다.")
        else:
            st.warning("주요 이벤트 분석을 위한 데이터가 없습니다.")

        # 인사이트 및 추천 섹션
        st.header("인사이트 및 추천")
        
        if not meta_rd_filtered.empty:
            # Best Performing Channels
            best_channels = meta_rd_filtered.groupby('Channel').agg({
                '지출 금액 (KRW)': 'sum',
                '구매': 'sum',
                '구매 전환값': 'sum'
            }).reset_index()
            best_channels['ROAS'] = best_channels['구매 전환값'] / best_channels['지출 금액 (KRW)'] * 100
            best_channels = best_channels.sort_values('ROAS', ascending=False).head(5)
            
            st.subheader("최고 성과 채널 (Top 5 ROAS)")
            st.dataframe(best_channels[['Channel', '지출 금액 (KRW)', '구매', '구매 전환값', 'ROAS']].style.format({
                '지출 금액 (KRW)': '{:,.0f}',
                '구매': '{:,.0f}',
                '구매 전환값': '{:,.0f}',
                'ROAS': '{:.2f}%'
            }))

            # Worst Performing Channels
            worst_channels = meta_rd_filtered.groupby('Channel').agg({
                '지출 금액 (KRW)': 'sum',
                '구매': 'sum',
                '구매 전환값': 'sum'
            }).reset_index()
            worst_channels['ROAS'] = worst_channels['구매 전환값'] / worst_channels['지출 금액 (KRW)'] * 100
            worst_channels = worst_channels.sort_values('ROAS').head(5)
            
            st.subheader("저성과 채널 (Bottom 5 ROAS)")
            st.dataframe(worst_channels[['Channel', '지출 금액 (KRW)', '구매', '구매 전환값', 'ROAS']].style.format({
                '지출 금액 (KRW)': '{:,.0f}',
                '구매': '{:,.0f}',
                '구매 전환값': '{:,.0f}',
                'ROAS': '{:.2f}%'
            }))

            # 인사이트 생성
            total_roas = metrics_df['ROAS (%)'].values[0]
            avg_ctr = metrics_df['CTR (%)'].values[0]
            avg_conversion_rate = metrics_df['구매전환율 (%)'].values[0]

            st.subheader("주요 인사이트")
            insights = [
                f"전체 ROAS는 {total_roas:.2f}%입니다.",
                f"평균 CTR은 {avg_ctr:.2f}%입니다.",
                f"구매전환율은 {avg_conversion_rate:.2f}%입니다.",
                f"최고 성과 채널은 '{best_channels.iloc[0]['Channel']}'로, ROAS {best_channels.iloc[0]['ROAS']:.2f}%를 기록했습니다.",
                f"저성과 채널인 '{worst_channels.iloc[0]['Channel']}'의 ROAS는 {worst_channels.iloc[0]['ROAS']:.2f}%입니다."
            ]
            for insight in insights:
                st.write(f"- {insight}")

        else:
            st.warning("인사이트를 생성할 데이터가 충분하지 않습니다.")

        # 데이터 다운로드 옵션
        st.header("데이터 다운로드")
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8-sig')

        # 각 데이터프레임이 비어있지 않은 경우에만 다운로드 버튼 생성
        if not meta_rd_filtered.empty:
            csv_meta_rd = convert_df_to_csv(meta_rd_filtered)
            st.download_button(
                label="Meta_RD 데이터 다운로드",
                data=csv_meta_rd,
                file_name='meta_rd_filtered.csv',
                mime='text/csv',
            )
        
        if not abr_raw_01_filtered.empty:
            csv_abr_raw_01 = convert_df_to_csv(abr_raw_01_filtered)
            st.download_button(
                label="ABR Raw 01 데이터 다운로드",
                data=csv_abr_raw_01,
                file_name='abr_raw_01_filtered.csv',
                mime='text/csv',
            )
        
        if not abr_raw_02_filtered.empty:
            csv_abr_raw_02 = convert_df_to_csv(abr_raw_02_filtered)
            st.download_button(
                label="ABR Raw 02 데이터 다운로드",
                data=csv_abr_raw_02,
                file_name='abr_raw_02_filtered.csv',
                mime='text/csv',
            )
        
        if not retention_filtered.empty:
            csv_retention = convert_df_to_csv(retention_filtered)
            st.download_button(
                label="Retention 데이터 다운로드",
                data=csv_retention,
                file_name='retention_filtered.csv',
                mime='text/csv',
            )

    else:
        st.info('마케팅 데이터가 포함된 엑셀 파일을 업로드해주세요.')

# 에러 처리 함수
def handle_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            logging.error(f"오류 발생: {str(e)}", exc_info=True)
            st.error("데이터 형식을 확인하고 다시 시도해주세요.")
    return wrapper

# 메인 함수에 에러 처리 적용
@handle_error
def main_with_error_handling():
    main()

if __name__ == '__main__':
    main_with_error_handling()