import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta, date
from io import BytesIO
import zipfile
import altair as alt
# --------------------------------------------------------------------------------------
# 銘柄とセクターの設定 (ユーザー提供の定義を使用)
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="energy-l",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
DEFAULT_SECTOR = "総合商社"
SECTORS_RAW = {
    "総合商社": {
        '8058.T': '三菱商事', '8031.T': '三井物産', '8001.T': '伊藤忠商事',
        '8053.T': '住友商事', '8002.T': '丸紅', '8015.T': '豊田通商',
        '2768.T': '双日', '8020.T': '兼松',
    },
    "エネルギー資源": {
        '5020.T': 'ＥＮＥＯＳＨＤ', '5019.T': '出光興産', '5021.T': 'コスモエネルギーＨＤ',
        '1605.T': 'ＩＮＰＥＸ', '1662.T': '石油資源開発', '1515.T': '日鉄鉱業',
    },
    "主要電力": {
        '9509.T': '北海道電力', '9506.T': '東北電力', '9501.T': '東京電力ＨＤ',
        '9502.T': '中部電力', '9503.T': '関西電力', '9505.T': '北陸電力',
        '9504.T': '中国電力', '9507.T': '四国電力', '9508.T': '九州電力',
        '9511.T': '沖縄電力', '9513.T': '電源開発',
    },
    "電力電設": {
        '1934.T': 'ユアテック', '1942.T': '関電工', '1946.T': 'トーエネック',
        '1944.T': 'きんでん', '1930.T': '北陸電気工事', '1941.T': '中電工',
        '1959.T': '九電工', '1939.T': '四電工',
    },
    "電設工事": {
        '1417.T': 'ミライト・ワン', '1721.T': 'コムシスＨＤ', '1951.T': 'エクシオグループ',
        '1945.T': '東京エネシス', '1950.T': '日本電設工業', '1938.T': '日本リーテック',
    },
}
SECTORS = SECTORS_RAW
ALL_STOCKS_MAP = {ticker: name for sector in SECTORS_RAW.values() for ticker, name in sector.items()}
ALL_TICKERS_FLAT = list(ALL_STOCKS_MAP.keys())
ALL_TICKERS_WITH_N225 = list(set(ALL_TICKERS_FLAT + ['^N225']))

# 銘柄コードからセクター名を取得するための逆引きマップを作成
TICKER_TO_SECTOR = {}
for sector_name, tickers in SECTORS_RAW.items():
    for ticker in tickers.keys():
        TICKER_TO_SECTOR[ticker] = sector_name

# --------------------------------------------------------------------------------------
# データ取得とキャッシュを行う関数 (統合版)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=True, ttl=timedelta(minutes=30))
def load_ohlcv_data(tickers_list):
    """
    OHLCVデータを取得しキャッシュする関数。
    Wide形式 (MultiIndex) と Long形式のDataFrameを返す。
    """
    unique_tickers = list(set(tickers_list))
    if not unique_tickers:
        return pd.DataFrame(), pd.DataFrame() 
    try:
        tickers_obj = yf.Tickers(unique_tickers)
        data_wide = tickers_obj.history(period="max", interval="1d", auto_adjust=True)
    except Exception as e:
        st.error(f"yfinanceデータ取得エラー: {e}")
        return pd.DataFrame(), pd.DataFrame() 
    if data_wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    if len(unique_tickers) == 1 and not isinstance(data_wide.columns, pd.MultiIndex):
        data_wide.columns = pd.MultiIndex.from_product([data_wide.columns, unique_tickers], names=['Variable', 'Ticker'])
    data_wide = data_wide.dropna(axis=0, how='all')
    data_long = data_wide.stack(level='Ticker', future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    data_long = data_long.rename(columns={'Close': 'Close_Today', 'Open': 'Open_Today', 'High': 'High_Today', 'Low': 'Low_Today', 'Volume': 'Volume_Today'})
    data_long['Close_Yesterday'] = data_long.groupby('Ticker')['Close_Today'].shift(1)
    return data_wide, data_long

# --------------------------------------------------------------------------------------
# ダウンロード機能用補助関数
# --------------------------------------------------------------------------------------
def is_multiindex(df):
    """DataFrameがMultiIndexを持つかチェック"""
    return isinstance(df.columns, pd.MultiIndex)

# A1表記の列名を取得するヘルパー関数 (pandas 0.23以降のxlwt.col_strの代わり)
def get_column_letter(col_idx):
    """0-indexed column index to A1 notation column letter (e.g., 0 -> A, 26 -> AA)"""
    col_idx += 1  # 1-indexedにする
    letter = ''
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        letter = chr(65 + remainder) + letter
    return letter

def calculate_returns_with_dates(df, ticker, name_map, 基準日):
    """
    特定の銘柄について、基準日を基準とした短期（過去6営業日）と長期の騰落率を計算し、
    指定された日次の安値、高値、終値、RG、ボラ（価格の差）を追加する。
    """
    if is_multiindex(df):
        if ticker not in df.columns.levels[1]:
            return pd.DataFrame()
        ohlc = df.loc[:, (slice(None), ticker)]
        ohlc.columns = ohlc.columns.get_level_values(0)
    else:
        # 1銘柄だけの場合
        ohlc = df
        ohlc.columns = ohlc.columns.get_level_values(0)

    ohlc = ohlc.dropna()
    if ohlc.empty or "Close" not in ohlc.columns:
        return pd.DataFrame()
    available_dates = ohlc.index[ohlc.index <= 基準日]
    if len(available_dates) == 0:
        return pd.DataFrame()

    base_date = available_dates.max()
    base_idx = ohlc.index.get_loc(base_date)
    
    # 銘柄データフレームの初期化
    result = pd.DataFrame()
    # セクター列を追加 (日経平均の場合は空欄/インデックス/その他を設定)
    sector_name = TICKER_TO_SECTOR.get(ticker, "")
    result["セクター"] = [sector_name] if ticker != '^N225' else ["インデックス"]
    result["銘柄コード"] = [ticker]
    result["銘柄名"] = [name_map.get(ticker, "日経平均" if ticker == '^N225' else ticker)]
    result["株価"] = [ohlc["Close"].iloc[base_idx]]
    
    # 短期（過去6営業日）の騰落率とRG/ボラを計算
    for i in range(0, 6): 
        target_idx = base_idx - i
        prior_idx = base_idx - (i + 1)

        if prior_idx >= 0 and target_idx >= 0:
            date_str = ohlc.index[target_idx].strftime("%Y-%m-%d")
            prior_close = ohlc["Close"].iloc[prior_idx] 
            
            # 安値/高値/終値 (騰落率 %)
            result[f"{date_str}_安値"] = ((ohlc["Low"].iloc[target_idx] - prior_close) / prior_close * 100).round(2)
            result[f"{date_str}_高値"] = ((ohlc["High"].iloc[target_idx] - prior_close) / prior_close * 100).round(2)
            result[f"{date_str}_終値"] = ((ohlc["Close"].iloc[target_idx] - prior_close) / prior_close * 100).round(2)
            
            # RG (値幅の騰落率 %)
            result[f"{date_str}_ＲＧ"] = ((ohlc["High"].iloc[target_idx] - ohlc["Low"].iloc[target_idx]) / prior_close * 100).round(2)
            
            # ボラ (値幅の金額)
            result[f"{date_str}_ボラ"] = (ohlc["High"].iloc[target_idx] - ohlc["Low"].iloc[target_idx]).round(2) 
        else:
            # データがない場合はNone
            date_str = ohlc.index[target_idx].strftime("%Y-%m-%d") if target_idx >= 0 and target_idx < len(ohlc.index) else f"D-{i+1}" 
            result[f"{date_str}_安値"] = None
            result[f"{date_str}_高値"] = None
            result[f"{date_str}_終値"] = None
            result[f"{date_str}_ＲＧ"] = None
            result[f"{date_str}_ボラ"] = None
            
    # 長期の騰落率を計算
    periods = {
        "5d": 5,
        "10d": 10,
        "1mo": 21,
        "2mo": 42,
        "3mo": 63,
        "4mo": 84,
        "5mo": 105,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
    }
    for label, days in periods.items():
        prior_idx = base_idx - days
        if prior_idx >= 0:
            prior_close = ohlc["Close"].iloc[prior_idx]
            base_close = ohlc["Close"].iloc[base_idx]
            result[label] = ((base_close - prior_close) / prior_close * 100).round(2)
        else:
            result[label] = None
            
    return result

# --------------------------------------------------------------------------------------
# グラフ機能用補助関数 
# --------------------------------------------------------------------------------------
def get_stock_name(ticker_code):
    """ティッカーコードから銘柄名を取得"""
    if ticker_code == '^N225':
        return "日経平均"
    return ALL_STOCKS_MAP.get(ticker_code, ticker_code)

def reset_stock_selection():
    """セクター変更時に銘柄選択をリセットするためのコールバック"""
    st.session_state["_stock_selection_needs_reset"] = True

def calculate_returns_and_range(df_ohlcv_long: pd.DataFrame, filtered_tickers: list) -> dict:
    """
    日次の終値、安値、高値、レンジの騰落率を計算し、グラフ用のLong形式データとして返す。
    """
    if df_ohlcv_long.empty or not filtered_tickers:
        return {}
    df = df_ohlcv_long[df_ohlcv_long['Ticker'].isin(filtered_tickers)].dropna(subset=['Close_Yesterday']).copy()
    df['Close_vs_Close'] = ((df['Close_Today'] - df['Close_Yesterday']) / df['Close_Yesterday']) * 100
    df['Close_vs_Low'] = ((df['Low_Today'] - df['Close_Yesterday']) / df['Close_Yesterday']) * 100
    df['Close_vs_High'] = ((df['High_Today'] - df['Close_Yesterday']) / df['Close_Yesterday']) * 100
    df['Daily_Range_Percent'] = ((df['High_Today'] - df['Low_Today']) / df['Close_Yesterday']) * 100
    df['Color_Close'] = df['Close_vs_Close'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df['Color_Low'] = df['Close_vs_Low'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df['Color_High'] = df['Close_vs_High'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df['Stock_Gained'] = df.groupby('Ticker')['Close_Today'].transform(lambda x: x.diff().gt(0))
    df['Color_Range'] = df['Stock_Gained'].apply(lambda x: 'Positive' if x else 'Negative')
    data_close = df[['Date', 'Ticker', 'Close_vs_Close', 'Color_Close']].rename(columns={'Close_vs_Close': 'Value', 'Color_Close': 'Color'})
    data_low = df[['Date', 'Ticker', 'Close_vs_Low', 'Color_Low']].rename(columns={'Close_vs_Low': 'Value', 'Color_Low': 'Color'})
    data_high = df[['Date', 'Ticker', 'Close_vs_High', 'Color_High']].rename(columns={'Close_vs_High': 'Value', 'Color_High': 'Color'})
    data_range = df[['Date', 'Ticker', 'Daily_Range_Percent', 'Color_Range']].rename(columns={'Daily_Range_Percent': 'Value', 'Color_Range': 'Color'})
    max_rows = 750 * len(filtered_tickers)
    data_close = data_close.tail(max_rows)
    data_low = data_low.tail(max_rows)
    data_high = data_high.tail(max_rows)
    data_range = data_range.tail(max_rows) 
    return {
        "終値": data_close,
        "安値": data_low,
        "高値": data_high,
        "レンジ": data_range
    }

def create_and_display_bar_charts(plot_df_all: pd.DataFrame, filtered_stocks: dict, tab_name: str, y_domain_gain=None):
    """
    各指標の棒グラフをAltairで描画する。
    """
    if plot_df_all.empty:
        st.info(f"{tab_name}グラフを表示するためのデータがありません。")
        return 
    current_plot_tickers = [t for t in filtered_stocks.keys() if t in plot_df_all['Ticker'].unique()] 
    if not current_plot_tickers:
        st.info(f"{tab_name}グラフを表示するためのデータがありません。")
        return 
    num_cols = 1 
    if tab_name != "レンジ" and y_domain_gain is not None:
        y_domain = y_domain_gain
    else:
        y_domain = 'unaggregated' 
    y_title = None
    y_format = "+.1f"
    color_range = ['#008000', '#C70025'] 
    plot_df_all['Date'] = plot_df_all['Date'].dt.date
    for ticker in current_plot_tickers:
        stock_name = get_stock_name(ticker) 
        plot_df = plot_df_all[plot_df_all['Ticker'] == ticker].copy() 
        x_format = "%Y/%m" 
        chart = alt.Chart(plot_df).mark_bar().encode(
            alt.X("Date:T", axis=alt.Axis(
                title=None,
                format=x_format,
                labelAngle=0,
                tickCount= 'month' 
            )),
            alt.Y("Value:Q", axis=alt.Axis(title=None, format=y_format),
                scale=alt.Scale(domain=y_domain)
            ),
            alt.Color('Color:N',
                scale=alt.Scale(domain=['Positive', 'Negative'], range=color_range),
                legend=None),
            tooltip=[
                alt.Tooltip("Date:T", title="日付", format="%Y/%m/%d"),
                alt.Tooltip("Value:Q", title="騰落率", format="+.2f"),
                alt.Tooltip("Color:N", title="傾向")
            ]
        ).properties(
            title=f"{ticker[:4]} {stock_name}",
            height=300, 
        ) 
        # 💡 修正: use_container_width=True を width='stretch' に変更
        st.altair_chart(chart, width='stretch')
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# データ取得の実行 (Wide/Long形式の両方を一度に取得)
# --------------------------------------------------------------------------------------
ohlcv_data_wide, ohlcv_data_long = pd.DataFrame(), pd.DataFrame()
try:
    with st.spinner(f"全銘柄の日次データをロード中 (最大期間)..."):
        # 日経平均は集計データには含めないが、グラフのために取得対象に含める
        download_tickers_plus_n225 = ALL_TICKERS_FLAT + ['^N225']
        ohlcv_data_wide, ohlcv_data_long = load_ohlcv_data(download_tickers_plus_n225) 
except Exception as e:
    if "YFRateLimitError" in str(e):
        st.warning("YFinanceの接続制限が発生しています。しばらくしてから再試行してください。")
        load_ohlcv_data.clear()
    else:
        st.error(f"データ読み込みエラー: {e}")
    st.stop() 
if ohlcv_data_wide.empty or ohlcv_data_long.empty:
    st.error("OHLCVデータが取得できませんでした。アプリケーションを終了します。")
    st.stop()

# --------------------------------------------------------------------------------------
# ウィジェットの配置 (基準日、セクター、銘柄)
# --------------------------------------------------------------------------------------
st.markdown("## ⚙️ Download")
""
col_download, col_select_sector, col_select_stock = st.columns([1, 1.5, 6])
with col_download:
    latest_date = ohlcv_data_wide.index.max().date() if not ohlcv_data_wide.empty else pd.Timestamp.today().date()
    # 基準日（ラベルは非表示だが、警告回避のため設定）
    基準日 = st.date_input("基準日_日付", latest_date, label_visibility="collapsed")
    基準日 = pd.Timestamp(基準日)
with col_select_sector:
    sector_options = list(SECTORS.keys())
    default_sector_key = DEFAULT_SECTOR
    default_sectors = st.session_state.get("multiselect_sectors", [default_sector_key])
    selected_sectors = st.multiselect(
        "セクターを選択",
        options=sector_options,
        default=default_sectors,
        key="multiselect_sectors",
        label_visibility="collapsed",
        on_change=reset_stock_selection
    ) 
SELECTED_SECTOR_STOCKS_MAP = {}
if selected_sectors:
    for sector in selected_sectors:
        SELECTED_SECTOR_STOCKS_MAP.update(SECTORS.get(sector, {}))
else:
    SELECTED_SECTOR_STOCKS_MAP = ALL_STOCKS_MAP 

stock_options = [name for name in SELECTED_SECTOR_STOCKS_MAP.values()]
all_current_stock_names = stock_options
if "日経平均" not in all_current_stock_names:
    all_current_stock_names.append("日経平均")
if "multiselect_stocks" not in st.session_state:
    st.session_state["multiselect_stocks"] = all_current_stock_names
elif st.session_state.get("_stock_selection_needs_reset"):
    st.session_state["multiselect_stocks"] = all_current_stock_names
    del st.session_state["_stock_selection_needs_reset"]
else:
    current_selection = st.session_state["multiselect_stocks"]
    st.session_state["multiselect_stocks"] = [name for name in current_selection if name in all_current_stock_names]

with col_select_stock:
    selected_stock_names = st.multiselect(
        "銘柄を選択",
        options=all_current_stock_names,
        key="multiselect_stocks",
        label_visibility="collapsed"
    ) 
FINAL_STOCKS_MAP = {}
name_to_ticker = {name: ticker for ticker, name in ALL_STOCKS_MAP.items()}
name_to_ticker["日経平均"] = '^N225'
for name in selected_stock_names:
    ticker = name_to_ticker.get(name)
    if ticker:
        FINAL_STOCKS_MAP[ticker] = name

FILTERED_STOCKS = FINAL_STOCKS_MAP
FILTERED_TICKERS = list(FINAL_STOCKS_MAP.keys())

# --- ダウンロード処理 ---
with col_download:
    # 💡 修正: use_container_width=True を width='stretch' に変更
    if st.button("📥 Download", width='stretch'): 
        all_results = []
        progress_bar = st.progress(0)
        # ダウンロード対象は設定されている全銘柄 (日経平均含む)
        download_tickers = ALL_TICKERS_FLAT + (['^N225'] if '^N225' in ALL_TICKERS_WITH_N225 else [])
        
        if not download_tickers:
            st.error("ダウンロード対象の銘柄がありません。")
        else:
            with st.spinner(f"全 {len(download_tickers)} 銘柄の騰落率計算中..."):
                ohlcv_data_filtered = ohlcv_data_wide[ohlcv_data_wide.index <= 基準日]
                
                for i, ticker in enumerate(download_tickers):
                    res = calculate_returns_with_dates(ohlcv_data_filtered, ticker, ALL_STOCKS_MAP, 基準日)
                    if not res.empty:
                        all_results.append(res)
                    progress = (i + 1) / len(download_tickers)
                    progress_bar.progress(progress) 
                
                progress_bar.empty() 
                
                if len(all_results) == 0:
                    st.error("騰落率の計算に成功した銘柄がありませんでした。基準日を確認してください。")
                else:
                    final_df = pd.concat(all_results, ignore_index=True)
                    
                    # 日経平均 (^N225) の行を先頭に移動させる
                    n225_df = final_df[final_df['銘柄コード'] == '^N225']
                    other_stocks_df = final_df[final_df['銘柄コード'] != '^N225']
                    other_stocks_df = other_stocks_df.sort_values("銘柄コード").reset_index(drop=True)
                    final_df = pd.concat([n225_df, other_stocks_df], ignore_index=True)
                    
                    # --- Excelファイル分割のための列名定義 ---
                    BASE_COLS = ["セクター", "銘柄コード", "銘柄名", "株価"]
                    DAILY_RETURNS_COLS = [col for col in final_df.columns if '_安値' in col or '_高値' in col or '_終値' in col]
                    LONG_TERM_RETURNS_COLS = [col for col in final_df.columns if col in ["5d", "10d", "1mo", "2mo", "3mo", "4mo", "5mo", "6mo", "1y", "2y"]]
                    TABLE1_COLS = BASE_COLS + DAILY_RETURNS_COLS + LONG_TERM_RETURNS_COLS
                    RG_VORA_COLS = [col for col in final_df.columns if '_ＲＧ' in col or '_ボラ' in col]
                    TABLE2_COLS = BASE_COLS + RG_VORA_COLS
                    CONDITIONAL_FORMAT_COLS = DAILY_RETURNS_COLS + LONG_TERM_RETURNS_COLS
                    
                    if CONDITIONAL_FORMAT_COLS:
                        first_data_col_idx = TABLE1_COLS.index(CONDITIONAL_FORMAT_COLS[0])
                        last_data_col_idx = TABLE1_COLS.index(CONDITIONAL_FORMAT_COLS[-1])
                        first_col_letter = get_column_letter(first_data_col_idx)
                        last_col_letter = get_column_letter(last_data_col_idx)
                    else:
                        first_data_col_idx = -1
                        last_data_col_idx = -1
                        
                    
                    split_size = 1000
                    excel_buffers = []
                    
                    for i in range(0, len(final_df), split_size):
                        chunk = final_df.iloc[i:i+split_size]
                        table1_df = chunk[TABLE1_COLS]
                        table2_df = chunk[TABLE2_COLS]
                        buffer = BytesIO()
                        
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            table1_df.to_excel(writer, sheet_name='Daily_Returns_LongTerm', index=False)
                            table2_df.to_excel(writer, sheet_name='Daily_Range_Vola', index=False)
                            
                            workbook = writer.book
                            worksheet = writer.sheets['Daily_Returns_LongTerm']

                            if first_data_col_idx != -1 and last_data_col_idx != -1:
                                format_positive = workbook.add_format({'font_color': '#008000', 'num_format': '0.0'}) # 緑
                                format_negative = workbook.add_format({'font_color': '#C70025', 'num_format': '0.0'}) # 赤
                                format_neutral = workbook.add_format({'num_format': '0.0'}) # ゼロ・未定義用 (デフォルト)

                                conditional_range = f"{first_col_letter}2:{last_col_letter}{len(chunk) + 1}"

                                worksheet.set_column(first_data_col_idx, last_data_col_idx, None, format_neutral)
                                
                                worksheet.conditional_format(
                                    conditional_range,
                                    {'type': 'cell', 'criteria': '>', 'value': 0, 'format': format_positive}
                                )

                                worksheet.conditional_format(
                                    conditional_range,
                                    {'type': 'cell', 'criteria': '<', 'value': 0, 'format': format_negative}
                                )

                        buffer.seek(0)
                        excel_buffers.append((f"daily_returns_part_{i//split_size + 1}.xlsx", buffer))
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for file_name, buffer in excel_buffers:
                            zf.writestr(file_name, buffer.getvalue())
                    zip_buffer.seek(0)
                    st.download_button(
                        label=f"✅ ZIPをダウンロード (全{len(final_df)}銘柄)",
                        data=zip_buffer,
                        file_name=f"daily_returns_at_{基準日.strftime('%Y%m%d')}.zip",
                        mime="application/zip",
                        # 💡 修正: use_container_width=True を width='stretch' に変更
                        width='stretch',
                    )

# --------------------------------------------------------------------------------------
# グラフ機能の実行 
# --------------------------------------------------------------------------------------
""
st.markdown("## 📊 Chart")
""
final_y_min_gain = -20.0
final_y_max_gain = 20.0
if FILTERED_STOCKS:
    chart_data = calculate_returns_and_range(ohlcv_data_long, FILTERED_TICKERS)
    tab_close, tab_low, tab_high, tab_range = st.tabs(["終値", "安値", "高値", "レンジ"])
    with tab_close:
        if "終値" in chart_data and not chart_data["終値"].empty:
            create_and_display_bar_charts(
                chart_data["終値"],
                FILTERED_STOCKS,
                "終値",
                [final_y_min_gain, final_y_max_gain]
            )
        else:
            st.info("終値グラフを表示するためのデータが不足しています。")
    with tab_low:
        if "安値" in chart_data and not chart_data["安値"].empty:
            create_and_display_bar_charts(
                chart_data["安値"],
                FILTERED_STOCKS,
                "安値",
                [final_y_min_gain, final_y_max_gain]
            )
        else:
            st.info("安値グラフを表示するためのデータが不足しています。")
    with tab_high:
        if "高値" in chart_data and not chart_data["高値"].empty:
            create_and_display_bar_charts(
                chart_data["高値"],
                FILTERED_STOCKS,
                "高値",
                [final_y_min_gain, final_y_max_gain]
            )
        else:
            st.info("高値グラフを表示するためのデータが不足しています。")
    with tab_range:
        if "レンジ" in chart_data and not chart_data["レンジ"].empty:
            create_and_display_bar_charts(
                chart_data["レンジ"],
                FILTERED_STOCKS,
                "レンジ",
            )
        else:
            st.info("レンジグラフを表示するためのデータが不足しています。")
else:
    st.info("グラフを表示する銘柄が選択されていません。セクターまたは銘柄を選択してください。")