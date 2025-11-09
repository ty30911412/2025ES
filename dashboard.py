# ---------------------------------------------------------------
# 誠致 2025 敬業度調查 - 互動式儀表板
# 執行方式: python3 -m streamlit run dashboard.py
# ---------------------------------------------------------------

from pandas.core.nanops import F
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import jieba 
import wordcloud
import matplotlib.pyplot as plt
import re # 用於質性頁面
import statsmodels.api as sm # [新增] 迴歸分析套件
from statsmodels.stats.proportion import proportion_confint

# --- 0. 頁面設定 ---
st.set_page_config(
    page_title="2025 誠致 Engagement Survey Dashboard",
    layout="wide"
)

# 設定 Plotly 預設主題
pio.templates.default = "plotly_white"

# --- 1. 資料載入 ---
# 使用 @st.cache_data 來加速載入，避免重複讀取
@st.cache_data
def load_data():
    try:
        # 彙總好的數值資料 (用於 Dashboard 主體)
        df_overall = pd.read_csv("numeric_descriptive_stats.csv")
        df_group = pd.read_csv("grouped_numeric_stats_by_Q2.csv")
        df_seniority = pd.read_csv("grouped_numeric_stats_by_Q4.csv")

        # 質性分析所需的「原始」資料
        df_codebook = pd.read_csv("codebook.csv")
        df_raw = pd.read_csv("2025 CZ Engagement survey (回覆) 的副本 - 表單回應 1.csv")
        # [新增] 載入 R 腳本清理後的「全數值」原始資料 (N=18)
        df_cleaned = pd.read_csv("cleaned_numeric_data.csv")
        # [重要] 幫原始資料套上 Q 編號
        df_raw.columns = df_codebook['New_Column'].values

        return df_overall, df_group, df_seniority, df_raw, df_codebook, df_cleaned

    except FileNotFoundError as e:
        st.error(f"錯誤：找不到必要的 CSV 檔案。請確保 {e.filename} 與 dashboard.py 在同一資料夾中。")
        st.error("請確認您已執行 R 腳本，並產出 'codebook.csv', 'numeric_descriptive_stats.csv' 等檔案。")
        return None, None, None, None, None

# [重要] 修改這一行，接收新載入的資料
df_overall, df_group, df_seniority, df_raw, df_codebook, df_cleaned = load_data()
st.sidebar.title("分析維度")
page = st.sidebar.radio(
    "選擇您要查看的頁面：",
    ("總體概況", 
    "選項顯著性",
     "依「組別」分析", 
     "依「年資」分析", 
     "關聯性分析",
     "質性回饋分析") # <-- 新增
)

# --- 3. 頁面內容 ---
st.title("2025 誠致 Engagement Survey Dashboard")

# ===================================================================
# 頁面一：總體概況
# ===================================================================
if page == "總體概況":
    
    st.header("總體概況 (N=18)")
    
    # (A) 顯示關鍵指標 (KPIs)
    st.subheader("關鍵指標 (Key Metrics)")
    try:
        # 抓取 Q104 (整體滿意度) 和 Q100 (留任傾向) 的平均分
        overall_satisfaction = df_overall.loc[df_overall['New_Column'] == 'Q104', 'Mean'].values[0]
        retention = df_overall.loc[df_overall['New_Column'] == 'Q100', 'Mean'].values[0]
        
        col1, col2 = st.columns(2)
        col1.metric("整體滿意度 (Q104)", f"{overall_satisfaction:.2f} / 10")
        col2.metric("留任傾向 (Q100)", f"{retention:.2f} / 2")
    except IndexError:
        st.error("錯誤：無法在 'numeric_descriptive_stats.csv' 中找到 Q104 或 Q100。請檢查 R 腳本是否正確執行。")

    st.divider()

    # (B) 顯示高分題 & 低分題
    st.subheader("高分題 vs. 低分題")
    
    # 過濾掉 'Q4', 'Q5', 'Q100' (因為它們的量尺不同，不適合放在一起比較)
    df_filtered_overall = df_overall[
        ~df_overall['New_Column'].isin(['Q4', 'Q5', 'Q100', 'Q104']) & (df_overall['N'] > 0)
    ]
    
    col1, col2 = st.columns(2)
    
    # 高分題
    with col1:
        st.write("平均分數最高的 10 題：")
        df_top10 = df_filtered_overall.nlargest(10, 'Mean')
        fig_top = px.bar(
            df_top10.sort_values('Mean'), # 排序以便繪圖
            x="Mean", 
            y="Original_Column", 
            orientation='h',
            text='Mean',
            title="Top 10 最高分題目"
        )
        fig_top.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        fig_top.update_layout(yaxis_title=None, xaxis_range=[df_top10['Mean'].min() * 0.9, df_top10['Mean'].max() * 1.05])
        st.plotly_chart(fig_top, use_container_width=True)

    # 低分題
    with col2:
        st.write("平均分數最低的 10 題：")
        df_low10 = df_filtered_overall.nsmallest(10, 'Mean')
        fig_low = px.bar(
            df_low10.sort_values('Mean', ascending=False), # 排序以便繪圖
            x="Mean", 
            y="Original_Column", 
            orientation='h',
            text='Mean',
            title="Top 10 最低分題目"
        )
        fig_low.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        fig_low.update_layout(yaxis_title=None, xaxis_range=[df_low10['Mean'].min() * 0.9, df_low10['Mean'].max() * 1.23])
        st.plotly_chart(fig_low, use_container_width=True)

    # --- [修改] 查看單一題目的描述性統計 (含選項分佈圖) ---
    st.subheader("查看單一題目統計與分佈")
    
    try:
        # (C.1) 取得所有數值型題目的列表
        all_numeric_questions = df_overall.sort_values(
            by='New_Column', 
            key=lambda col: col.str.replace('Q', '').astype(int)
        )['Original_Column'].tolist()
        
        # (C.2) 建立下拉選單
        selected_question_overall = st.selectbox(
            "請選擇您要查看統計數據的問題：",
            all_numeric_questions
        )
        
        # (C.3) 篩選出該題目的統計數據
        selected_stats = df_overall[df_overall['Original_Column'] == selected_question_overall]
        
        if not selected_stats.empty:
            
            # --- (A) 顯示描述性統計表格 (保留) ---
            st.write(f"#### 「{selected_question_overall}」的描述性統計：")
            st.dataframe(selected_stats[['N', 'Mean', 'SD', 'Median', 'Min', 'Max']].iloc[0].round(2).to_frame(name='數值'))

# --- (B) [新增] 顯示選項分佈長條圖 ---
            try:
                # B.1: 取得 Q 編號 (例如 'Q15')
                selected_q_id = selected_stats['New_Column'].values[0]
                
                # B.2: 從 df_raw (原始資料) 中取得該欄位的次數分配
                #      dropna() 會移除 'NA' (未填答)
                df_counts = df_raw[selected_q_id].dropna().value_counts().reset_index()
                df_counts.columns = ['選項 (原始文字)', '次數 (N)']
                
                # B.3: 繪圖 (依選項文字排序，使其符合 1, 2, 3, 4 的順序)
                fig_dist = px.bar(
                    df_counts.sort_values(by='選項 (原始文字)'), 
                    x='選項 (原始文字)', 
                    y='次數 (N)', 
                    text='次數 (N)',
                    title=f"「{selected_question_overall}」的選項分佈"
                )
                fig_dist.update_traces(textposition='outside')
                
                # B.4: [*** 修正 ***] 將 yaxis_range 放在正確的 update_layout 中
                max_y_val = df_counts['次數 (N)'].max()
                fig_dist.update_layout(
                    xaxis_title="填答選項", 
                    yaxis_title="次數 (N)",
                    yaxis_range=[0, max_y_val * 1.15] # 增加 15% 緩衝
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)

            except KeyError:
                # 如果 Q 編號不在 df_raw 中 (理論上不應發生)
                st.warning(f"無法在原始資料 (df_raw) 中找到 {selected_q_id} 的分佈資料。")
            except Exception as e_dist:
                st.error(f"繪製選項分佈圖時發生錯誤: {e_dist}")

        else:
            st.warning("找不到此題目的統計數據。")
            
    except Exception as e:
        st.error(f"查詢單一題目統計時發生錯誤: {e}")

# ===================================================================
# 頁面二：選項顯著性 (CI Test)  <-- [*** 新增區塊 ***]
# ===================================================================
elif page == "選項顯著性":
    st.header("選項顯著性分析 (95% 信賴區間)")
    st.warning(f"""
    **[分析說明]：**
    由於總樣本數 N={len(df_cleaned)} 過小，不適合執行傳統的卡方檢定（違反期望次數 > 5 的假設）。
    
    此頁面採用在統計上更穩健的**「95% 信賴區間 (Confidence Intervals)」** 來比較選項之間的差異。
    """)
    st.markdown("---")
    
    try:
        # 1. 取得所有「數值型」題目的列表 (用於下拉選單)
        all_numeric_questions = df_overall.sort_values(
            by='New_Column', 
            key=lambda col: col.str.replace('Q', '').astype(int)
        )['Original_Column'].tolist()
        
        # 2. 建立下拉選單
        selected_question_ci = st.selectbox(
            "請選擇您要分析顯著性的問題：",
            all_numeric_questions,
            key="ci_select"
        )
        
        # 3. 取得 Q 編號
        selected_q_id = df_overall[df_overall['Original_Column'] == selected_question_ci]['New_Column'].values[0]

        # 4. 從 df_raw (原始資料) 中取得該欄位的次數分配
        df_counts = df_raw[selected_q_id].dropna().value_counts().reset_index()
        df_counts.columns = ['選項 (原始文字)', '次數 (N)']
        
        # 5. 計算總 N 數 (用於計算佔比)
        N_total = df_counts['次數 (N)'].sum()
        
        # 6. 計算佔比與 95% 信賴區間
        df_counts['佔比'] = df_counts['次數 (N)'] / N_total
        
        # 使用 statsmodels 計算信賴區間 (Wilson C.I. 較適合小樣本)
        ci_low, ci_high = proportion_confint(df_counts['次數 (N)'], N_total, method='wilson')
        
        # 計算誤差線 (error bar) 需要的值
        df_counts['CI (下限)'] = ci_low
        df_counts['CI (上限)'] = ci_high
        df_counts['誤差 (上)'] = df_counts['CI (上限)'] - df_counts['佔比']
        df_counts['誤差 (下)'] = df_counts['佔比'] - df_counts['CI (下限)']

        
        # 7. 繪圖 (依選項文字排序)
        fig_ci = px.bar(
            df_counts.sort_values(by='選項 (原始文字)'), 
            x='選項 (原始文字)', 
            y='佔比', 
            text=df_counts.apply(lambda row: f"{row['佔比']:.1%} (N={row['次數 (N)']})", axis=1), # 顯示百分比和 N 數
            title=f"「{selected_question_ci}」的選項分佈 (含 95% 信賴區間)",
            error_y='誤差 (上)', # 加入誤差線 (上)
            error_y_minus='誤差 (下)' # 加入誤差線 (下)
        )
        fig_ci.update_traces(textposition='outside')
        
        # 8. 調整 Y 軸緩衝，並設定 Y 軸為百分比格式
        max_y_val = df_counts['CI (上限)'].max() # 以信賴區間的上限為基準
        fig_ci.update_layout(
            xaxis_title="填答選項", 
            yaxis_title="佔比 (Percentage)",
            yaxis_range=[0, max_y_val * 1.15], # 增加 15% 緩衝
            yaxis_tickformat='.0%' # Y 軸改為百分比
        )
        
        st.plotly_chart(fig_ci, use_container_width=True)
        
        st.info(
            "**如何解讀上圖：**\n"
            f"此圖表顯示了 N={N_total} 筆有效回覆中，各選項的佔比及其 95% 信賴區間 (黑色的T字誤差線)。\n\n"
            "**如果兩個選項的 95% 信賴區間（黑色誤差線）沒有互相重疊，代表它們的佔比差異達到了統計上的顯著水準 (p < .05)。**\n\n"
            f"（*例如：若『4-非常同意』的區間 [33%, 77%] 與『1-非常不同意』的區間 [0%, 18%] 沒有重疊，則代表選擇 4 的人顯著多於選擇 1 的人。*）"
        )

    except Exception as e_ci:
        st.error(f"繪製選項顯著性圖表時發生錯誤: {e_ci}")
        st.error(f"請確認 `statsmodels` 套件已加入 `requirements.txt` 並安裝成功。")

# ===================================================================
# 頁面二：依「組別」分析
# ===================================================================
elif page == "依「組別」分析":
    st.header("依「組別」(Q2) 分析")
    
    # (A) 讓使用者選擇要比較的題目
    question_list = df_group['Original_Column'].unique()
    
    selected_question = st.selectbox(
        "請選擇您要比較的問題：",
        question_list
    )
    
    # (B) 篩選出該題目的資料
    df_group_filtered = df_group[df_group['Original_Column'] == selected_question]
    
    # [新增] 抓取該題的「整體平均數」
    try:
        overall_mean_row = df_overall[df_overall['Original_Column'] == selected_question]
        overall_mean = overall_mean_row['Mean'].values[0]
    except (IndexError, TypeError):
        overall_mean = 0 # 備用，以防萬一
    
    # (C) 繪製分組長條圖
    if not df_group_filtered.empty:
        fig_group = px.bar(
            df_group_filtered,
            x="Q2",
            y="Mean",
            color="Q2",
            text="Mean",
            title=f"各組別在「{selected_question}」的平均分數"
        )
        fig_group.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        
        # [修改] 加入整體平均紅線 (使用手動座標)
        fig_group.add_hline(
            y=overall_mean, 
            line_dash="dot", 
            line_color="red",
            
            # 使用 annotation 字典手動指定位置
            annotation=dict(
                text=f"Mean: {overall_mean:.2f}",
                xref="paper",       # 使用圖表寬度的百分比
                x=0.85,             # 放在 95% 的位置 (非 100%)
                xanchor='right',    # 文字的右側對齊 95% 的位置
                yref="y",
                y=overall_mean,
                yanchor='bottom',   # 錨定在線的下方 (文字在線的上方)
                font=dict(color="gray"),
                showarrow=True
            )
        )
        
        # [修改] 整合 Y 軸範圍 (確保能容納長條圖與紅線)
        max_val = max(df_group_filtered['Mean'].max(), overall_mean)
        fig_group.update_layout(
            xaxis_title="組別",
            yaxis_range=[0, max_val * 1.15] # 增加 15% 緩衝
        )

        st.plotly_chart(fig_group, use_container_width=True)
        
        # (D) [重要] 顯示 N 數
        st.subheader("樣本數 (N) 提醒")
        st.write("請注意：由於 N 數極小，以下圖表僅供『描述性觀察』，不具統計推論意義。")
        st.dataframe(df_group_filtered[['Q2', 'N', 'Mean', 'SD', 'Median']].set_index('Q2'))
        
    else:
        st.warning("找不到此題目的分組資料。")

# ===================================================================
# 頁面三：依「年資」分析
# ===================================================================
elif page == "依「年資」分析":
    st.header("依「年資」(Q4) 分析")

    # (A) 讓使用者選擇要比較的題目
    question_list_sen = df_seniority['Original_Column'].unique()
    
    selected_question_sen = st.selectbox(
        "請選擇您要比較的問題：",
        question_list_sen
    )
    
    # (B) 篩選出該題目的資料
    df_sen_filtered = df_seniority[df_seniority['Original_Column'] == selected_question_sen]
    
    # [新增] 抓取該題的「整體平均數」
    try:
        overall_mean_row_sen = df_overall[df_overall['Original_Column'] == selected_question_sen]
        overall_mean_sen = overall_mean_row_sen['Mean'].values[0]
    except (IndexError, TypeError):
        overall_mean_sen = 0 # 備用，以防万一
    
    # (C) [重要] 確保年資的排序正確
    seniority_order = ["1 年以下", "1-2 年", "2-3 年", "3 年以上"]
    
    if not df_sen_filtered.empty:
        fig_sen = px.bar(
            df_sen_filtered,
            x="Q4_grouped",
            y="Mean",
            color="Q4_grouped",
            text="Mean",
            title=f"不同年資在「{selected_question_sen}」的平均分數"
        )
        fig_sen.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        
        # [修改] 加入整體平均紅線 (使用手動座標)
        fig_sen.add_hline(
            y=overall_mean_sen, 
            line_dash="dot", 
            line_color="red",
            
            # 使用 annotation 字典手動指定位置
            annotation=dict(
                text=f"Mean: {overall_mean_sen:.2f}",
                xref="paper",       # 使用圖表寬度的百分比
                x=0.85,             # 放在 95% 的位置 (非 100%)
                xanchor='right',    # 文字的右側對齊 95% 的位置
                yref="y",
                y=overall_mean_sen,
                yanchor='bottom',   # 錨定在線的下方 (文字在線的上方)
                font=dict(color="gray"),
                showarrow=True
            )
        )

        # [修改] 整合 Y 軸範圍 (確保能容納長條圖與紅線)
        max_val_sen = max(df_sen_filtered['Mean'].max(), overall_mean_sen)
        fig_sen.update_layout(
            xaxis_title="總工作年資",
            xaxis_categoryorder='array',
            xaxis_categoryarray=seniority_order,
            yaxis_range=[0, max_val_sen * 1.15] # 增加 15% 緩衝
        )

        st.plotly_chart(fig_sen, use_container_width=True)
        
        # (D) [重要] 顯示 N 數
        st.subheader("樣本數 (N) 提醒")
        st.write("請注意：由於 N 數極小，以下圖表僅供『描述性觀察』，不具統計推論意義。")
        st.dataframe(df_sen_filtered[['Q4_grouped', 'N', 'Mean', 'SD', 'Median']].set_index('Q4_grouped'))
        
    else:
        st.warning("找不到此題目的年資資料。")

# ===================================================================
# 頁面四：關聯性分析 (Correlation)  <-- [*** 修改後的區塊 ***]
# ===================================================================
elif page == "關聯性分析":
    
    st.header("關聯性分析")
    st.warning(f"""
    **[重要] 統計限制提醒：**
    由於總樣本數 N={len(df_cleaned)}，以下的相關係數與迴歸分析僅供**描述性觀察**。
    此樣本數 (N=18) 太小，無法進行有意義的統計推論。
    """)
    st.markdown("---")

    # --- 1. 準備題目列表 ---
    
    # 依變項 (Y) 的 Q 編號 (固定)
    Y_OPTIONS = {
        "Q104: 整體而言，我對於在 誠致 工作的滿意度是：": "Q104",
        "Q100: 我預計會在 誠致 持續任職": "Q100"
    }
    
    # 自變項 (X) 的選項 (排除 Q100, Q104, 和其他非評分題)
    # 我們從 df_overall 抓取所有題目，再排除 Y 軸的
    x_options_df = df_overall[
        ~df_overall['New_Column'].isin(['Q100', 'Q104', 'Q4', 'Q5']) & (df_overall['N'] > 0)
    ].sort_values(by='New_Column', key=lambda col: col.str.replace('Q', '').astype(int))
    
    # 建立 {題目文字: Q編號} 的 X 軸字典
    X_OPTIONS = x_options_df.set_index('Original_Column')['New_Column'].to_dict()
    
    # 建立 {Q編號: 題目文字} 的反向字典，供迴歸表使用
    X_MAP_INV = x_options_df.set_index('New_Column')['Original_Column'].to_dict()

    # --- 2. 建立子分頁 (Tabs) ---
    tab1, tab2, tab3 = st.tabs([
        "1. 動態散佈圖 (Scatter Plot)", 
        "2. 相關係數總表 (Correlation Rank)", 
        "3. 最佳預測變項 (Regression)"
    ])

    # ==================== TAB 1: 動態散佈圖 ====================
    with tab1:
        st.subheader("動態散佈圖 (X vs. Y)")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_y_text = st.selectbox(
                "選擇 依變項 (Y 軸)：",
                Y_OPTIONS.keys(),
                key="scatter_y"
            )
        with col2:
            selected_x_text = st.selectbox(
                "選擇 自變項 (X 軸)：",
                X_OPTIONS.keys(),
                key="scatter_x"
            )

        # 取得 Q 編號
        y_q_id = Y_OPTIONS[selected_y_text]
        x_q_id = X_OPTIONS[selected_x_text]
        
        st.markdown("---")

        # 準備分析資料
        df_corr_data = df_cleaned[[y_q_id, x_q_id]].dropna()
        df_corr_data.columns = ['Y_Value', 'X_Value']
        
        st.subheader(f"分析結果 (有效 N = {len(df_corr_data)})")

        # 計算相關係數
        correlation_r = df_corr_data['X_Value'].corr(df_corr_data['Y_Value'])
        st.metric(f"Pearson 相關係數 (r)", f"{correlation_r:.4f}")
        
        # 繪製散佈圖
        fig_corr = px.scatter(
            df_corr_data,
            x="X_Value",
            y="Y_Value",
            labels={"X_Value": f"X: {selected_x_text}", "Y_Value": f"Y: {selected_y_text}"},
            title=f"關聯性散佈圖",
            trendline="ols",
        )
        
        if y_q_id == 'Q100':
             fig_corr.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 0.5, 1.5, 2.0],
                    ticktext=["不考慮", "考慮 1 年內", "考慮 1-2 年", "考慮 2 年以上"]
                )
            )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==================== TAB 2: 相關係數總表 ====================
    with tab2:
        st.subheader("相關係數總表 (依 r 排序)")
        
        selected_y_corr_text = st.selectbox(
            "選擇要排序的 依變項 (Y 軸)：",
            Y_OPTIONS.keys(),
            key="corr_y"
        )
        y_q_id_corr = Y_OPTIONS[selected_y_corr_text]

        all_correlations = []
        y_series = df_cleaned[y_q_id_corr].dropna()
        
        for x_text, x_id in X_OPTIONS.items():
            x_series = df_cleaned[x_id]
            combined_series = pd.DataFrame({'Y': y_series, 'X': x_series}).dropna()
            
            if len(combined_series) > 1:
                corr = combined_series['Y'].corr(combined_series['X'])
                all_correlations.append({
                    "題目 (自變項)": x_text,
                    "相關係數 (r)": corr,
                    "N": len(combined_series)
                })
        
        df_all_corr_sorted = pd.DataFrame(all_correlations).sort_values(
            by="相關係數 (r)", 
            ascending=False
        )
        
        st.dataframe(
            df_all_corr_sorted.style.format({'相關係數 (r)': '{:.4f}'}),
            height=600,
            use_container_width=True
        )

    # ==================== TAB 3: 最佳預測變項 (迴歸) ====================
    with tab3:
        st.subheader("最佳預測變項 (依 R-Squared 排序)")
        st.error(f"**[!] 統計警告**：N=18，此表**不具推論意義**，僅供描述性參考。")

        selected_y_reg_text = st.selectbox(
            "選擇要預測的 依變項 (Y 軸)：",
            Y_OPTIONS.keys(),
            key="reg_y"
        )
        y_q_id_reg = Y_OPTIONS[selected_y_reg_text]

        all_regressions = []
        
        # 取得 Y 軸的數值 (N=18)
        Y_series_reg = df_cleaned[y_q_id_reg].dropna()

        # 遍歷所有 X 變項
        for x_text, x_id in X_OPTIONS.items():
            
            X_series_reg = df_cleaned[x_id]
            
            # 準備 X, Y 資料 (移除 NA)
            df_reg_data = pd.DataFrame({'Y': Y_series_reg, 'X': X_series_reg}).dropna()
            
            N_effective = len(df_reg_data)
            
            # 必須至少有 N > 2 才能進行簡單迴歸
            if N_effective > 2:
                Y_reg = df_reg_data['Y']
                X_reg = df_reg_data['X']
                
                # [重要] statsmodels 預設沒有截距項 (intercept)，我們必須手動加入
                X_reg_with_const = sm.add_constant(X_reg) 
                
                try:
                    model = sm.OLS(Y_reg, X_reg_with_const).fit()
                    
                    all_regressions.append({
                        "題目 (自變項)": x_text,
                        "R-Squared (解釋力)": model.rsquared,
                        "Coef (係數)": model.params.iloc[1], # [1] 是 X 的係數, [0] 是 const
                        "P>|t| (p-value)": model.pvalues.iloc[1],
                        "N (有效樣本)": N_effective
                    })
                except Exception as e:
                    # 處理計算錯誤 (例如 X 也是常數)
                    all_regressions.append({
                        "題目 (自變項)": x_text,
                        "R-Squared (解釋力)": 0.0,
                        "Coef (係數)": None,
                        "P>|t| (p-value)": None,
                        "N (有效樣本)": N_effective
                    })

        # 轉換為 DataFrame 並排序
        df_all_reg_sorted = pd.DataFrame(all_regressions).sort_values(
            by="R-Squared (解釋力)", 
            ascending=False
        )

        # 顯示表格
        st.dataframe(
            df_all_reg_sorted.style.format({
                'R-Squared (解釋力)': '{:.4f}',
                'Coef (係數)': '{:.3f}',
                'P>|t| (p-value)': '{:.3f}'
            }),
            height=600,
            use_container_width=True
        )
# ===================================================================
# 頁面四：質性回饋分析 (修正版，處理 int 錯誤)
# ===================================================================
elif page == "質性回饋分析":
    
    st.header("質性回饋分析")
    
# --- (A) 準備質性資料 ---
    
    # [真正正確的清單 v5.1，基於 codebook.csv]
    # (包含 Q105, Q108; 排除 Q106, Q107)
    QUALITATIVE_Q_NUMBERS = [
        "Q13",  # 承上題，如有「其他」原因...
        "Q42",  # 請描述或形容後台與學校協作的關係？
        "Q43",  # 與學校協作方式，做什麼樣的改變...
        "Q44",  # 針對 KIST 聯盟，我想要許願...
        "Q45",  # 針對聯盟協作經驗，我想要補充...
        "Q70",  # 我最喜歡 誠致 的地方是：
        "Q71",  # 如果調整或改變這些事情...
        "Q72",  # 相較過去我待過的組織...[更好]
        "Q73",  # 相較過去我待過的組織...[需要加強]
        "Q83",  # 誠致 所提供的 職場體驗，做什麼樣的改變...
        "Q95",  # 我目前工作上，最有意義感或持續成長的時刻？
        "Q102", # 對於自己在 誠致 的發展規劃與許願？
        "Q105", # 歡迎補充上一題滿意度評分的原因是？
        "Q107", #針對 誠致，我想要許願（非必填）
        "Q108"  # 最後，我還想說（非必填）
    ]
    
    # 建立一個 {Q編號: 原始題目} 的對應字典
    qual_questions_map = df_codebook[
        df_codebook['New_Column'].isin(QUALITATIVE_Q_NUMBERS)
    ].set_index('New_Column')['Original_Column'].to_dict()

    # 建立一個 {原始題目: Q編號} 的反向對應
    qual_questions_map_inv = {v: k for k, v in qual_questions_map.items()}

    # --- (B) 建立子頁面 (Tabs) ---
    tab1, tab2 = st.tabs(["回饋瀏覽", "詞雲"])

# --- Tab 1: 互動式回饋瀏覽器 ---
    with tab1:
        st.subheader("回饋瀏覽")
        
        # (1) 篩選器 [已移除組別與年資篩選]
        # 篩選質性問題
        selected_q_text = st.selectbox(
            "選擇要查看的質性問題：",
            qual_questions_map.values()
        )
        selected_q_id = qual_questions_map_inv[selected_q_text]

        # (2) 篩選資料 [已移除]
        # (No filtering needed)
            
        # (3) 顯示結果
        st.divider()
        
        # 取得該題目的所有非空回饋 (直接從 df_raw 取得)
        feedbacks = df_raw[selected_q_id].dropna().tolist()
        
        st.write(f"#### 顯示 {len(feedbacks)} 筆回饋 (來自 {selected_q_text})")
        
        # --- [BUG 修正] ---
        # 補回顯示回饋的迴圈 (Loop)
        if not feedbacks:
            st.info("此問題沒有任何回饋。")
        else:
            # 使用 expander 顯示每一筆回饋
            for i, feedback in enumerate(feedbacks):
                with st.expander(f"回饋 #{i+1}"):
                    st.write(str(feedback)) 
        # --- [修正結束] ---
    # --- Tab 2: 詞雲 ---
    with tab2:
        st.subheader("詞雲")
        st.warning("""
        **詞雲的限制：**
        1.  **斷詞：** 使用 `jieba` 斷詞，可能不完美。
        2.  **脈絡：** 詞雲會遺失「不滿意」的脈絡，請謹慎解讀。
        """)
        
        # (1) 讓使用者提供字體路徑
        font_path = ("jf-openhuninn-2.1.ttf")
        
        # (2) 選擇要生成詞雲的問題
        selected_q_text_wc = st.selectbox(
            "選擇要生成詞雲的質性問題：",
            qual_questions_map.values(),
            key="wc_select" # key 避免和 tab 1 衝突
        )
        
        if st.button("生成詞雲"):
            try:
                # 載入詞雲套件
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                import jieba
                
                # (A) 取得所有文字並合併
                selected_q_id_wc = qual_questions_map_inv[selected_q_text_wc]
                
                # --- [FIX] ---
                # 1. 取得 non-null list
                raw_list = df_raw[selected_q_id_wc].dropna().tolist()
                # 2. 強制將 list 中的 "所有" 項目轉換為 string
                string_list = [str(item) for item in raw_list]
                # 3. 再 join
                text_corpus = " ".join(string_list)
                # --- [FIX END] ---
                
                if not text_corpus:
                    st.info("此問題沒有任何文字回饋可生成詞雲。")
                else:
                    # (B) 建立中文停用詞 (Stop Words)
                    stop_words = set([
                        "的", "了", "我", "你", "他", "她", "我們", "你們", "他們", "她們",
                        "是", "在", "有", "也", "會", "就", "都", "還", "與", "和", "或",
                        "一個", "一些", "這個", "那個", "這些", "那些", "可以", "可能", 
                        "覺得", "希望", "比較", "如果", "但", "但是", "所以", "因為",
                        "不", "沒", "太", "很", "更", "最",
                        " ", "\n", "nan"
                    ])
                    
                    # (C) 使用 Jieba 斷詞
                    word_list = jieba.cut_for_search(text_corpus)
                    
                    # 過濾掉停用詞
                    filtered_words = [
                        word for word in word_list 
                        if word not in stop_words and len(word.strip()) > 1 
                    ]
                    
                    if not filtered_words:
                        st.info("過濾停用詞後，沒有足夠的詞彙可生成詞雲。")
                    else:
                        # (D) 生成詞雲
                        wc = WordCloud(
                            font_path=font_path, 
                            width=800,
                            height=400,
                            background_color="white",
                            collocations=False 
                        ).generate(" ".join(filtered_words))
                        
                        # (E) 繪製
                        fig, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)

            except FileNotFoundError:
                st.error(f"錯誤：找不到字體檔案 '{font_path}'。請確認路徑是否正確。")
            except ImportError:
                st.error("錯誤：缺少必要的套件。請執行 `pip3 install jieba wordcloud matplotlib`")
            except Exception as e:
                st.error(f"生成詞雲時發生錯誤：{e}")