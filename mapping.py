import pandas as pd
import os
import re
from rapidfuzz import fuzz # 用於模糊比對

# --- 設定 ---
BACKEND_FILE = os.path.join('numeric_descriptive_stats.csv')
TEACHER_FILE = '2025_Teacher_ES - mean.csv'
OUTPUT_FILE = 'Backend_vs_Teacher_Comparison_ADVANCED.xlsx'
FUZZY_THRESHOLD = 80 # 模糊比對的相似度門檻 (0-100)

def normalize_string(text):
    """
    清理函式：移除所有標點符號、空格、和特殊字元，
    只保留中文、英文和數字，以便比對。
    """
    if not isinstance(text, str):
        return ""
    # 移除所有非中、英、數的字元
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    # 移除所有空白 (雖然上一步已處理，但多做一層保險)
    text = re.sub(r'\s+', '', text)
    return text.lower() # 轉為小寫

print("腳本開始執行...")

# 1. 讀取後台資料 (Backend)
try:
    df_backend = pd.read_csv(BACKEND_FILE)
    df_backend = df_backend[['Original_Column', 'Mean', 'N']].rename(
        columns={'Mean': 'Mean_Backend', 'N': 'N_Backend'}
    )
    # 建立正規化 Key
    df_backend['normalized_key'] = df_backend['Original_Column'].apply(normalize_string)
    print(f"成功讀取 {len(df_backend)} 筆後台資料。")
except Exception as e:
    print(f"讀取後台資料 {BACKEND_FILE} 失敗: {e}")
    exit()

# 2. 讀取學校資料 (Teacher)
try:
    df_teacher = pd.read_csv(TEACHER_FILE)
    df_teacher = df_teacher.dropna(subset=['學校平均值'])
    df_teacher = df_teacher[['問題', '學校平均值']].rename(
        columns={'問題': 'Original_Column', '學校平均值': 'Mean_Teacher'}
    )
    df_teacher['Mean_Teacher'] = pd.to_numeric(df_teacher['Mean_Teacher'], errors='coerce')
    df_teacher = df_teacher.dropna(subset=['Mean_Teacher'])
    # 建立正規化 Key
    df_teacher['normalized_key'] = df_teacher['Original_Column'].apply(normalize_string)
    print(f"成功讀取 {len(df_teacher)} 筆學校資料。")
except Exception as e:
    print(f"讀取學校資料 {TEACHER_FILE} 失敗: {e}")
    exit()

# --- 階段一：正規化匹配 (Normalized Match) ---
print("執行階段一：正規化匹配...")

# 使用 'normalized_key' 進行 inner join
df_normalized_match = pd.merge(
    df_backend, 
    df_teacher, 
    on='normalized_key', 
    how='inner',
    suffixes=('_Backend', '_Teacher') # 替原始題目欄位加上後綴
)

# 重新整理欄位
df_normalized_match = df_normalized_match[[
    'Original_Column_Backend', 
    'Mean_Backend', 
    'N_Backend', 
    'Original_Column_Teacher', 
    'Mean_Teacher',
    'normalized_key'
]]

print(f"階段一找到 {len(df_normalized_match)} 筆高信心匹配。")

# --- 找出階段一未匹配上的資料 ---
# 找出後台未匹配的 (Left Only)
backend_matched_keys = df_normalized_match['normalized_key']
df_backend_unmatched = df_backend[~df_backend['normalized_key'].isin(backend_matched_keys)].copy()

# 找出學校未匹配的 (Right Only)
teacher_matched_keys = df_normalized_match['normalized_key']
df_teacher_unmatched = df_teacher[~df_teacher['normalized_key'].isin(teacher_matched_keys)].copy()

print(f"階段一後，後台剩 {len(df_backend_unmatched)} 筆未匹配, 學校剩 {len(df_teacher_unmatched)} 筆未匹配。")

# --- 階段二：模糊匹配 (Fuzzy Match) ---
print(f"執行階段二：模糊匹配 (相似度 > {FUZZY_THRESHOLD}%) ...")

suggestions = []
# 為了避免重複建議，我們在找到匹配後就將其從列表中移除
teacher_fuzzy_matched_indices = set()

# 遍歷所有 "未匹配的後台題目"
for b_idx, b_row in df_backend_unmatched.iterrows():
    best_match = None
    best_score = FUZZY_THRESHOLD # 必須高於門檻

    # 遍歷所有 "未匹配的學校題目"
    for t_idx, t_row in df_teacher_unmatched.iterrows():
        # 如果這個學校題目已經被用過了，就跳過
        if t_idx in teacher_fuzzy_matched_indices:
            continue
            
        # 我們用 "原始題目" 來計算模糊分數，因為 "正規化" 後的 key 可能太短
        score = fuzz.ratio(b_row['Original_Column'], t_row['Original_Column'])
        
        if score > best_score:
            best_score = score
            best_match = t_row
            best_match_index = t_idx
            
    # 如果找到了高於門檻的最佳匹配
    if best_match is not None:
        suggestions.append({
            'Similarity_Score': best_score,
            'Backend_Question': b_row['Original_Column'],
            'Mean_Backend': b_row['Mean_Backend'],
            'Suggested_Teacher_Question': best_match['Original_Column'],
            'Mean_Teacher': best_match['Mean_Teacher'],
        })
        # 將這個學校題目標記為 "已使用"，避免 1-to-N 匹配
        teacher_fuzzy_matched_indices.add(best_match_index)

df_fuzzy_suggestions = pd.DataFrame(suggestions).sort_values(by='Similarity_Score', ascending=False)
print(f"階段二找到 {len(df_fuzzy_suggestions)} 筆模糊匹配建議。")


# --- 整理最終未匹配的清單 (Stages 1 & 2 都沒上的) ---

# 後台
backend_fuzzy_matched = set(df_fuzzy_suggestions['Backend_Question'])
df_backend_final_unmatched = df_backend_unmatched[
    ~df_backend_unmatched['Original_Column'].isin(backend_fuzzy_matched)
]

# 學校
teacher_fuzzy_matched = set(df_fuzzy_suggestions['Suggested_Teacher_Question'])
df_teacher_final_unmatched = df_teacher_unmatched[
    ~df_teacher_unmatched['Original_Column'].isin(teacher_fuzzy_matched)
]

print(f"最終剩餘：後台 {len(df_backend_final_unmatched)} 筆, 學校 {len(df_teacher_final_unmatched)} 筆完全無法匹配。")

# 4. 寫入 Excel
print(f"正在將結果寫入 Excel: {OUTPUT_FILE}")
try:
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_normalized_match.to_excel(writer, sheet_name='1_Normalized_Match (高信心)', index=False)
        df_fuzzy_suggestions.to_excel(writer, sheet_name=f'2_Fuzzy_Suggestions (>{FUZZY_THRESHOLD}%)', index=False)
        df_backend_final_unmatched.to_excel(writer, sheet_name='3_Backend_Only (最終未匹配)', index=False)
        df_teacher_final_unmatched.to_excel(writer, sheet_name='4_Teacher_Only (最終未匹配)', index=False)
        
    print("-" * 30)
    print(f"成功！進階比較報告已儲存為: {OUTPUT_FILE}")
    print("請先查看 '1_Normalized_Match'，再查看 '2_Fuzzy_Suggestions'。")
    print("-" * 30)

except Exception as e:
    print(f"寫入 Excel 失敗: {e}")
    print("請檢查您是否已安裝 'rapidfuzz' 且檔案未被開啟。")