import os
import tempfile

# ====================== 1. 核心：自动设置临时路径 ======================
def set_custom_temp_dir():
    """自动设置固定的临时路径（无交互输入）"""
    custom_temp_path = "D:/temp1"
    try:
        if not os.path.exists(custom_temp_path):
            os.makedirs(custom_temp_path, exist_ok=True)
            print(f"已创建自定义临时文件夹: {custom_temp_path}")
        
        # 验证路径可写性
        test_file = os.path.join(custom_temp_path, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        
        # 修改环境变量（仅当前程序有效）
        os.environ["TMP"] = custom_temp_path
        os.environ["TEMP"] = custom_temp_path
        
        # 验证临时路径生效
        temp_file = tempfile.NamedTemporaryFile(dir=custom_temp_path, delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        os.remove(temp_file_path)
        
        print(f"临时路径已自动设置为: {custom_temp_path}")
        return True
    except PermissionError:
        print(f"错误：无权限访问 {custom_temp_path}，请检查路径权限")
        return False
    except Exception as e:
        print(f"设置临时路径失败: {str(e)}")
        return False

# 先设置临时路径（必须在导入scipy/sklearn等库之前执行）
set_custom_temp_dir()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math
import io

# 设置页面配置
st.set_page_config(layout="wide", page_title="大平原植被与干旱监测仪表盘")

# ==========================================
# 1. 数据加载与预处理
# ==========================================

@st.cache_data
def load_data():
    # 这里假设文件名为 GreatPlains_8day_merged.csv 并且在同一目录下
    # 实际运行时，请确保文件存在
    file_path = 'D:/12138/大数据系统原理与应用/期末作业/数据/GreatPlains_8day_merged.csv'
    
    try:
        # 尝试读取本地文件
        df = pd.read_csv(file_path, sep='\t') # 你的数据看起来像是制表符或空格分隔
        if df.shape[1] == 1: # 如果分隔符不对，尝试逗号
             df = pd.read_csv(file_path, sep=',')
    except FileNotFoundError:
        st.error(f"未找到文件: {file_path}。请确保CSV文件在脚本运行目录下。")
        return None

    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 列名重命名（为了显示更友好）
    df = df.rename(columns={
        'ndvi': 'NDVI (植被指数)',
        'precip_8d_sum_mm': '8日累计降水 (mm)',
        'soil_moisture_8d_mean': '8日平均土壤湿度 (m³/m³)',
        'temp_8d_mean_C': '8日平均气温 (°C)',
        'pet_8d_sum_mm': '8日累计潜在蒸散 (mm)'
    })
    
    return df

# ==========================================
# 2. 模型定义 (Transformer for Time Series)
# ==========================================

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src):
        # src shape: [batch_size, seq_len, input_dim]
        src = self.embedding(src) # [batch, seq, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :] # 取最后一个时间步的输出
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 数据准备函数
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # 预测下一步
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 3. 页面布局与逻辑
# ==========================================

def main():
    st.title("🌿 美国南部大平原植被干旱监测与多源驱动预测系统")
    st.markdown("""
    > **数据来源说明**：
    > 本系统基于 **MODIS (NDVI)**, **CHIRPS (降水)**, **ERA5-Land (土壤湿度、气温、蒸散)** 构建了 2000-2024 年的多源时间序列数据集。
    > 数据已统一处理为 8 天时间分辨率，并针对研究区域（105°W–95°W，32°N–40°N）进行了区域聚合。
    """)

    df = load_data()
    if df is None:
        return

    # 侧边栏控制
    st.sidebar.header("全局设置")
    
    # 特征选择
    feature_cols = ['NDVI (植被指数)', '8日累计降水 (mm)', '8日平均土壤湿度 (m³/m³)', '8日平均气温 (°C)', '8日累计潜在蒸散 (mm)']
    selected_feature = st.sidebar.selectbox("选择主要观测特征", feature_cols, index=0)
    
    # TAB页切换
    tab1, tab2, tab3 = st.tabs(["📊 数据概览与统计", "🗺️ 区域动态地图", "🤖 Transformer 未来预测"])

    # --- TAB 1: 数据概览 ---
    with tab1:
        st.header("历史数据时间序列分析")
        
        # 绘制主特征曲线
        fig_main = px.line(df, x='date', y=selected_feature, title=f"{selected_feature} 2000-2024 变化趋势")
        fig_main.update_layout(hovermode="x unified")
        st.plotly_chart(fig_main, use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # 多变量对比图
            st.subheader("多变量协同变化")
            compare_features = st.multiselect("选择要对比的特征", feature_cols, default=['NDVI (植被指数)', '8日累计降水 (mm)'])
            if compare_features:
                # 标准化以便在同一轴上显示趋势
                df_norm = df.copy()
                for col in compare_features:
                    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                
                fig_compare = px.line(df_norm, x='date', y=compare_features, title="标准化趋势对比 (归一化 0-1)")
                st.plotly_chart(fig_compare, use_container_width=True)
        
        with col2:
            # 相关性热力图
            st.subheader("特征相关性分析")
            corr = df[feature_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Pearson 相关系数")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("提示：NDVI 通常与降水和土壤湿度呈正相关，与潜在蒸散可能呈复杂关系（视水分限制条件而定）。")

    # --- TAB 2: 地图可视化 ---
    with tab2:
        st.header("研究区域动态监测")
        st.markdown("通过拖动下方进度条，查看选定特征在**大平原研究区 (105°W–95°W, 32°N–40°N)** 内随时间的变化情况。")
        st.markdown("*注：由于数据已进行区域平均处理，地图颜色代表整个区域在该时间点的平均状态。*")

        # 时间滑块
        min_date = df['date'].min().to_pydatetime()
        max_date = df['date'].max().to_pydatetime()
        
        selected_date = st.slider(
            "选择时间",
            min_value=min_date,
            max_value=max_date,
            value=min_date,
            format="YYYY-MM-DD"
        )
        
        # 找到最接近的数据
        nearest_row = df.iloc[(df['date'] - selected_date).abs().argsort()[:1]]
        current_val = nearest_row[selected_feature].values[0]
        current_date_str = nearest_row['date'].dt.strftime('%Y-%m-%d').values[0]

        # 计算颜色的归一化值
        min_val = df[selected_feature].min()
        max_val = df[selected_feature].max()
        
        # 定义颜色映射逻辑
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        # 根据特征选择不同的色带
        if 'NDVI' in selected_feature:
            cmap = cm.get_cmap('RdYlGn') # 红-黄-绿 (植被)
        elif '降水' in selected_feature or '土壤' in selected_feature:
            cmap = cm.get_cmap('Blues') # 蓝 (水)
        elif '气温' in selected_feature or '蒸散' in selected_feature:
            cmap = cm.get_cmap('YlOrRd') # 黄-红 (热)
        else:
            cmap = cm.get_cmap('viridis')

        rgba = cmap(norm(current_val))
        hex_color = mcolors.to_hex(rgba)

        # 创建地图
        m = folium.Map(location=[36, -100], zoom_start=6, tiles="CartoDB positron")

        # 绘制矩形区域
        bounds = [[32, -105], [40, -95]] # Lat range, Lon range
        
        folium.Rectangle(
            bounds=bounds,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup=folium.Popup(f"<b>日期:</b> {current_date_str}<br><b>{selected_feature}:</b> {current_val:.4f}", max_width=300),
            tooltip="点击查看详情"
        ).add_to(m)

        # 显示地图
        col_map, col_info = st.columns([3, 1])
        with col_map:
            st_folium(m, width=800, height=500)
        
        with col_info:
            st.metric(label=f"当前日期: {current_date_str}", value=f"{current_val:.4f}")
            st.write(f"**{selected_feature}** 在全时段内的范围:")
            st.write(f"最小值: {min_val:.4f}")
            st.write(f"最大值: {max_val:.4f}")
            st.progress((current_val - min_val) / (max_val - min_val))

    # --- TAB 3: 预测建模 ---
    with tab3:
        st.header("多源驱动未来预测 (Transformer)")
        st.markdown("""
        本模块利用 **Transformer** 深度学习模型，学习 NDVI 与气象因子（降水、气温等）的历史时序关系。
        模型训练完成后，将执行**多步滚动预测**，生成超出当前数据集时间范围的未来趋势。
        """)

        col_params, col_train = st.columns([1, 3])
        
        with col_params:
            st.subheader("模型交互参数")
            forecast_steps = st.number_input("未来预测步数 (每步8天)", min_value=4, max_value=46, value=12, help="预测未来多少个8天周期")
            seq_length = st.slider("回顾窗口大小 (Seq Length)", 4, 24, 12, help="模型利用过去多少个时间步来预测下一步")
            epochs = st.slider("训练轮次 (Epochs)", 10, 200, 50)
            hidden_dim = st.selectbox("Transformer 隐藏层维度", [32, 64, 128], index=1)
            
            start_train = st.button("🚀 开始训练并预测", type="primary")

        if start_train:
            with col_train:
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # 1. 数据准备
                status_text.text("正在准备张量数据...")
                
                # 使用所有特征进行预测
                data_values = df[feature_cols].values
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data_values)
                
                # 目标是预测 NDVI (feature_cols[0])
                # 这里的简单演示是：用过去N天的所有特征，预测下一天的所有特征（或者仅NDVI）
                # 为了支持多步滚动预测，我们训练模型预测所有特征，以便将预测值作为下一步的输入
                
                X, y = create_sequences(data_scaled, seq_length)
                
                # 转换为 Tensor
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(y) # 预测所有特征
                
                # 划分训练集测试集 (这里主要用全量数据训练以获得最好的未来预测能力，或者留一小部分验证)
                train_size = int(len(X) * 0.9)
                X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
                y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
                
                # 2. 模型初始化
                input_dim = len(feature_cols)
                output_dim = len(feature_cols) # 输出所有特征以支持滚动预测
                
                model = TimeSeriesTransformer(
                    input_dim=input_dim, 
                    d_model=hidden_dim, 
                    nhead=4, 
                    num_layers=2, 
                    output_dim=output_dim
                )
                
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                # 3. 训练循环
                model.train()
                train_losses = []
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    
                    if (epoch + 1) % 10 == 0:
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"训练中... Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

                status_text.text("训练完成！正在进行未来推演...")
                
                # 4. 未来多步预测 (Extrapolation)
                model.eval()
                future_predictions = []
                
                # 初始输入：数据的最后 seq_length 个时间步
                current_seq = torch.FloatTensor(data_scaled[-seq_length:]).unsqueeze(0) # [1, seq_len, input_dim]
                
                with torch.no_grad():
                    for _ in range(forecast_steps):
                        # 预测下一步
                        next_step = model(current_seq) # [1, output_dim]
                        
                        # 保存预测结果
                        future_predictions.append(next_step.numpy()[0])
                        
                        # 更新输入序列：移除第一个，加入预测出的这一个
                        # next_step shape is [1, feature_dim], need to reshape to match dim
                        next_step_reshaped = next_step.unsqueeze(1) # [1, 1, input_dim]
                        current_seq = torch.cat((current_seq[:, 1:, :], next_step_reshaped), dim=1)

                # 5. 反归一化与可视化
                future_pred_scaled = np.array(future_predictions)
                future_pred_original = scaler.inverse_transform(future_pred_scaled)
                
                # 构建未来时间轴
                last_date = df['date'].iloc[-1]
                future_dates = [last_date + pd.Timedelta(days=8 * (i + 1)) for i in range(forecast_steps)]
                
                # 创建预测结果 DataFrame
                pred_df = pd.DataFrame(future_pred_original, columns=feature_cols)
                pred_df['date'] = future_dates
                pred_df['type'] = '未来预测 (Forecast)'
                
                # 历史数据最后一段用于连接
                history_plot_df = df.tail(100).copy() # 只画最近100个点避免图太挤
                history_plot_df['type'] = '历史观测 (History)'
                
                # 合并数据用于绘图
                combined_df = pd.concat([history_plot_df, pred_df], ignore_index=True)
                
                # 绘制预测图
                st.subheader(f"未来 {forecast_steps*8} 天 {selected_feature} 趋势预测")
                
                fig_forecast = px.line(
                    combined_df, 
                    x='date', 
                    y=selected_feature, 
                    color='type',
                    color_discrete_map={'历史观测 (History)': 'gray', '未来预测 (Forecast)': 'red'},
                    title=f"Transformer 多步滚动预测结果: {selected_feature}"
                )
                # 添加预测区间的背景色
                fig_forecast.add_vrect(
                    x0=last_date, 
                    x1=future_dates[-1], 
                    fillcolor="red", 
                    opacity=0.1, 
                    layer="below", 
                    line_width=0,
                    annotation_text="预测区间"
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # 训练损失曲线
                with st.expander("查看模型训练损失 (Loss Curve)"):
                    st.line_chart(train_losses)
                    st.caption("MSE Loss 随 Epochs 下降情况，若曲线未收敛，请增加训练轮次。")

if __name__ == "__main__":
    main()
