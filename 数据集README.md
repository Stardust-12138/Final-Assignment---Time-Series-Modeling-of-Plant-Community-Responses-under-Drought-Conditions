---
pretty_name: "Great Plains 8-day Multisource NDVI–Climate Time Series (2000–2024)"
language:
  - en
  - zh
license: cc-by-4.0
task_categories:
  - time-series-forecasting
  - regression
size_categories:
  - 1K<n<10K
tags:
  - remote-sensing
  - NDVI
  - CHIRPS
  - ERA5-Land
  - drought
  - climate
  - time-series
  - united-states
---

# Great Plains 8-day Multisource NDVI–Climate Time Series (2000–2024)

## 1. 数据集概述 (Dataset Summary)

本数据集以 **美国南部大平原草原区** 为研究区域，范围约为：

- 经度：105°W–95°W  
- 纬度：32°N–40°N  

该区域为典型干旱敏感区，植被以草原为主，对降水异常和干旱事件高度敏感。

本数据集整合了 2000–2024 年间的多源观测，包含：

1. **MODIS Terra NDVI（8 日合成）**
2. **CHIRPS 日尺度降水**
3. **ERA5-Land 日尺度表层土壤含水量、2 m 气温、潜在蒸散**
4. 以 NDVI 时间步为主轴构建的 **8 日对齐多变量时间序列**（统一建模输入）

适合于：

- 干旱监测和评估  
- NDVI 与气候驱动因子的时滞/响应分析  
- 时序预测任务：ARIMA、多变量 LSTM、Encoder–Decoder 等  
- 多源气象–遥感数据融合研究  

---

## 2. 数据来源 (Data Sources)

### 2.1 NDVI（MODIS Terra MOD09A1）

- 产品：MODIS/061/MOD09A1（8 日地表反射率，500 m）
- 时间范围：2000-02-18 – 2024-12-26（8 日时间步，共约 1143 条记录）
- 波段：
  - RED: `sur_refl_b01`
  - NIR: `sur_refl_b02`
- 计算公式：

\[
\mathrm{NDVI} = \frac{NIR - RED}{NIR + RED}
\]

在 Google Earth Engine (GEE) 平台上，对 MOD09A1 进行质量控制与云/雪掩膜后，计算研究区范围内的 **区域平均 NDVI**，得到 8 日 NDVI 时间序列。

**相关文件：**

- `GreatPlains_MOD09A1_NDVI_8day_2000_2024.csv`  
  - 行数：1143  
  - 时间范围：2000-02-18 至 2024-12-26  

字段：

| 列名          | 类型    | 含义                                       |
|---------------|---------|--------------------------------------------|
| `system:index`| string  | GEE 生成的影像索引（如 `2000_02_18`）      |
| `date`        | string  | 日期，格式 `YYYY-MM-DD`                   |
| `ndvi`        | float   | 研究区内区域平均 NDVI                     |
| `.geo`        | string  | GEE 导出时附带的几何信息（MultiPoint，空）|

---

### 2.2 日降水（CHIRPS）

- 数据集：UCSB-CHG/CHIRPS/DAILY
- 空间分辨率：0.05°
- 时间分辨率：日
- 时间范围：2000-01-01 – 2024-12-30（共 9131 天）
- 处理方式：在 GEE 中对研究区范围进行区域平均，得到日平均降水（单位：mm/day）

**相关文件：**

- `GreatPlains_CHIRPS_DailyPrecip_2000_2024.csv`  
  - 行数：9131  
  - 时间范围：2000-01-01 至 2024-12-30  

字段：

| 列名           | 类型    | 含义                                       |
|----------------|---------|--------------------------------------------|
| `system:index` | int     | GEE 生成的索引（如 `20000101`）            |
| `date`         | string  | 日期，`YYYY-MM-DD`                        |
| `precip_daily` | float   | 研究区日平均降水量（mm/day）              |
| `.geo`         | string  | GEE 附带几何信息（MultiPoint，空）        |

---

### 2.3 土壤湿度、气温与潜在蒸散（ERA5-Land）

- 数据集：ECMWF/ERA5_LAND/DAILY_AGGR
- 时间分辨率：日
- 时间范围：2000-01-01 – 2024-12-30（共 9131 天）
- 空间处理：在 GEE 中对研究区进行区域平均

原始变量：

- `volumetric_soil_water_layer_1` → 本文件列名为 `soil_moisture_daily`  
  - 表层土壤体积含水量（0–7 cm），单位：m³/m³  
- `temperature_2m` → 本文件列名为 `temp2m_daily_K`  
  - 日平均 2 m 气温，单位：K  
- `potential_evaporation_sum` → 本文件列名为 `pet_daily_m`  
  - 日累计潜在蒸散量，单位：m/day，且为负值（表示向上的蒸发通量）

**相关文件：**

- `GreatPlains_ERA5L_SoilTempPET_Daily_2000_2024.csv`

字段：

| 列名                 | 类型    | 含义                                                |
|----------------------|---------|-----------------------------------------------------|
| `system:index`       | int     | GEE 生成索引                                        |
| `date`               | string  | 日期，`YYYY-MM-DD`                                 |
| `pet_daily_m`        | float   | 日累计潜在蒸散量（m/day，负值）                    |
| `soil_moisture_daily`| float   | 表层土壤体积含水量（m³/m³）                         |
| `temp2m_daily_K`     | float   | 2 m 日平均气温（K）                                 |
| `.geo`               | string  | 几何信息                                            |

在后续构建 8 日对齐数据时，潜在蒸散将通过 **取相反数并乘以 1000** 转换为 mm/day；气温将通过减去 273.15 转换为 ℃。

---

## 3. 数据预处理与 8 日合成 (Preprocessing & 8-day Aggregation)

为实现多源数据的统一建模，本研究进行如下预处理步骤（主要在本地 Python 环境中完成）：

### 3.1 时间格式与排序

- 使用 pandas 读入所有 CSV
- 将 `date` 字段转换为 `datetime` 类型，并以上午 00:00 作为时间索引
- 按日期升序排序

### 3.2 单位转换（适用于 ERA5-Land 日数据）

- 气温：
  - `temp2m_daily_K` → 摄氏度：  
    \[
    T\_{\mathrm{C}} = T\_{\mathrm{K}} - 273.15
    \]
- 潜在蒸散：
  - `pet_daily_m` 为 m/day 且为负值  
  - 转换为 mm/day 且为正：
    \[
    \mathrm{PET\_{mm/day}} = -\mathrm{pet\_daily\_m} \times 1000
    \]

### 3.3 按 NDVI 时间步聚合日尺度数据

- NDVI 8 日产品的时间步被视为 **主时间轴**
- 对于每一个 NDVI 时间点 \(t\)，在日尺度序列中取以 \(t\) 为中心的 8 日时间窗 \([t-4, t+3]\)：
  - **降水 (CHIRPS)**：求 8 日累积降水量（mm/8 days）
  - **潜在蒸散 (ERA5-Land)**：求 8 日累积潜在蒸散量（mm/8 days）
  - **土壤湿度**：求 8 日平均表层土壤含水量（m³/m³）
  - **气温**：求 8 日平均气温（℃）

### 3.4 缺失值处理

- CHIRPS 与 ERA5-Land 在该区域覆盖较完整，缺失值极少
- 对个别 NDVI 或气象变量缺测时间点：
  - 使用基于时间的 **线性插值** 修复，最多连续插补 2 个时间步
  - 对仍无法插补的少数起始或末尾时间点，直接删除对应样本

### 3.5 最终 8 日多变量数据集

由上述步骤得到的 8 日对齐多变量时间序列存入：

- `GreatPlains_8day_merged.csv`  
  - 行数：1143  
  - 时间范围：2000-02-18 至 2024-12-26  

字段：

| 列名                    | 类型    | 含义                                                      |
|-------------------------|---------|-----------------------------------------------------------|
| `date`                  | string  | NDVI 时间点日期（8 日间隔）                              |
| `ndvi`                  | float   | 区域平均 NDVI                                             |
| `precip_8d_sum_mm`      | float   | 以 NDVI 时间点为中心窗口计算的 8 日累计降水（mm/8 days） |
| `soil_moisture_8d_mean` | float   | 8 日平均表层土壤体积含水量（m³/m³）                       |
| `temp_8d_mean_C`        | float   | 8 日平均 2 m 气温（℃）                                    |
| `pet_8d_sum_mm`         | float   | 8 日累计潜在蒸散量（mm/8 days）                           |

该文件可直接作为 **统一建模输入** 用于 ARIMA、多变量 LSTM、Encoder–Decoder 等时序预测/回归任务。

---

## 4. 文件结构 (Files and Structure)

推荐仓库中文件结构如下：

```text
GreatPlains-Multisource-2000-2024/
├── GreatPlains_8day_merged.csv
├── GreatPlains_MOD09A1_NDVI_8day_2000_2024.csv
├── GreatPlains_CHIRPS_DailyPrecip_2000_2024.csv
├── GreatPlains_ERA5L_SoilTempPET_Daily_2000_2024.csv
└── README.md  # 本文档
