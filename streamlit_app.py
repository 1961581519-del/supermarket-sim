import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# ==========================================
# 1. 页面与全局参数初始化
# ==========================================
st.set_page_config(page_title="永辉超市库存动态仿真沙盘", layout="wide")
st.title("🛒 永辉超市生鲜区库存动态仿真沙盘")
st.markdown("通过蒙特卡洛仿真，测试不同静态补货策略在随机需求与极端波动下的真实运营表现。")

# 固定基础参数 (来自 Pre2 校准数据)
MEAN_DEMAND = 1200       # 日均需求
STD_DEMAND = 360         # 日需求标准差
LEAD_TIME = 1            # 提前期(天)
UNIT_MARGIN = 12.5       # 单件毛利
STOCKOUT_PENALTY = 6.82  # 单位缺货损失
DELIVERY_COST = 30       # 单次物流费用
CAPACITY_LIMIT = 800     # 库容物理极限
WARNING_LIMIT = 683      # 损耗预警线

# ==========================================
# 2. 侧边栏：交互式控制面板
# ==========================================
st.sidebar.header("⚙️ 运营决策控制台")

target_sl = st.sidebar.slider(
    "目标服务水平 (SL)",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
    help="决定安全库存的核心指标"
)

freq_options = {"2次/天": 2, "3次/天": 3}
freq_choice = st.sidebar.radio("补货频率 (f)", list(freq_options.keys()), index=1)
freq = freq_options[freq_choice]

extreme_weather = st.sidebar.checkbox(
    "🌪️ 引入周末极端波动",
    value=True,
    help="勾选后，每周六/日需求均值飙升至1500，标准差放大至500"
)

# ==========================================
# 3. 核心计算：根据服务水平反推安全库存
# ==========================================
# 补货周期 R = 1 / 频率
R = 1.0 / freq
# 总保护期 T = L + R
T = LEAD_TIME + R
# 保护期标准差
std_T = STD_DEMAND * np.sqrt(T)
# 计算 Z 值
z_value = norm.ppf(target_sl)
# 动态计算安全库存
calculated_ss = z_value * std_T

st.sidebar.markdown("---")
st.sidebar.metric("理论计算安全库存 (件)", f"{int(calculated_ss)}")
if calculated_ss > CAPACITY_LIMIT:
    st.sidebar.error("⚠️ 警告：理论安全库存已突破库容极限！")

# ==========================================
# 4. 蒙特卡洛仿真引擎
# ==========================================
np.random.seed(42)  # 固定种子保证演示可复现
days = 365
periods_per_day = freq
total_periods = days * periods_per_day

# 4.1 生成 365 天随机需求序列
demand_series = []
for day in range(1, days + 1):
    if extreme_weather and (day % 7 in [6, 0]):
        demand = max(0, np.random.normal(1500, 500))
    else:
        demand = max(0, np.random.normal(MEAN_DEMAND, STD_DEMAND))
    demand_series.append(demand)

# 将日需求平摊到各个补货周期
period_demands = []
for d in demand_series:
    for _ in range(periods_per_day):
        period_demands.append(d / periods_per_day)

# 4.2 初始化仿真状态
order_up_to_S = (MEAN_DEMAND / periods_per_day) + (calculated_ss / periods_per_day)
inventory = order_up_to_S
in_transit_orders = [0] * periods_per_day  # 长度代表在途延迟时间

total_sales = 0
total_stockouts = 0
total_holding_cost = 0
days_over_capacity = 0
inventory_history = []

# 4.3 执行时间步循环
for t in range(total_periods):
    # 期初到货
    arrival = in_transit_orders.pop(0)
    inventory += arrival

    # 期中消耗
    current_demand = period_demands[t]
    if inventory >= current_demand:
        sales = current_demand
        inventory -= sales
    else:
        sales = inventory
        total_stockouts += (current_demand - inventory)
        inventory = 0
    total_sales += sales

    # 期末核算持有成本 (非线性阶梯)
    if inventory <= 683:
        hc_rate = 0.25
    elif inventory <= 800:
        hc_rate = 0.30
    else:
        hc_rate = 0.35
        days_over_capacity += (1 / periods_per_day)

    total_holding_cost += (inventory * hc_rate / (365 * periods_per_day))

    # 发出新订单
    order_qty = max(0, order_up_to_S - inventory)
    in_transit_orders.append(order_qty)

    # 记录每日最终库存用于绘图
    if (t + 1) % periods_per_day == 0:
        inventory_history.append(inventory)

# ==========================================
# 5. KPI 计算与顶栏展示
# ==========================================
revenue = total_sales * UNIT_MARGIN
stockout_cost = total_stockouts * STOCKOUT_PENALTY
logistics_cost = days * freq * DELIVERY_COST
net_profit = revenue - total_holding_cost - stockout_cost - logistics_cost
actual_stockout_rate = total_stockouts / sum(demand_series)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 年度总利润 (元)", f"¥ {net_profit:,.0f}")
col2.metric("📉 实际系统缺货率", f"{actual_stockout_rate * 100:.2f}%")
col3.metric("📦 阶梯持有成本 (元)", f"¥ {total_holding_cost:,.0f}")
col4.metric("🚨 爆仓天数 (>800件)", f"{days_over_capacity:.0f} 天")

# ==========================================
# 6. 数据可视化：动态折线图 (Plotly)
# ==========================================
st.markdown("### 📊 365天期末库存动态波动轨迹")

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=inventory_history,
    mode='lines',
    name='期末库存水平',
    line=dict(color='#1f77b4', width=2)
))

# 添加预警线和极限线
fig.add_hline(y=CAPACITY_LIMIT, line_dash="dash", line_color="red",
              annotation_text="库容极限 (800)", annotation_position="top right")
fig.add_hline(y=WARNING_LIMIT, line_dash="dot", line_color="orange",
              annotation_text="损耗预警线 (683)", annotation_position="bottom right")

fig.update_layout(
    xaxis_title="天数 (Days)",
    yaxis_title="库存数量 (件)",
    template="plotly_white",
    height=400,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# 结论提示框
st.info("💡 **系统洞察**：尝试将目标服务水平拉至 **98%**，您会发现理论安全库存突破物理极限，爆仓天数急剧上升，导致总利润由于高昂的惩罚性持有成本而断崖式下跌。这证明了在不扩容的前提下，盲目追求极高服务水平是不可行的。")
