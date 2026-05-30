import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. 核心参数与基础设定
# ==========================================
st.set_page_config(page_title="零售库存仿真仪表盘", layout="wide")
st.title("永辉超市生鲜区：库存与服务水平动态仿真沙盘")

# 侧边栏：交互输入控制
st.sidebar.header("仿真参数控制面板")
target_sl = st.sidebar.slider("目标服务水平 (SL)", 0.80, 0.99, 0.95, 0.01)
freq_options = {"2次/天": 2, "3次/天": 3}
freq_choice = st.sidebar.radio("补货频率 (f)", list(freq_options.keys()), index=1)
f = freq_options[freq_choice]

extreme_volatility = st.sidebar.checkbox("引入极端周末波动", value=True)

# 固定常量设定
DAYS = 365
L = 1  # 提前期 1天
MAX_CAPACITY = 800  # 物理库容上限
UNIT_MARGIN = 12.5  # 单件毛利预估
UNIT_STOCKOUT_PENALTY = 6.82
DELIVERY_COST = 30
BASE_MEAN = 1200
BASE_STD = 360

# ==========================================
# 2. 蒙特卡洛仿真引擎
# ==========================================
@st.cache_data
def run_simulation(sl, freq, extreme_vol):
    np.random.seed(42) # 保证每次调整参数时结果可重现对比

    # 动态计算不同频率下的安全库存和目标库存上限 (Order-up-to level)
    # R: 补货周期，T: 保护期 (R+L)
    R_days = 1 / freq
    T_days = R_days + L
    z_val = norm.ppf(sl)

    # 假设每日需求在各个补货周期内均匀分布
    cycle_mean = BASE_MEAN * R_days
    cycle_std = BASE_STD * np.sqrt(R_days)

    # 理论安全库存 (以天为维度折算)
    SS = z_val * BASE_STD * np.sqrt(T_days)
    order_up_to = BASE_MEAN * T_days + SS

    # 初始化记录数组
    inventory_levels = np.zeros(DAYS)
    stockout_units_arr = np.zeros(DAYS)
    holding_costs_arr = np.zeros(DAYS)
    daily_demand_arr = np.zeros(DAYS)

    current_inventory = order_up_to  # 初始库存满载
    pipeline_orders = np.zeros(L + 1) # 在途订单队列

    for day in range(DAYS):
        # 1. 生成当天随机需求
        is_weekend = (day % 7 == 5) or (day % 7 == 6)
        if extreme_vol and is_weekend:
            daily_demand = np.random.normal(1500, 500)
        else:
            daily_demand = np.random.normal(BASE_MEAN, BASE_STD)
        daily_demand = max(0, daily_demand)
        daily_demand_arr[day] = daily_demand

        # 2. 接收在途订单 (简单按天结算)
        arrived_order = pipeline_orders[0]
        pipeline_orders[:-1] = pipeline_orders[1:]
        pipeline_orders[-1] = 0
        current_inventory += arrived_order

        # 3. 满足需求与缺货统计
        if current_inventory >= daily_demand:
            current_inventory -= daily_demand
            stockout_units = 0
        else:
            stockout_units = daily_demand - current_inventory
            current_inventory = 0

        stockout_units_arr[day] = stockout_units
        inventory_levels[day] = current_inventory

        # 4. 阶梯非线性持有成本结算
        if current_inventory <= 683:
            holding_cost = current_inventory * (0.25 * 50 / 365)
        elif current_inventory <= 800:
            holding_cost = current_inventory * (0.30 * 50 / 365)
        else:
            holding_cost = current_inventory * (0.35 * 50 / 365)

        holding_costs_arr[day] = holding_cost

        # 5. 期末制定补货决策 (简单的基准库存策略)
        inventory_position = current_inventory + sum(pipeline_orders)
        order_qty = max(0, order_up_to - inventory_position)
        pipeline_orders[-1] = order_qty

    # 汇总KPI
    total_demand = np.sum(daily_demand_arr)
    total_stockout = np.sum(stockout_units_arr)
    actual_sl = 1 - (total_stockout / total_demand) if total_demand > 0 else 0

    total_sales = total_demand - total_stockout
    gross_profit = total_sales * UNIT_MARGIN
    total_stockout_loss = total_stockout * UNIT_STOCKOUT_PENALTY
    total_holding_cost = np.sum(holding_costs_arr)
    total_logistics_cost = DAYS * freq * DELIVERY_COST

    net_profit = gross_profit - total_stockout_loss - total_holding_cost - total_logistics_cost
    over_capacity_days = np.sum(inventory_levels > MAX_CAPACITY)
    warning_days = np.sum((inventory_levels > 683) & (inventory_levels <= 800))

    return {
        "inventory_levels": inventory_levels,
        "net_profit": net_profit,
        "actual_sl": actual_sl,
        "total_holding_cost": total_holding_cost,
        "over_capacity_days": over_capacity_days,
        "warning_days": warning_days
    }

# 运行仿真
results = run_simulation(target_sl, f, extreme_volatility)

# ==========================================
# 3. 仪表盘 UI 渲染与可视化
# ==========================================
# 顶部 KPI 指标卡片
col1, col2, col3, col4 = st.columns(4)
col1.metric("年度综合净利润 (元)", f"¥{results['net_profit']:,.0f}")
col2.metric("实际达成服务水平", f"{results['actual_sl']*100:.2f}%", delta=f"{results['actual_sl'] - target_sl:.2%} 偏差")
col3.metric("年总持有成本 (元)", f"¥{results['total_holding_cost']:,.0f}")
col4.metric("严重爆仓天数 (>800件)", f"{results['over_capacity_days']} 天", delta_color="inverse")

st.markdown("---")

# 中间核心图表：全年库存动态波动图
st.subheader("全年期末库存动态波动轨迹 (365天)")

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(results['inventory_levels'], color='#1f77b4', linewidth=1.5, alpha=0.8, label="每日期末库存")

# 添加物理极限与预警线
ax.axhline(y=800, color='red', linestyle='--', linewidth=2, label="物理库容极限 (800件) - 触发35%最高损耗")
ax.axhline(y=683, color='orange', linestyle=':', linewidth=2, label="运营预警线 (683件) - 触发30%较高损耗")

ax.fill_between(range(DAYS), 800, results['inventory_levels'], where=(results['inventory_levels'] > 800), color='red', alpha=0.3)
ax.fill_between(range(DAYS), 683, 800, where=((results['inventory_levels'] > 683) & (results['inventory_levels'] <= 800)), color='orange', alpha=0.2)

ax.set_xlabel("天数 (Day)", fontsize=12)
ax.set_ylabel("库存件数 (Units)", fontsize=12)
ax.set_title(f"库存波动轨迹仿真 | 目标SL: {target_sl*100:.1f}% | 频率: {f}次/天", fontsize=14)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# 底部诊断与决策建议
st.subheader("💡 仿真诊断报告")
if results['over_capacity_days'] > 20:
    st.error(f"**高风险警告：** 当前策略导致全年发生 **{results['over_capacity_days']}** 天爆仓。盲目追求过高的目标服务水平（{target_sl*100}%），在面对极端不确定性时，非线性惩罚成本已经严重吞噬利润。建议降低服务水平目标，或实施动态补货调控！")
elif results['warning_days'] > 50:
    st.warning(f"**运营承压：** 处于黄色预警区间（高损耗）的天数高达 **{results['warning_days']}** 天，虽然未严重爆仓，但门店运营缓冲极低。")
else:
    st.success("**系统健康：** 当前参数组合在应对需求波动时表现稳健，既保证了良好的顾客体验，又完美规避了爆仓引发的非线性损耗。")
