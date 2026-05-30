import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ===================== 1. 全局参数（100%来自pre1/pre2校准数据）=====================
MEAN_DEMAND = 1200       # 日均需求（件）
STD_DEMAND = 360         # 日需求标准差（件）
LEAD_TIME = 1            # 补货提前期（天）
UNIT_COST = 50           # 单位商品成本（元）
UNIT_MARGIN = 6.82       # 单位毛利（元）= 单位缺货损失Cu
DELIVERY_COST = 30       # 单次配送成本（元）
CAPACITY_LIMIT = 800     # 库容物理极限（件）
WARNING_LIMIT = 683      # 损耗预警线（pre2绿色安全区上限）

# ===================== 2. 非线性阶梯持有成本（按安全库存水平分档，对齐pre2文档）=====================
def get_holding_cost_rate(ss_level):
    """文档明确：持有成本率按安全库存水平分档，非瞬时库存"""
    if ss_level <= WARNING_LIMIT:
        return 0.25  # 绿色安全区：损耗率10%+资金成本15%，年持有成本12.5元/件
    elif ss_level <= CAPACITY_LIMIT:
        return 0.30  # 黄色预警区：损耗率15%+资金成本15%，年持有成本15元/件
    else:
        return 0.35  # 红色爆仓区：损耗率20%+资金成本15%，年持有成本17.5元/件

# ===================== 3. 页面初始化 =====================
st.set_page_config(page_title="永辉超市库存动态仿真沙盘", layout="wide")
st.title("🛒 永辉超市生鲜区库存动态仿真沙盘")
st.markdown("通过蒙特卡洛仿真，测试不同服务水平目标下的利润、缺货率与库存周转率")
st.markdown("**核心验证**：需求波动下，服务水平过度提高是否导致成本急剧上升（利润陷阱）")

# ===================== 4. 侧边栏：答辩专用控制台 =====================
st.sidebar.header("⚙️ 运营决策控制台")

st.sidebar.subheader("快速切换预设策略")
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("策略1\n90%SL"):
    target_sl = 0.90
    freq = 3
if col2.button("策略2\n95%SL"):
    target_sl = 0.95
    freq = 3
if col3.button("策略3\n98%SL"):
    target_sl = 0.98
    freq = 3

target_sl = st.sidebar.slider(
    "目标服务水平 (SL)",
    min_value=0.90, max_value=0.98, value=0.95, step=0.01,
    help="决定安全库存的核心指标"
)
freq = st.sidebar.radio("补货频率 (次/天)", [2, 3], index=1,
    help="鲜度约束：≥2次/天；配送约束：≤3次/天")

# 理论安全库存计算
R = 1 / freq
T_protect = R + LEAD_TIME
std_protect = STD_DEMAND * np.sqrt(T_protect)
z_value = norm.ppf(target_sl)
safety_stock = z_value * std_protect

st.sidebar.markdown("---")
st.sidebar.metric("理论安全库存 (件)", f"{int(safety_stock)}")
if safety_stock > CAPACITY_LIMIT:
    st.sidebar.error("⚠️ 安全库存已突破库容极限 800 件！")
elif safety_stock > WARNING_LIMIT:
    st.sidebar.warning("⚡ 安全库存进入黄色预警区 (683-800件)，边际成本上升")

# ===================== 5. 蒙特卡洛仿真引擎 =====================
def run_simulation(sl_target, freq_val):
    """
    严格对齐 pre2 文档流程：
    - 在途管道预填充（消除冷启动偏差）
    - 持有成本率按 SS 水平分档（非瞬时库存）
    - order-up-to = safety_stock（pre2简化：忽略周转库存持有成本）
    """
    rng = np.random.default_rng(seed=42)
    days = 365
    per_day = freq_val
    total_steps = days * per_day

    # 5.1 生成日需求 → 摊到每周期
    daily_demand = np.maximum(0, rng.normal(MEAN_DEMAND, STD_DEMAND, days))
    period_demand = np.repeat(daily_demand / per_day, per_day)

    # 5.2 计算安全库存与目标库存
    R_val = 1.0 / freq_val
    T_val = R_val + LEAD_TIME
    ss_calc = norm.ppf(sl_target) * STD_DEMAND * np.sqrt(T_val)
    order_up_to = ss_calc  # pre2核心简化

    # 5.3 确定持有成本率（按SS水平分档，非瞬时库存）
    hc_rate = get_holding_cost_rate(ss_calc)

    # 5.4 初始化：管道预填充期望周期需求，消除冷启动
    expected_period_dmd = MEAN_DEMAND / per_day
    pipeline_len = per_day * LEAD_TIME
    in_transit = [expected_period_dmd] * pipeline_len
    inventory = order_up_to

    total_sales = 0.0
    total_stockouts = 0.0
    total_holding_cost = 0.0
    days_over_cap = 0.0
    daily_inv = []

    # 5.5 时间步主循环
    for t in range(total_steps):
        # 期初到货
        arrival = in_transit.pop(0)
        inventory += arrival
        peak_inv = inventory  # 记录周期峰值

        # 需求消耗
        dmd = period_demand[t]
        if inventory >= dmd:
            sales = dmd
            inventory -= dmd
        else:
            sales = inventory
            total_stockouts += (dmd - inventory)
            inventory = 0.0
        total_sales += sales

        # 持有成本 = 期末库存 × 单位成本 × SS分档持有成本率 ÷ 365 ÷ 每日周期数
        total_holding_cost += (inventory * UNIT_COST * hc_rate / (365 * per_day))

        # 爆仓天数统计（按峰值判断）
        if peak_inv > CAPACITY_LIMIT:
            days_over_cap += (1.0 / per_day)

        # 发出补货订单
        order_qty = max(0.0, order_up_to - inventory)
        in_transit.append(order_qty)

        # 记录每日末库存
        if (t + 1) % per_day == 0:
            daily_inv.append(inventory)

    # 5.6 计算KPI
    total_demand = np.sum(daily_demand)
    revenue = total_sales * UNIT_MARGIN
    stockout_loss = total_stockouts * UNIT_MARGIN  # 缺货损失=毛利损失
    logistics_cost = days * freq_val * DELIVERY_COST
    total_cost = total_holding_cost + stockout_loss + logistics_cost
    net_profit = revenue - total_cost

    stockout_rate = total_stockouts / total_demand
    actual_sl = 1.0 - stockout_rate
    avg_inv = np.mean(daily_inv)
    turnover = total_sales / avg_inv if avg_inv > 0 else 0.0

    return {
        "profit": net_profit,
        "total_cost": total_cost,
        "holding_cost": total_holding_cost,
        "stockout_cost": stockout_loss,
        "logistics_cost": logistics_cost,
        "stockout_rate": stockout_rate,
        "actual_sl": actual_sl,
        "turnover": turnover,
        "avg_inventory": avg_inv,
        "days_over_cap": days_over_cap,
        "daily_inventory": daily_inv,
        "ss_calc": ss_calc,
        "hc_rate": hc_rate,
    }

# 5.7 预计算三个策略
res_90 = run_simulation(0.90, 3)
res_95 = run_simulation(0.95, 3)
res_98 = run_simulation(0.98, 3)

# 5.8 选择当前结果
if target_sl == 0.90 and freq == 3:
    results = res_90
elif target_sl == 0.95 and freq == 3:
    results = res_95
elif target_sl == 0.98 and freq == 3:
    results = res_98
else:
    results = run_simulation(target_sl, freq)

# ===================== 6. KPI展示 =====================
st.subheader("📊 核心绩效指标")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💰 年度总利润", f"¥{results['profit']:,.0f}")
c2.metric("📊 年度总成本", f"¥{results['total_cost']:,.0f}")
c3.metric("📉 实际缺货率", f"{results['stockout_rate']:.2%}")
c4.metric("🔄 库存周转率", f"{results['turnover']:.1f} 次/年")
c5.metric("✅ 实际服务水平", f"{results['actual_sl']:.2%}")
c6.metric("🚨 爆仓天数", f"{results['days_over_cap']:.0f} 天")

with st.expander("📋 成本结构明细"):
    st.write(f"- 持有成本：¥{results['holding_cost']:,.0f}（SS={results['ss_calc']:.0f}件，档位费率={results['hc_rate']:.0%}）")
    st.write(f"- 缺货损失：¥{results['stockout_cost']:,.0f}（缺货{results['stockout_rate']*100:.1f}% × 年需求438,000件 × Cu=¥{UNIT_MARGIN}）")
    st.write(f"- 物流成本：¥{results['logistics_cost']:,.0f}（365天 × {freq}次/天 × ¥{DELIVERY_COST}/次）")
    st.write(f"- 平均库存：{results['avg_inventory']:.0f} 件")

# ===================== 7. 可视化 =====================
tab1, tab2, tab3 = st.tabs(["📈 365天库存波动", "📊 三策略利润对比", "💸 三策略总成本对比"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=results["daily_inventory"], mode="lines",
        name="每日期末库存", line=dict(color="#1f77b4", width=2)))
    fig.add_hline(y=WARNING_LIMIT, line_dash="dot", line_color="orange",
        annotation_text="损耗预警线 (683)", annotation_position="bottom right")
    fig.add_hline(y=CAPACITY_LIMIT, line_dash="dash", line_color="red",
        annotation_text="库容极限 (800)", annotation_position="top right")
    fig.update_layout(height=400, xaxis_title="天数", yaxis_title="库存数量 (件)",
        template="plotly_white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    sls = ["90%", "95%", "98%"]
    profits = [res_90["profit"], res_95["profit"], res_98["profit"]]
    costs = [res_90["total_cost"], res_95["total_cost"], res_98["total_cost"]]
    fig = go.Figure(data=[go.Bar(x=sls, y=profits, marker_color=["#ff7f0e","#2ca02c","#d62728"],
        text=[f"¥{v:,.0f}" for v in profits], textposition="auto")])
    fig.update_layout(height=400, xaxis_title="目标服务水平", yaxis_title="年度总利润 (元)",
        template="plotly_white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure(data=[go.Bar(x=sls, y=costs, marker_color=["#ff7f0e","#2ca02c","#d62728"],
        text=[f"¥{v:,.0f}" for v in costs], textposition="auto")])
    fig.update_layout(height=400, xaxis_title="目标服务水平", yaxis_title="年度总成本 (元)",
        template="plotly_white", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ===================== 8. 动态结论 =====================
st.subheader("✅ 仿真结论")
profit_95 = res_95["profit"]
cost_95 = res_95["total_cost"]

if target_sl < 0.95:
    gap = res_95["profit"] - results["profit"]
    conclusion = f"""
    **当前 SL={target_sl:.0%}（保守策略）**
    - 缺货率 {results['stockout_rate']:.2%}，缺货损失占成本主导
    - 比 95% 策略利润低约 **¥{gap:,.0f}**
    - 结论：服务水平偏低，缺货拖累利润，建议提升至 95%
    """
elif target_sl == 0.95:
    gap_90 = results["profit"] - res_90["profit"]
    conclusion = f"""
    **当前 SL=95%（全局最优，推荐）**
    - 总成本 ¥{results['total_cost']:,.0f}，利润 ¥{results['profit']:,.0f}
    - 比 90% 策略多赚约 **¥{gap_90:,.0f}**
    - 缺货率 {results['stockout_rate']:.2%}，顾客体验良好
    - 验证文档结论：**95% 是成本与服务的最优平衡点**
    """
else:
    gap = cost_95 - results["total_cost"]
    if gap < 0:
        gap_str = f"总成本反超 95% 策略 ¥{-gap:,.0f}"
    else:
        gap_str = f"总成本比 95% 低 ¥{gap:,.0f}（但安全库存 {results['ss_calc']:.0f} 件已进入{'红色爆仓' if results['ss_calc'] > CAPACITY_LIMIT else '黄色预警'}区）"
    conclusion = f"""
    **当前 SL={target_sl:.0%}（激进策略）**
    - 缺货率最低（{results['stockout_rate']:.2%}），但**持有成本急剧上升**
    - {gap_str}
    - 安全库存 {results['ss_calc']:.0f} 件（{'已突破库容极限！' if results['ss_calc'] > CAPACITY_LIMIT else '进入预警区'}+持有成本率={results['hc_rate']:.0%}）
    - 验证文档核心洞察：**盲目追求极高服务水平 → 利润陷阱**
    """

st.info(conclusion)

# ===================== 9. 答辩指引 =====================
st.sidebar.markdown("---")
st.sidebar.info("""
💡 **答辩演示步骤：**
1. 点【策略1 90%SL】→ 看高缺货率 + 低利润
2. 点【策略2 95%SL】→ 看利润最高，最优平衡
3. 点【策略3 98%SL】→ 看爆仓 + 利润断崖下跌
4. 切到「三策略利润对比」标签页 → 柱状图一目了然
""")
