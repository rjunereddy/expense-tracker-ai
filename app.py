import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from generate_data import generate
from categorize import train, load_model, predict
from anomaly import flag_transaction_outliers, weekly_spend_alerts
from forecast import forecast_next_period, get_forecast_timeline
from nlp_query import answer_query
from auth import register_user, authenticate_user
from subscriptions import detect_subscriptions
from db import get_transactions, save_transactions, append_transaction

st.set_page_config(page_title="Expense Tracker AI Portal", page_icon="💸", layout="wide")


@st.cache_data
def get_data(username):
    df = get_transactions(username)
    return df


@st.cache_resource
def get_model(df, username):
    import categorize
    user_model_path = f"data/categorizer_{username}.joblib"
    categorize.MODEL_PATH = user_model_path
    if not os.path.exists(user_model_path):
        if not df.empty and len(df) > 10:
            train(df)
        else:
            # Fallback to the base model if user has no data yet
            categorize.MODEL_PATH = "data/categorizer.joblib"
            if not os.path.exists("data/categorizer.joblib"):
                # If absolute base model doesn't exist, we must generate a tiny bit of training data just for the ML engine
                from generate_data import generate
                train(generate(n_days=30))
    return load_model()


def main():
    # Load fonts & modern glassmorphic card styling via CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Metrics panel glassmorphic styling */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 20px;
    }
    
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        margin-top: 5px;
    }
    
    .metric-lbl {
        font-size: 14px;
        color: #a0aec0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Elegant gradient heading styling */
    .header-tag {
        background: linear-gradient(135deg, #7f00ff 0%, #e100ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Alert cards for anomaly flags */
    .alert-card {
        background: rgba(255, 61, 0, 0.08);
        border-left: 4px solid #ff3d00;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #f8f9fa;
        font-size: 15px;
    }

    /* Auth container styling */
    .auth-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        max-width: 500px;
        margin: 50px auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize Auth State
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.full_name = None

    # --- RENDER AUTHENTICATION VIEW ---
    if not st.session_state.authenticated:
        st.markdown("<h1 style='text-align: center;'>💸 <span class='header-tag'>Expense Tracker AI</span> Portal</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 40px;'>Empowering your finances with state-of-the-art anomaly checks, linear forecasting, and grounded NLP analysis.</p>", unsafe_allow_html=True)
        
        col_left, col_mid, col_right = st.columns([1, 1.5, 1])
        with col_mid:
            st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
            tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Create Account"])
            
            with tab_login:
                st.subheader("Welcome Back")
                login_user = st.text_input("Username", key="login_user_input")
                login_pass = st.text_input("Password", type="password", key="login_pass_input")
                if st.button("Log In", type="primary", use_container_width=True):
                    success, user_meta = authenticate_user(login_user, login_pass)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = user_meta["username"]
                        st.session_state.full_name = user_meta["full_name"]
                        st.toast(f"Welcome back, {user_meta['full_name']}!", icon="👋")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                        
            with tab_signup:
                st.subheader("Get Started")
                reg_name = st.text_input("Full Name", placeholder="e.g. Mallik", key="reg_name_input")
                reg_user = st.text_input("Choose Username", key="reg_user_input")
                reg_pass = st.text_input("Choose Password", type="password", key="reg_pass_input")
                if st.button("Create Account", use_container_width=True):
                    if reg_name.strip() and reg_user.strip() and reg_pass.strip():
                        success, message = register_user(reg_user, reg_pass, reg_name)
                        if success:
                            st.success("Account created successfully! You can now log in.")
                            st.toast("Registration complete!", icon="🎉")
                        else:
                            st.error(message)
                    else:
                        st.error("Please fill in all fields.")
            st.markdown("</div>", unsafe_allow_html=True)
        return  # Stop executing main dashboard if not authenticated

    # --- RENDER MAIN DASHBOARD (AUTHENTICATED) ---
    username = st.session_state.username
    full_name = st.session_state.full_name

    st.markdown(f"<h1>💸 <span class='header-tag'>Expense Tracker AI</span> Analytics</h1>", unsafe_allow_html=True)
    st.caption(
        f"Logged in as {full_name} (@{username}) • Auto-categorization • Anomaly detection • Spend forecasting"
    )

    df = get_data(username)
    model = get_model(df, username)

    # Initialize budget state per-user
    budget_key = f"budgets_{username}"
    if budget_key not in st.session_state:
        st.session_state[budget_key] = {
            "Rent & Housing": 22000.0,
            "Education": 15000.0,
            "Travel": 10000.0,
            "Shopping": 8000.0,
            "Groceries": 7000.0,
            "Health": 5000.0,
            "Food & Dining": 4000.0,
            "Utilities": 3000.0,
            "Entertainment": 2000.0,
            "Transport": 2000.0,
        }

    with st.sidebar:
        st.header("👤 Profile")
        st.write(f"**User**: {full_name}")
        st.write(f"**Username**: @{username}")
        if st.button("🚪 Log Out", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.full_name = None
            st.toast("Logged out successfully.", icon="🚪")
            st.rerun()
            
        st.write("---")
        st.header("⚙️ User Controls")
        
        # 1. Manual transaction logger
        with st.expander("➕ Log New Transaction", expanded=False):
            st.write("Record a manual spend. It will be auto-classified instantly.")
            with st.form("manual_tx_form", clear_on_submit=True):
                tx_type = st.radio("Type", ["Expense", "Income"], horizontal=True)
                tx_desc = st.text_input("Merchant / Description", placeholder="e.g. STARBUCKS COFFEE BLR")
                tx_amount = st.number_input("Amount (₹)", min_value=0.01, step=10.0, format="%.2f")
                tx_date = st.date_input("Transaction Date")
                submitted = st.form_submit_button("Log & Classify")
                if submitted:
                    if tx_desc.strip():
                        # Predict category using the user's specific model
                        pred_cat, pred_conf = predict([tx_desc], model)[0]
                        new_row = {
                            "date": tx_date.strftime("%Y-%m-%d"),
                            "description": tx_desc.strip(),
                            "amount": round(tx_amount, 2),
                            "category": pred_cat if tx_type == "Expense" else "Income",
                            "type": tx_type
                        }
                        # Append directly via DB abstraction
                        append_transaction(username, new_row)
                        
                        # Clear streamlit local cache
                        st.cache_data.clear()
                        st.toast(f"Logged! Auto-categorized as **{pred_cat}** ({pred_conf*100:.0f}% confidence)", icon="✅")
                        st.rerun()
                    else:
                        st.error("Please enter a valid description.")

        # 2. Upload CSV
        with st.expander("📤 Import Statements", expanded=False):
            uploaded = st.file_uploader(
                "Select CSV (requires: date, description, amount)", type="csv"
            )
            if uploaded is not None:
                user_df = pd.read_csv(uploaded)
                preds = predict(user_df["description"].tolist(), model)
                user_df["category"] = [p[0] for p in preds]
                user_df["confidence"] = [round(p[1], 2) for p in preds]
                
                # Option to overwrite or merge
                import_mode = st.radio("Import Mode", ["Merge with existing records", "Overwrite existing records"])
                if st.button("Apply Import"):
                    if import_mode == "Overwrite existing records":
                        save_transactions(username, user_df)
                    else:
                        old_df = get_transactions(username)
                        # ensure txn_id is unique
                        start_id = int(old_df["txn_id"].max() + 1) if not old_df.empty else 1
                        user_df["txn_id"] = range(start_id, start_id + len(user_df))
                        combined = pd.concat([old_df, user_df[["txn_id", "date", "description", "amount", "category", "type"]]], ignore_index=True)
                        save_transactions(username, combined)
                        
                    st.cache_data.clear()
                    st.success("Ledger updated successfully!")
                    st.rerun()

        # 3. Monthly Budget Manager
        with st.expander("🎯 Set Monthly Limits", expanded=False):
            st.write("Set spending thresholds per category:")
            new_budgets = {}
            for cat, val in st.session_state[budget_key].items():
                new_budgets[cat] = st.number_input(f"{cat} (₹)", min_value=0.0, value=float(val), step=500.0)
            
            if st.button("Save Budgets"):
                st.session_state[budget_key] = new_budgets
                st.success("Budgets saved successfully!")
                st.rerun()

        st.write("---")
        st.header("📅 Global Date Filter")
        if not df.empty:
            min_d = pd.to_datetime(df["date"]).min().date()
            max_d = pd.to_datetime(df["date"]).max().date()
            date_range = st.date_input("Select Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(pd.to_datetime(df["date"]).dt.date >= start_date) & (pd.to_datetime(df["date"]).dt.date <= end_date)]

        st.write("---")
        st.header("💾 Export Data")
        if not df.empty:
            @st.cache_data
            def convert_df(d):
                return d.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Filtered Data (CSV)",
                convert_df(df),
                "transactions.csv",
                "text/csv",
                use_container_width=True
            )

        st.write("---")
        
        # 4. Reset operation
        if st.button("🗑️ Delete All My Records", type="secondary", use_container_width=True):
            save_transactions(username, pd.DataFrame(columns=["txn_id", "date", "description", "amount", "category", "type", "username"]))
            st.cache_data.clear()
            st.success("All records deleted. You now have a clean slate!")
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🚨 Anomalies", "📈 Forecast & Trends", "💬 Ask a question"]
    )

    # ---------------- Tab 1: Overview ----------------
    with tab1:
        # MoM calculations
        max_date = pd.to_datetime(df["date"]).max() if not df.empty else pd.Timestamp.now()
        recent_30 = df[(pd.to_datetime(df["date"]) > max_date - pd.Timedelta(days=30))]
        prev_30 = df[(pd.to_datetime(df["date"]) > max_date - pd.Timedelta(days=60)) & (pd.to_datetime(df["date"]) <= max_date - pd.Timedelta(days=30))]
        
        recent_spend = recent_30[recent_30["type"] == "Expense"]["amount"].sum()
        prev_spend = prev_30[prev_30["type"] == "Expense"]["amount"].sum()
        spend_delta = ((recent_spend - prev_spend) / prev_spend * 100) if prev_spend > 0 else 0
        spend_delta_str = f"+{spend_delta:.1f}%" if spend_delta > 0 else f"{spend_delta:.1f}%"
        spend_color = "#ff3d00" if spend_delta > 0 else "#00e676"
        
        total_income = df[df["type"] == "Income"]["amount"].sum()
        total_spend = df[df["type"] == "Expense"]["amount"].sum()
        net_cash = total_income - total_spend

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Total Income</div><div class='metric-val' style='color:#00e676;'>₹{total_income:,.0f}</div></div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Total Spend</div><div class='metric-val' style='color:#ff3d00;'>₹{total_spend:,.0f}</div><div style='font-size:12px; color:{spend_color}'>{spend_delta_str} vs last 30d</div></div>",
                unsafe_allow_html=True
            )
        with col3:
            net_color = "#00e676" if net_cash >= 0 else "#ff3d00"
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Net Cash Flow</div><div class='metric-val' style='color:{net_color};'>₹{net_cash:,.0f}</div></div>",
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"<div class='metric-box'><div class='metric-lbl'>Transactions</div><div class='metric-val' style='color:#2979ff;'>{len(df):,}</div></div>",
                unsafe_allow_html=True
            )

        chart_col, budget_col = st.columns([1.3, 1])

        with chart_col:
            st.subheader("Spending Analysis")
            expenses_only = df[df["type"] == "Expense"]
            by_cat = expenses_only.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(
                by_cat, x="category", y="amount", 
                title="Spend by Category",
                color="amount",
                color_continuous_scale="Purples",
                labels={"amount": "Amount (₹)", "category": "Category"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Outfit",
                title_font_size=18,
                title_font_color="#f8f9fa",
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Amount (₹)"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, width="stretch")

            df_dated = df.copy()
            df_dated["date"] = pd.to_datetime(df_dated["date"])
            daily = df_dated.groupby(df_dated["date"].dt.to_period("W").apply(lambda p: p.start_time))["amount"].sum().reset_index()
            fig2 = px.line(
                daily, x="date", y="amount", 
                title="Weekly Spend Over Time", 
                markers=True
            )
            fig2.update_traces(line_color="#7f00ff", marker=dict(size=6, color="#e100ff"))
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Outfit",
                title_font_size=18,
                title_font_color="#f8f9fa",
                xaxis=dict(showgrid=False, title="Week Starting"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Amount (₹)")
            )
            st.plotly_chart(fig2, width="stretch")

            # Top merchants
            st.subheader("Top Merchants (Expenses)")
            from categorize import clean_text
            expenses_only["merchant"] = expenses_only["description"].apply(lambda x: clean_text(x).split()[0] if clean_text(x) else "Unknown")
            top_m = expenses_only.groupby("merchant")["amount"].sum().sort_values(ascending=False).head(5).reset_index()
            fig_pie = px.pie(top_m, names="merchant", values="amount", hole=0.4)
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Outfit", title_font_color="#f8f9fa", margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, width="stretch")

        with budget_col:
            st.subheader("Monthly Budget Health")
            st.caption("Latest calendar month's aggregated spending vs set budget thresholds:")

            # Calculate latest month's spend
            latest_date = pd.to_datetime(df["date"]).max()
            first_of_latest_month = latest_date.replace(day=1)
            df_latest_month = df[pd.to_datetime(df["date"]) >= first_of_latest_month]
            month_spend = df_latest_month.groupby("category")["amount"].sum().to_dict()

            for cat, limit in st.session_state[budget_key].items():
                spent = month_spend.get(cat, 0.0)
                pct = min(spent / limit, 1.0) if limit > 0 else 0.0
                pct_val = (spent / limit) * 100 if limit > 0 else 0.0

                color = "#2ecc71"  # green
                if pct_val > 100:
                    color = "#ff3d00"  # red
                elif pct_val > 80:
                    color = "#ffc400"  # yellow-orange

                progress_html = f"""
                <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #f8f9fa;">{cat}</span>
                        <span style="color: #a0aec0;">₹{spent:,.0f} / ₹{limit:,.0f} ({pct_val:.0f}%)</span>
                    </div>
                    <div style="width: 100%; background: rgba(255,255,255,0.08); height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="width: {pct*100}%; background: {color}; height: 100%; border-radius: 3px; transition: width 0.5s ease-in-out;"></div>
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)

        st.subheader("Transaction Records (Editable)")
        st.caption("You can edit or delete rows here directly. Changes are saved when you click the button.")
        edited_df = st.data_editor(
            df.sort_values("date", ascending=False), 
            use_container_width=True, 
            height=300,
            num_rows="dynamic",
            key="tx_editor"
        )
        if st.button("Save Table Changes", type="primary"):
            save_transactions(username, edited_df)
            st.cache_data.clear()
            st.success("Changes saved successfully!")
            st.rerun()

    # ---------------- Tab 2: Anomalies ----------------
    with tab2:
        st.subheader("Weekly Spending Alerts")
        alerts = weekly_spend_alerts(df)
        if alerts:
            for a in alerts:
                st.markdown(f"<div class='alert-card'>{a}</div>", unsafe_allow_html=True)
        else:
            st.success("No weekly pattern shifts detected. Spending remains within typical statistical boundaries.")

        st.subheader("Transaction Outlier Analysis")
        flagged = flag_transaction_outliers(df)
        outliers = flagged[flagged["is_outlier"]].sort_values("date", ascending=False)
        st.write(f"The Isolation Forest flagged **{len(outliers)} unusual single transactions** based on statistical density.")

        # Plot outliers in a scatter plot
        fig_out = px.scatter(
            flagged, x="date", y="amount", 
            color="is_outlier", 
            color_discrete_map={True: "#ff3d00", False: "rgba(255,255,255,0.25)"},
            title="Outlier Transactions Highlighted",
            labels={"amount": "Amount (₹)", "is_outlier": "Is Outlier?"},
            hover_data=["description", "category"]
        )
        fig_out.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Outfit",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_out, width="stretch")

        st.dataframe(
            outliers[["date", "description", "amount", "category"]],
            width="stretch"
        )

    # ---------------- Tab 3: Forecast ----------------
    with tab3:
        st.subheader("Day of Week Spending Heatmap")
        expenses = df[df["type"] == "Expense"].copy()
        if not expenses.empty:
            expenses["dow"] = pd.to_datetime(expenses["date"]).dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_spend = expenses.groupby("dow")["amount"].mean().reindex(dow_order).reset_index()
            fig_dow = px.bar(dow_spend, x="dow", y="amount", title="Average Spend by Day of Week", color="amount", color_continuous_scale="Purples")
            fig_dow.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family="Outfit")
            st.plotly_chart(fig_dow, width="stretch")

        st.write("---")
        st.subheader("🔄 Detected Subscriptions & Recurring Bills")
        subs = detect_subscriptions(df)
        if not subs.empty:
            st.dataframe(subs, use_container_width=True)
            st.info(f"Detected {len(subs)} recurring charges totaling ~₹{subs['estimated_yearly_cost'].sum():,.0f} per year.")
        else:
            st.success("No recurring subscriptions detected.")

        st.write("---")
        st.subheader("Category Expenditure Forecast")
        st.write("Recency-weighted trend analysis projecting spend for future weeks:")

        f_weeks = st.slider("Forecast Horizon (weeks)", min_value=1, max_value=4, value=2)

        # dropdown category selector
        cats_list = ["All Categories"] + sorted(df["category"].unique().tolist())
        selected_cat = st.selectbox("View forecast detail for:", cats_list)

        timeline_df = get_forecast_timeline(df, periods_ahead=f_weeks)

        if selected_cat == "All Categories":
            # Group by week and type
            agg_timeline = timeline_df.groupby(["week", "type"])[["amount", "lower_bound", "upper_bound"]].sum().reset_index()
            agg_timeline = agg_timeline.sort_values("week")

            hist = agg_timeline[agg_timeline["type"] == "Historical"]
            fore = agg_timeline[agg_timeline["type"] == "Forecast"]
            # Connect the end of history with the start of forecast
            if not hist.empty and not fore.empty:
                fore = pd.concat([hist.tail(1), fore], ignore_index=True)

            fig_fore = go.Figure()

            # Historical line
            fig_fore.add_trace(go.Scatter(
                x=hist["week"], y=hist["amount"],
                mode="lines+markers",
                name="Historical Spend",
                line=dict(color="#00e676", width=3),
                marker=dict(size=6)
            ))

            # Forecasted line
            fig_fore.add_trace(go.Scatter(
                x=fore["week"], y=fore["amount"],
                mode="lines+markers",
                name="Projected Spend",
                line=dict(color="#7f00ff", width=3, dash="dash"),
                marker=dict(size=6, color="#e100ff")
            ))

            # CI Bounds Shaded Area
            fig_fore.add_trace(go.Scatter(
                x=pd.concat([fore["week"], fore["week"].iloc[::-1]]),
                y=pd.concat([fore["upper_bound"], fore["lower_bound"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(127, 0, 255, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Uncertainty Range (~80% CI)"
            ))

            fig_fore.update_layout(
                title="Overall Weekly Expenditure Forecast Timeline",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Outfit",
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Weekly Spend (₹)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fore, width="stretch")

            # Display Forecast Table
            st.subheader("Category-wise Breakdowns for Next Week")
            fc_next = forecast_next_period(df)
            st.dataframe(fc_next, width="stretch")

            total_next_week = fc_next["forecast_amount"].sum()
            st.metric("Total predicted spend next week", f"₹{total_next_week:,.0f}")

        else:
            cat_timeline = timeline_df[timeline_df["category"] == selected_cat].sort_values("week")
            hist = cat_timeline[cat_timeline["type"] == "Historical"]
            fore = cat_timeline[cat_timeline["type"] == "Forecast"]

            if not hist.empty and not fore.empty:
                fore = pd.concat([hist.tail(1), fore], ignore_index=True)

            fig_fore = go.Figure()

            # Historical line
            fig_fore.add_trace(go.Scatter(
                x=hist["week"], y=hist["amount"],
                mode="lines+markers",
                name="Historical Spend",
                line=dict(color="#00e676", width=3),
                marker=dict(size=6)
            ))

            # Forecasted line
            fig_fore.add_trace(go.Scatter(
                x=fore["week"], y=fore["amount"],
                mode="lines+markers",
                name="Projected Spend",
                line=dict(color="#7f00ff", width=3, dash="dash"),
                marker=dict(size=6, color="#e100ff")
            ))

            # CI Bounds Shaded Area
            fig_fore.add_trace(go.Scatter(
                x=pd.concat([fore["week"], fore["week"].iloc[::-1]]),
                y=pd.concat([fore["upper_bound"], fore["lower_bound"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(127, 0, 255, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Uncertainty Range (~80% CI)"
            ))

            fig_fore.update_layout(
                title=f"Weekly Spend Timeline & Projection - {selected_cat}",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Outfit",
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Weekly Spend (₹)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_fore, width="stretch")

            # Detail Metrics
            cat_fc = forecast_next_period(df)
            cat_fc_row = cat_fc[cat_fc["category"] == selected_cat]
            if not cat_fc_row.empty:
                f_amt = cat_fc_row["forecast_amount"].values[0]
                l_bound = cat_fc_row["lower_bound"].values[0]
                u_bound = cat_fc_row["upper_bound"].values[0]
                st.metric(
                    f"Forecasted {selected_cat} Spend Next Week", 
                    f"₹{f_amt:,.2f}",
                    help=f"Confidence interval: ₹{l_bound:,.2f} to ₹{u_bound:,.2f}"
                )

    # ---------------- Tab 4: Ask a Question ----------------
    with tab4:
        st.subheader("Grounded Natural Language Query Engine")
        st.write("Type a question to query your spending data. This layer parses intent and computes values directly from database records to guarantee factual accuracy.")

        st.markdown("""
        **Suggested advanced queries you can try:**
        - *"How much did I spend at Amazon last month?"*
        - *"What was my average Swiggy transaction this month?"*
        - *"How many shopping transactions did I make between 500 and 1500 last month?"*
        - *"What was my highest transaction above 2000 in June?"*
        - *"How much did I spend in total between 2026-06-01 and 2026-06-15?"*
        """)

        question = st.text_input("Ask a question about your expenses:", placeholder="e.g. How much did I spend on dining last month?")
        if question:
            answer = answer_query(question, df)
            st.markdown(f"""
            <div style="background: rgba(127, 0, 255, 0.05); border: 1px solid rgba(127, 0, 255, 0.2); border-radius: 8px; padding: 18px; font-size: 16px; color: #f8f9fa;">
                <span style="font-weight: 700; color: #e100ff;">Answer:</span> {answer}
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
