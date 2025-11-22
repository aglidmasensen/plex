import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
from itertools import combinations
import xlsxwriter
from io import BytesIO

# ==========================================
#        MODULE 1: 2^k (USER'S CODE)
# ==========================================

def generate_design_matrix_2k(k):
    N = 2 ** k
    # Generate the basic factors
    factors = []
    for i in range(k):
        factor = np.tile([-1 if j < N//2**(i+1) else 1 for j in range(N//2**i)], 2**i)
        factors.append(factor)
    # All combinations for interactions
    num_terms = 2 ** k
    X = np.ones((N, num_terms))
    # x0 is already 1
    col_idx = 1
    # Main effects
    for i in range(k):
        X[:, col_idx] = factors[i]
        col_idx += 1
    # Interactions
    for order in range(2, k+1):
        for combo in combinations(range(k), order):
            interaction = np.prod([factors[j] for j in combo], axis=0)
            X[:, col_idx] = interaction
            col_idx += 1
    return X

def module_1_logic():
    st.header("Module 1: Plans factoriels complets à deux niveaux (2k)")
    
    # Inputs
    k = st.number_input("Enter k , sekcem K:", min_value=1, step=1, value=1, key="m1_k")
    y_str = st.text_area("Enter y vector , sekcem tigejdit n y:", key="m1_y")
    y0_str = st.text_area("Enter y0 vector , sekcem tigejdit n y0:", key="m1_y0")

    if st.button("Run Analysis (2k) , selhu tasleḍt", key="m1_btn"):
        if not y_str or not y0_str or k < 1:
            st.error("Please enter all inputs, ttxil sekcem ak isekkilen.")
            return

        # Parse vectors
        try:
            y = np.array([float(line.strip()) for line in y_str.splitlines() if line.strip()])
            y0 = np.array([float(line.strip()) for line in y0_str.splitlines() if line.strip()])
        except ValueError:
            st.error("Error parsing numbers. Please check inputs.")
            return
            
        N = len(y)
        n0 = len(y0)

        if N != 2 ** k:
            st.error(f"y must have length 2^k, y ilaq adis3u = {2**k}")
            return
        if n0 < 2:
            st.error("y0 must have at least 2 points for variance.")
            return

        # Generate X
        X = generate_design_matrix_2k(k)

        # Compute bj
        XtX_inv = np.linalg.inv(X.T @ X)
        bj = XtX_inv @ X.T @ y

        # Reproducibility
        y0bar = np.mean(y0)
        S_rep2 = np.sum((y0 - y0bar)**2) / (n0 - 1)
        s_rep = np.sqrt(S_rep2)
        sbj = s_rep / np.sqrt(N)

        # Student t-test
        tj = np.abs(bj) / sbj
        alpha = 0.05
        ddl = n0 - 1
        t_tab = stats.t.ppf(1 - alpha/2, ddl)  # two-tailed
        bjs = np.where(tj > t_tab, bj, 0)

        # Display bj results
        terms = ['b0'] + [f'b{i+1}' for i in range(k)] + ['interactions...'] * (len(bj) - 1 - k)
        bj_df = pd.DataFrame({
            'Term': terms,
            'bj': bj,
            'tj': tj,
            'Significant': tj > t_tab
        })
        st.subheader("Full Coefficients (bj)")
        st.dataframe(bj_df)
        st.write(f"t_tab (ddl={ddl}, α={alpha}): {t_tab:.3f}")

        # Reduced predictions
        y_pred = X @ bjs

        # Residual variance
        l = np.sum(bjs != 0)
        S_res2 = np.sum((y - y_pred)**2) / (N - l)

        # Fisher F-test
        F_calc = S_res2 / S_rep2
        ddl1 = N - l
        ddl2 = n0 - 1
        F_tab = stats.f.ppf(1 - alpha, ddl1, ddl2)
        is_valid = F_calc <= F_tab

        st.subheader("Fisher Test")
        st.write(f"S_rep²: {S_rep2:.3f}")
        st.write(f"S_residual²: {S_res2:.3f}")
        st.write(f"F_calc: {F_calc:.3f}")
        st.write(f"F_tab (ddl1={ddl1}, ddl2={ddl2}, α={alpha}): {F_tab:.3f}")
        if is_valid:
            st.success("Model is VALID (no lack of fit)")
        else:
            st.error("Model shows lack of fit")

        # === F2 + R² + R²' + Mmoy ===
        ybar = np.mean(y)
        SS_reg = np.sum((y_pred - ybar)**2)
        SS_tot = np.sum((y - ybar)**2)

        # F2
        if l > 1:
            F2_calc = SS_reg / (l - 1) / S_res2
        else:
            F2_calc = np.nan
        ddl_F2_num = l - 1
        ddl_F2_den = N - l
        if ddl_F2_num > 0 and ddl_F2_den > 0:
            F2_tab = stats.f.ppf(1 - alpha, ddl_F2_num, ddl_F2_den)
            F2_valid = F2_calc > F2_tab if not np.isnan(F2_calc) else False
        else:
            F2_tab = np.nan
            F2_valid = False

        st.subheader("F2 Validation (Regression Significance)")
        st.write(f"F2_calc = {F2_calc:.3f}" if not np.isnan(F2_calc) else "F2_calc = N/A (l=1)")
        st.write(f"F2_tab (ddl_num={ddl_F2_num}, ddl_den={ddl_F2_den}): {F2_tab:.3f}" if not np.isnan(F2_tab) else "F2_tab = N/A")
        if F2_valid:
            st.success("F2 > F2_tab → Model SIGNIFICANT at 95%")
        else:
            st.error("F2 ≤ F2_tab → Model NOT significant")

        # R² and R²'
        R2 = SS_reg / SS_tot if SS_tot > 0 else 0
        R2_adj = R2 - (1 - R2) * (l - 1) / (N - l) if (N - l) > 0 else np.nan

        st.subheader("Goodness of Fit")
        st.write(f"R² = {R2:.4f}")
        st.write(f"R²' (adjusted) = {R2_adj:.4f}" if not np.isnan(R2_adj) else "R²' = N/A")

        # Mmoy
        ei = y - y_pred
        Mmoy = np.sum(np.abs(ei)) / N
        st.write(f"Moyenne des résidus absolus (Mmoy) = {Mmoy:.4f}")

        # Reduced coefficients
        bjs_df = pd.DataFrame({'Term': terms, 'bjs': bjs})
        st.subheader("Significant Coefficients (bjs)")
        st.dataframe(bjs_df[bjs_df['bjs'] != 0])
        
        # Plots
        if k <= 3:
            st.subheader("Response Surface")
            if k == 1:
                x1 = X[:, 1]
                fig = go.Figure(go.Scatter(x=x1, y=y_pred, mode='lines', name='Prédit'))
                fig.add_scatter(x=x1, y=y, mode='markers', name='Observé')
            elif k == 2:
                x1, x2 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
                y_surf = bjs[0] + bjs[1]*x1 + bjs[2]*x2 + bjs[3]*x1*x2
                fig = go.Figure(go.Surface(x=x1, y=x2, z=y_surf))
            elif k == 3:
                st.info("For k=3, showing slice at x3=0")
                x1, x2 = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
                x3 = 0
                y_surf = (bjs[0] + bjs[1]*x1 + bjs[2]*x2 + bjs[3]*x3 +
                          bjs[4]*x1*x2 + bjs[5]*x1*x3 + bjs[6]*x2*x3 + bjs[7]*x1*x2*x3)
                fig = go.Figure(go.Surface(x=x1, y=x2, z=y_surf))
            st.plotly_chart(fig)


# ==========================================
#        MODULE 2: PLAN COMPOSITE CENTRÉ (CCD)
# ==========================================

def generate_ccd_matrix(k, alpha, n_total_runs):
    """
    Generates standard order CCD matrix:
    1. Factorial Points (2^k)
    2. Axial Points (2k)
    3. Center Points (n0)
    """
    import itertools
    
    # 1. Factorial Part (Cube)
    levels = [-1, 1]
    factorial_part = np.array(list(itertools.product(levels, repeat=k)))
    
    # 2. Axial Part (Star)
    axial_part = np.zeros((2*k, k))
    row_idx = 0
    for i in range(k):
        # -alpha
        axial_part[row_idx, i] = -alpha
        row_idx += 1
        # +alpha
        axial_part[row_idx, i] = alpha
        row_idx += 1
        
    # 3. Center Part
    n0 = n_total_runs - len(factorial_part) - len(axial_part)
    center_part = np.zeros((n0, k))
    
    # Stack them: X_linear columns (Factors)
    X_linear = np.vstack([factorial_part, axial_part, center_part])
    N = len(X_linear)
    
    # --- Build Matrix X (Chapter 9 Model) ---
    # Order: b0, b_i (linear), b_ii (quadratic), b_ij (interactions)
    
    # Intercept
    X0 = np.ones((N, 1))
    
    # Linear (Xi)
    # X_linear is already correct
    
    # Quadratic (Xi^2)
    X_quad = X_linear ** 2
    
    # Interactions (Xi * Xj)
    X_inter = []
    inter_labels = []
    for i in range(k):
        for j in range(i+1, k):
            X_inter.append(X_linear[:, i] * X_linear[:, j])
            inter_labels.append(f"b{i+1}{j+1}")
            
    X_inter = np.column_stack(X_inter) if X_inter else np.zeros((N, 0))
    
    # Assemble X
    X = np.hstack([X0, X_linear, X_quad, X_inter])
    
    # Create Labels
    labels = ['b0'] 
    labels += [f"b{i+1}" for i in range(k)] 
    labels += [f"b{i+1}{i+1}" for i in range(k)]
    labels += inter_labels
    
    return X, labels, n0

def module_2_logic():
    st.header("Module 2: Plan Composite Centré (CCD)")
    st.caption("Calculations based on Chapter 9: Plans composites centrés")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.number_input("k (Factors/Facteurs)", min_value=2, value=2, step=1, key="m2_k")
    with col2:
        alpha = st.number_input("α (Alpha/Axial)", value=1.414, step=0.01, format="%.3f", key="m2_alpha")
    with col3:
        st.info("n0 is detected automatically from y0")

    st.markdown("---")
    st.write("Enter the 3 response vectors (Respect order!) / Sekcem 3 tigejdine:")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**1. y (Factorial/Factoriel)**\n({2**k} values/tigejdine)")
        y_fact_str = st.text_area("y (cube)", height=150, key="m2_yfact")
    with c2:
        st.markdown(f"**2. y' (Axial/Étoile)**\n({2*k} values/tigejdine)")
        y_ax_str = st.text_area("y' (star)", height=150, key="m2_yax")
    with c3:
        st.markdown("**3. y0 (Center/Centre)**\n(n0 values/tigejdine)")
        y0_str = st.text_area("y0 (center)", height=150, key="m2_y0")

    if st.button("Run Analysis (CCD) , selhu tasleḍt", key="m2_btn"):
        try:
            # 1. Parse Inputs
            y_fact = np.array([float(x) for x in y_fact_str.splitlines() if x.strip()])
            y_ax = np.array([float(x) for x in y_ax_str.splitlines() if x.strip()])
            y0 = np.array([float(x) for x in y0_str.splitlines() if x.strip()])
            
            # 2. Validate Lengths
            if len(y_fact) != 2**k:
                st.error(f"Error: y (Factorial) must have {2**k} values. Found {len(y_fact)}.")
                return
            if len(y_ax) != 2*k:
                st.error(f"Error: y' (Axial) must have {2*k} values. Found {len(y_ax)}.")
                return
            if len(y0) < 2:
                st.error("Error: y0 (Center) must have at least 2 values (replicates) for stats.")
                return
                
            # 3. Combine Y total
            y_total = np.concatenate([y_fact, y_ax, y0])
            N = len(y_total)
            
            # 4. Generate Matrix X
            X, labels, n0_detected = generate_ccd_matrix(k, alpha, N)
            
            # 5. Reproducibility (S_rep^2) - Base for all stats (Page 62)
            y0_bar = np.mean(y0)
            S_rep2 = np.sum((y0 - y0_bar)**2) / (n0_detected - 1)
            s_rep = np.sqrt(S_rep2)
            
            st.write(f"**Variance de Reproductibilité ($S^2_{{rep}}$)**: {S_rep2:.4f} (ddl = {n0_detected-1})")
            
            # 6. Calculate Coefficients (bj)
            XtX = X.T @ X
            if np.linalg.cond(XtX) > 1e10:
                st.error("Matrix is singular. Check inputs.")
                return
            XtX_inv = np.linalg.inv(XtX)
            bj = XtX_inv @ X.T @ y_total
            
            # 7. Student Test (Signification des coefficients)
            # Variance of bj = S_rep2 * Cjj (diagonal of XtX_inv) (Page 62)
            Cjj = np.diag(XtX_inv)
            var_bj = S_rep2 * Cjj
            sbj = np.sqrt(var_bj)
            
            tj = np.abs(bj) / sbj
            alpha_val = 0.05
            ddl_err = n0_detected - 1
            t_tab = stats.t.ppf(1 - alpha_val/2, ddl_err)
            
            is_sig = tj > t_tab
            bjs = np.where(is_sig, bj, 0)
            
            # Calculate Lambda (Number of significant coefficients including b0)
            lambda_val = np.count_nonzero(bjs)
            
            bj_df = pd.DataFrame({
                'Term': labels,
                'bj': bj,
                'tj': tj,
                'Significant': is_sig
            })
            st.subheader("1. Coefficients & Test de Student")
            st.dataframe(bj_df)
            st.write(f"$t_{{tab}}$ (0.05, {ddl_err}): **{t_tab:.3f}**")
            st.write(f"Nombre de coefficients significatifs ($\\lambda$): **{lambda_val}**")
            
            # 8. Recherche de Biais (Validation du modèle réduit) - (Page 63)
            # Uses the REDUCED model (bjs)
            y_pred_reduced = X @ bjs
            
            # Variance Résiduelle (S^2_res)
            residuals = y_total - y_pred_reduced
            SS_res_total = np.sum(residuals**2)
            ddl_res_total = N - lambda_val
            
            S_res2 = SS_res_total / ddl_res_total
            
            st.subheader("2. Recherche de Biais (Fisher Test 1)")
            F_biais = S_res2 / S_rep2
            F_biais_tab = stats.f.ppf(1 - alpha_val, ddl_res_total, ddl_err)
            
            st.write(f"Variance Résiduelle ($S^2_{{res}}$): {S_res2:.4f} (ddl={ddl_res_total})")
            st.write(f"Variance Reproductibilité ($S^2_{{rep}}$): {S_rep2:.4f} (ddl={ddl_err})")
            st.write(f"$F_{{calc}}$ = $S^2_{{res}} / S^2_{{rep}}$: **{F_biais:.4f}**")
            st.write(f"$F_{{tab}}$ ({ddl_res_total}, {ddl_err}): **{F_biais_tab:.4f}**")
            
            if F_biais <= F_biais_tab:
                st.success("✅ Modèle sans biais (Validé)")
            else:
                st.error("❌ Modèle biaisé (Non validé)")
                
            # 9. Test de Signification de la Régression (Fisher Test 2) - (Page 63)
            # Compares Regression Variance to Residual Variance
            y_grand_mean = np.mean(y_total)
            
            # SS_reg (Regression Sum of Squares) using Reduced Model
            SS_reg = np.sum((y_pred_reduced - y_grand_mean)**2)
            ddl_reg = lambda_val - 1
            
            st.subheader("3. Signification de la Régression (Fisher Test 2)")
            if ddl_reg > 0:
                MS_reg = SS_reg / ddl_reg
                F_reg = MS_reg / S_res2
                F_reg_tab = stats.f.ppf(1 - alpha_val, ddl_reg, ddl_res_total)
                
                st.write(f"$MS_{{reg}}$: {MS_reg:.4f} (ddl={ddl_reg})")
                st.write(f"$F_{{calc}}$ = $MS_{{reg}} / S^2_{{res}}$: **{F_reg:.4f}**")
                st.write(f"$F_{{tab}}$ ({ddl_reg}, {ddl_res_total}): **{F_reg_tab:.4f}**")
                
                if F_reg > F_reg_tab:
                    st.success("✅ Régression Significative")
                else:
                    st.error("❌ Régression Non Significative")
            else:
                st.warning("Pas assez de termes significatifs pour tester la régression.")

            # 10. R2 & R2 adjusted & Mmoy - (Page 64)
            SS_tot = np.sum((y_total - y_grand_mean)**2)
            R2 = 1 - (SS_res_total / SS_tot)
            R2_adj = 1 - ( (1 - R2) * (N - 1) / (N - lambda_val) )
            
            Mmoy = np.mean(np.abs(residuals))
            
            st.subheader("4. Qualité de l'ajustement")
            st.write(f"$R^2$: **{R2:.4f}**")
            st.write(f"$R^2_{{ajusté}}$: **{R2_adj:.4f}**")
            st.write(f"Moyenne des résidus absolus ($M_{{moy}}$): **{Mmoy:.4f}**")
            
            # 11. Equation Reduced
            st.subheader("Modèle Réduit")
            eq_parts = []
            for i, c in enumerate(bjs):
                if c != 0:
                    eq_parts.append(f"{c:+.3f}*{labels[i]}")
            st.code("Y = " + " ".join(eq_parts))
            
            # 12. Plots
            if k == 2:
                st.subheader("Surface de Réponse (2D)")
                grid_range = np.linspace(-alpha, alpha, 30)
                x1g, x2g = np.meshgrid(grid_range, grid_range)
                
                # Manual reconstruction for k=2 standard order
                # labels order: b0, b1, b2, b11, b22, b12
                b = bjs
                z = (b[0] + b[1]*x1g + b[2]*x2g + 
                     b[3]*x1g**2 + b[4]*x2g**2 + 
                     b[5]*x1g*x2g)
                     
                fig = go.Figure(data=[go.Surface(z=z, x=x1g, y=x2g)])
                fig.update_layout(title='Surface Response', height=500)
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")

def main():
    st.set_page_config(layout="wide", page_title="GPE-M2 Software")
    st.title("Plan d'Expérience , aɣawas n tirmit made by GPE-M2")
    
    menu = ["Module 1: 2^k Factorial", "Module 2: CCD (Composite Centré)"]
    choice = st.sidebar.selectbox("Select Module / Fren Taɣult", menu)
    
    if choice == "Module 1: 2^k Factorial":
        module_1_logic()
    else:
        module_2_logic()

if __name__ == "__main__":
    main()
