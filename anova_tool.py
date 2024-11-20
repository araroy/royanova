import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# Title and Description
st.title("ANOVA and ANCOVA Analysis Tool")
st.markdown("""
Upload your data (CSV or Excel), specify dependent and independent variables, and perform ANOVA or ANCOVA.
Includes options for custom contrasts and visualization.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Determine the file type and read the data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.write(df.head())

        # Select Dependent Variable
        dependent_var = st.selectbox("Select Dependent Variable (DV)", options=df.columns)

        # Select Independent Variables (Factors)
        independent_vars = st.multiselect(
            "Select Independent Variables (Factors)", 
            options=[col for col in df.columns if col != dependent_var]
        )

        # Select Covariates (Optional)
        covariates = st.multiselect(
            "Select Covariates (Optional, for ANCOVA)", 
            options=[col for col in df.columns if col not in independent_vars and col != dependent_var]
        )

        # Check that variables are selected
        if not independent_vars:
            st.error("Please select at least one independent variable.")
            st.stop()

        # Specify model formula
        formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
        if covariates:
            formula += " + " + " + ".join(covariates)

        st.markdown(f"**Model Formula**: `{formula}`")

        # Run ANOVA or ANCOVA
        if st.button("Run Analysis"):
            try:
                model = ols(formula, data=df).fit()
                anova_results = anova_lm(model, typ=2)  # Type-II ANOVA
                st.write("### ANOVA/ANCOVA Results:")
                st.write(anova_results)

                # Bar Plot of Means
                st.markdown("### Visualization:")
                means = df.groupby(independent_vars)[dependent_var].mean().reset_index()
                fig, ax = plt.subplots()
                means.plot(kind='bar', x=independent_vars, y=dependent_var, ax=ax, legend=False)
                ax.set_title(f"Mean {dependent_var} by {', '.join(independent_vars)}")
                ax.set_ylabel(dependent_var)
                ax.set_xlabel(", ".join(independent_vars))
                st.pyplot(fig)

                # Post-Hoc Contrasts
                st.markdown("### Post-Hoc Contrasts:")
                if len(independent_vars) == 1:
                    # Pairwise Tukey HSD for single-factor ANOVA
                    tukey_results = pairwise_tukeyhsd(
                        df[dependent_var],
                        df[independent_vars[0]],
                        alpha=0.05
                    )
                    st.write(tukey_results)
                else:
                    st.info("Post-hoc contrasts are only supported for single-factor ANOVA.")
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
