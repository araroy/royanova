import streamlit as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Enhanced Statistical Analysis Tool")
st.markdown("""
Perform ANOVA, Chi-Square Test, PROCESS Models, and Advanced Data Cleaning:
- Flexible data cleaning: create new variables using operations like mean, sum, subtraction, or merging.
- Perform ANOVA with covariates, bar chart, and pairwise contrasts.
- Chi-Square Test with count tables and bar chart visualization.
- Add custom labels for categorical variables.
- PROCESS Models: Mediation (Model 4), Moderated Mediation (Model 7), and Moderated Moderation (Model 14).
""")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load the dataset
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.write(df.head())

        # Initialize session state for dynamic variables
        if "df" not in st.session_state:
            st.session_state["df"] = df

        # Data Cleaning Section
        st.markdown("### Data Cleaning")
        st.write("Create new variables by applying operations on existing columns.")

        operation = st.selectbox("Select Cleaning Operation", ["None", "Mean", "Sum", "8 - Variable", "Merge Two Columns (Remove Blanks)"])
        if operation != "None":
            if operation == "Mean":
                columns_to_average = st.multiselect("Select Columns to Average", options=st.session_state["df"].columns)
                new_variable_name = st.text_input("New Variable Name", "mean_variable")
                if st.button("Create Mean Variable"):
                    st.session_state["df"][new_variable_name] = st.session_state["df"][columns_to_average].mean(axis=1)
                    st.success(f"New variable '{new_variable_name}' created.")
            
            elif operation == "Sum":
                columns_to_sum = st.multiselect("Select Columns to Sum", options=st.session_state["df"].columns)
                new_variable_name = st.text_input("New Variable Name", "sum_variable")
                if st.button("Create Sum Variable"):
                    st.session_state["df"][new_variable_name] = st.session_state["df"][columns_to_sum].sum(axis=1)
                    st.success(f"New variable '{new_variable_name}' created.")

            elif operation == "8 - Variable":
                column_to_subtract = st.selectbox("Select Column to Subtract from 8", options=st.session_state["df"].columns)
                new_variable_name = st.text_input("New Variable Name", "subtract_variable")
                if st.button("Create Subtracted Variable"):
                    st.session_state["df"][new_variable_name] = 8 - st.session_state["df"][column_to_subtract]
                    st.success(f"New variable '{new_variable_name}' created.")

            elif operation == "Merge Two Columns (Remove Blanks)":
                col1 = st.selectbox("Select First Column", options=st.session_state["df"].columns, key="merge_col1")
                col2 = st.selectbox("Select Second Column", options=st.session_state["df"].columns, key="merge_col2")
                new_variable_name = st.text_input("New Variable Name", "merged_variable")
                if st.button("Merge Columns"):
                    st.session_state["df"][new_variable_name] = st.session_state["df"][col1].combine_first(st.session_state["df"][col2])
                    st.success(f"New variable '{new_variable_name}' created.")

        # Updated DataFrame
        df = st.session_state["df"]
        st.write("Updated Data:")
        st.write(df.head())

        # Analysis Selection
        analysis_type = st.selectbox("Select Analysis Type", ["ANOVA", "Chi-Square Test", "Model 4 (Mediation)", "Model 7 (Moderated Mediation)", "Model 14 (Moderated Moderation)"])

        if analysis_type == "ANOVA":
            # ANOVA Section
            st.markdown("### ANOVA Analysis")

            dependent_var = st.selectbox("Select Dependent Variable (DV)", options=df.columns)
            independent_var = st.selectbox("Select Independent Variable (Factor)", options=[col for col in df.columns if col != dependent_var])

            # Add covariates
            covariates = st.multiselect("Select Covariates (Optional)", options=[col for col in df.columns if col not in [dependent_var, independent_var]])

            # Relabel categorical variable levels
            if df[independent_var].nunique() <= 2:
                st.markdown("### Label Categorical Levels")
                unique_levels = df[independent_var].unique()
                label_mapping = {}
                for level in unique_levels:
                    label_mapping[level] = st.text_input(f"Rename Level '{level}'", value=str(level))
                df[independent_var] = df[independent_var].replace(label_mapping)

            if st.button("Run ANOVA"):
                try:
                    # ANOVA formula with covariates
                    covariate_formula = " + ".join(covariates)
                    formula = f"{dependent_var} ~ C({independent_var})"
                    if covariate_formula:
                        formula += f" + {covariate_formula}"

                    model = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # Calculate group statistics
                    group_stats = df.groupby(independent_var)[dependent_var].agg(['mean', 'std']).reset_index()

                    # Display ANOVA table and group stats
                    st.markdown("### ANOVA Results")
                    st.write("**ANOVA Table**")
                    st.write(anova_table)

                    st.write("**Group Means and Standard Deviations**")
                    st.write(group_stats)

                    # Bar Chart
                    st.markdown("### Visualization: Group Means")
                    fig, ax = plt.subplots()
                    bars = group_stats.plot(kind='bar', x=independent_var, y='mean', yerr='std', ax=ax, legend=False, color='skyblue', capsize=4)
                    ax.set_title(f"Mean {dependent_var} by {independent_var}")
                    ax.set_xlabel(independent_var)
                    ax.set_ylabel(dependent_var)

                    # Annotate means on the bar chart
                    for bar in ax.patches:
                        ax.annotate(f"{bar.get_height():.2f}",
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    ha='center', va='bottom')

                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during ANOVA: {e}")

        elif analysis_type == "Chi-Square Test":
            st.markdown("### Chi-Square Test")

            row_variable = st.selectbox("Select Row Variable", options=df.columns)
            col_variable = st.selectbox("Select Column Variable", options=[col for col in df.columns if col != row_variable])

            if st.button("Run Chi-Square Test"):
                try:
                    # Create contingency table
                    contingency_table = pd.crosstab(df[row_variable], df[col_variable])
                    st.write("**Contingency Table**")
                    st.write(contingency_table)

                    # Perform Chi-Square Test
                    chi2, p, dof, expected = chi2_contingency(contingency_table)

                    st.markdown("### Chi-Square Test Results")
                    st.write(f"**Chi-Square Statistic**: {chi2:.2f}")
                    st.write(f"**Degrees of Freedom**: {dof}")
                    st.write(f"**p-value**: {p:.4f}")

                    # Bar Chart
                    st.markdown("### Visualization: Counts")
                    fig, ax = plt.subplots()
                    contingency_table.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'orange'], figsize=(8, 5))
                    ax.set_title(f"Counts by {row_variable} and {col_variable}")
                    ax.set_xlabel(row_variable)
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during Chi-Square Test: {e}")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
