
import streamlit as st

st.set_page_config(
    page_title="Brain Tumor ML Analytics",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor ML Analytics")
st.caption("Machine Learning Capstone Project")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Classification",
        "Regression",
        "Clustering",
        "Dataset Explorer",
    ],
)

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "An interactive dashboard for exploring "
        "brain-tumor machine learning experiments."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Tasks", "3")
    col2.metric("Dataset", "Segmentation Features")
    col3.metric("Status", "Setup")



elif page == "Classification":
    from pathlib import Path
    import joblib
    import pandas as pd

    st.header("Brain Tumor Classification")
    st.caption("Saved Logistic Regression model")

    project_dir = Path(__file__).parent
    model_path = project_dir / "models" / "final_logistic_regression_model.pkl"
    scaler_path = project_dir / "models" / "classification_scaler.pkl"
    dataset_path = project_dir / "generated_data" / "segmentation_features.csv"

    if not model_path.exists() or not scaler_path.exists():
        st.error("The saved classification model or scaler was not found.")
    elif not dataset_path.exists():
        st.error("The segmentation feature dataset was not found.")
    else:
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            df = pd.read_csv(dataset_path)

            if model.n_features_in_ != scaler.n_features_in_:
                st.error("The model and scaler expect different numbers of features.")
                st.stop()
            
            st.write("Model classes:", model.classes_)
            st.write("Dataset columns:", df.columns.tolist())
            st.write(
                "Model expected features:",
                getattr(model, "n_features_in_", "Unknown")
            )
            st.write(
                "Scaler expected features:",
                getattr(scaler, "n_features_in_", "Unknown")
            )

            if hasattr(model, "feature_names_in_"):
                st.write(
                    "Model training feature order:",
                    model.feature_names_in_.tolist()
                )

            if hasattr(scaler, "feature_names_in_"):
                st.write(
                    "Scaler training feature order:",
                    scaler.feature_names_in_.tolist()
                )


            feature_columns = (
                scaler.feature_names_in_.tolist()
                if hasattr(scaler, "feature_names_in_")
                else [
                    col for col in df.columns
                    if col not in ["image_name", "tumor_type"]
                ]
            )

            missing_features = [
                col for col in feature_columns
                if col not in df.columns
            ]

            if missing_features:
                st.error(f"Dataset is missing required features: {missing_features}")
                st.stop()
            
            st.write("Choose a row from the existing feature dataset.")

            row_index = st.number_input(
                "Dataset row index",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1,
            )

            selected_row = df.iloc[int(row_index)]
            input_df = pd.DataFrame(
                [selected_row[feature_columns].to_dict()],
                columns=feature_columns,
            )

            st.subheader("Input features")
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

            if st.button("Predict tumor type"):
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]

                st.success(f"Predicted class: {prediction}")

                try:
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(scaled_input)[0]

                        probability_df = pd.DataFrame({
                            "Tumor class": model.classes_,
                            "Probability (%)": (
                                probabilities * 100
                            ).round(2),
                        }).sort_values(
                            "Probability (%)",
                            ascending=False,
                        )

                        st.subheader("Class probabilities")
                        st.dataframe(
                            probability_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.bar_chart(
                            probability_df.set_index("Tumor class")
                        )
                except Exception as probability_error:
                    st.warning(
                        "The class prediction succeeded, but probability "
                        f"display failed: {probability_error}"
                    )

                if "tumor_type" in df.columns:
                    st.write(
                        "Existing dataset label:",
                        selected_row["tumor_type"],
                    )

                st.caption(
                    "Experimental ML output for a dataset row, "
                    "not a medical diagnosis."
                )

        except Exception as exc:
            st.error(f"Classification error: {exc}")




elif page == "Regression":
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd

    st.header("Tumor Area Prediction")
    st.caption(
        "Predict tumor area using your trained Polynomial Regression model. "
        "This is an experimental ML output, not a medical diagnosis."
    )

    project_dir = Path(__file__).parent
    model_path = (
        project_dir / "models" /
        "final_polynomial_regression_model.pkl"
    )
    scaler_path = project_dir / "models" / "regression_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        st.error(
            "Regression model or scaler not found. "
            "Check the files in your project's models folder."
        )
    else:
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            st.subheader("Enter the tumor features")

            with st.form("regression_prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    tumor_width = st.number_input(
                        "Tumor width",
                        min_value=0.0,
                        value=88.0,
                        step=1.0,
                    )
                    tumor_height = st.number_input(
                        "Tumor height",
                        min_value=0.0,
                        value=85.0,
                        step=1.0,
                    )
                    aspect_ratio = st.number_input(
                        "Aspect ratio",
                        min_value=0.0,
                        value=1.0353,
                        step=0.01,
                        format="%.4f",
                    )
                    mean_intensity = st.number_input(
                        "Mean intensity",
                        min_value=0.0,
                        value=118.2617,
                        step=1.0,
                    )

                with col2:
                    std_intensity = st.number_input(
                        "Standard deviation of intensity",
                        min_value=0.0,
                        value=63.2196,
                        step=1.0,
                    )
                    min_intensity = st.number_input(
                        "Minimum intensity",
                        min_value=0.0,
                        value=1.0,
                        step=1.0,
                    )
                    max_intensity = st.number_input(
                        "Maximum intensity",
                        min_value=0.0,
                        value=242.0,
                        step=1.0,
                    )

                submitted = st.form_submit_button(
                    "Predict tumor area",
                    use_container_width=True,
                )

            if submitted:
                feature_names = [
                    "tumor_width",
                    "tumor_height",
                    "aspect_ratio",
                    "mean_intensity",
                    "std_intensity",
                    "min_intensity",
                    "max_intensity",
                ]

                input_df = pd.DataFrame(
                    [[
                        tumor_width,
                        tumor_height,
                        aspect_ratio,
                        mean_intensity,
                        std_intensity,
                        min_intensity,
                        max_intensity,
                    ]],
                    columns=feature_names,
                )

                try:
                    scaled_input = scaler.transform(input_df)
                    prediction = float(
                        np.asarray(model.predict(scaled_input)).ravel()[0]
                    )

                    st.subheader("Prediction result")
                    st.metric(
                        "Predicted tumor area",
                        f"{prediction:,.2f}",
                    )

                    if prediction < 0:
                        st.warning(
                            "The model returned a negative area. "
                            "Treat this as an unreliable prediction and "
                            "review the model and input features."
                        )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        except Exception as exc:
            st.error(f"Could not load the regression model: {exc}")



elif page == "Clustering":
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )

    st.header("Tumor Feature Clustering")
    st.caption(
        "Explore unsupervised groups in the extracted tumor features. "
        "Clusters are patterns in the data, not tumor diagnoses."
    )

    project_dir = Path(__file__).parent
    dataset_path = (
        project_dir / "generated_data" / "segmentation_features.csv"
    )

    if not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")
    else:
        try:
            df = pd.read_csv(dataset_path)

            excluded_columns = ["image_name", "tumor_type"]
            feature_columns = [
                col for col in df.columns
                if col not in excluded_columns
                and pd.api.types.is_numeric_dtype(df[col])
            ]

            if len(feature_columns) < 2:
                st.error("At least two numeric features are required.")
            else:
                X = df[feature_columns].replace(
                    [np.inf, -np.inf], np.nan
                )
                valid_rows = X.notna().all(axis=1)
                X = X.loc[valid_rows].copy()
                valid_df = df.loc[valid_rows].copy()

                if len(X) < 3:
                    st.error("Not enough valid rows for clustering.")
                else:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    st.subheader("Dataset overview")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Valid observations", len(X))
                    col2.metric("Numeric features", len(feature_columns))
                    col3.metric("Missing/invalid rows removed",
                                len(df) - len(X))

                    st.write("Features used:")
                    st.code(", ".join(feature_columns))

                    max_clusters = min(10, len(X) - 1)
                    k = st.slider(
                        "Number of clusters",
                        min_value=2,
                        max_value=max_clusters,
                        value=min(4, max_clusters),
                    )

                    algorithm = st.selectbox(
                        "Clustering algorithm",
                        [
                            "K-Means",
                            "Agglomerative — Ward",
                            "Agglomerative — Complete",
                            "Agglomerative — Average",
                        ],
                    )

                    if algorithm == "K-Means":
                        cluster_model = KMeans(
                            n_clusters=k,
                            random_state=42,
                            n_init=10,
                        )
                    else:
                        linkage_method = algorithm.split("—")[1].strip().lower()
                        cluster_model = AgglomerativeClustering(
                            n_clusters=k,
                            linkage=linkage_method,
                        )

                    labels = cluster_model.fit_predict(X_scaled)

                    silhouette = silhouette_score(X_scaled, labels)
                    db_score = davies_bouldin_score(X_scaled, labels)
                    ch_score = calinski_harabasz_score(X_scaled, labels)

                    st.subheader("Clustering evaluation")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Silhouette score", f"{silhouette:.4f}")
                    c2.metric("Davies–Bouldin score", f"{db_score:.4f}")
                    c3.metric("Calinski–Harabasz score", f"{ch_score:.2f}")

                    st.caption(
                        "Higher Silhouette and Calinski–Harabasz scores "
                        "are generally better; lower Davies–Bouldin is better. "
                        "Compare these metrics together."
                    )

                    result_df = valid_df.copy()
                    result_df["Cluster"] = labels.astype(str)

                    st.subheader("Cluster sizes")
                    cluster_sizes = (
                        result_df["Cluster"]
                        .value_counts()
                        .sort_index()
                        .rename_axis("Cluster")
                        .reset_index(name="Observations")
                    )
                    st.dataframe(
                        cluster_sizes,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("PCA visualization")
                    pca = PCA(n_components=2, random_state=42)
                    X_pca = pca.fit_transform(X_scaled)

                    plot_df = pd.DataFrame({
                        "PC1": X_pca[:, 0],
                        "PC2": X_pca[:, 1],
                        "Cluster": labels.astype(str),
                    })

                    fig, ax = plt.subplots(figsize=(9, 5))
                    for cluster_name in sorted(
                        plot_df["Cluster"].unique(),
                        key=int,
                    ):
                        points = plot_df[
                            plot_df["Cluster"] == cluster_name
                        ]
                        ax.scatter(
                            points["PC1"],
                            points["PC2"],
                            label=f"Cluster {cluster_name}",
                            alpha=0.7,
                            s=25,
                        )

                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_title(f"{algorithm} (k={k})")
                    ax.legend()
                    ax.grid(alpha=0.2)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.write(
                        "Variance explained by the two PCA components: "
                        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
                    )

                    st.subheader("Cluster assignments")
                    st.dataframe(
                        result_df[
                            ["image_name", "tumor_type"] + feature_columns
                            + ["Cluster"]
                        ] if "image_name" in result_df.columns
                        and "tumor_type" in result_df.columns
                        else result_df[feature_columns + ["Cluster"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    if "tumor_type" in valid_df.columns:
                        with st.expander(
                            "Compare clusters with existing dataset labels"
                        ):
                            st.caption(
                                "The labels below are used only for "
                                "post-clustering interpretation; they are "
                                "not included in model fitting."
                            )
                            cross_tab = pd.crosstab(
                                valid_df["tumor_type"],
                                labels,
                                rownames=["Existing tumor label"],
                                colnames=["Cluster"],
                            )
                            st.dataframe(
                                cross_tab,
                                use_container_width=True,
                            )

        except Exception as exc:
            st.error(f"Clustering failed: {exc}")



elif page == "Dataset Explorer":
    st.header("Dataset Explorer")

    from pathlib import Path
    import pandas as pd

    dataset_path = (
        Path(__file__).parent
        / "generated_data"
        / "segmentation_features.csv"
    )

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)

        st.success("Dataset loaded successfully!")

        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", f"{df.shape[1]:,}")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Column Information")
        st.write(
            pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
            })
        )

    else:
        st.error(
            f"Dataset not found at: {dataset_path}"
        )
        st.info(
            "Check that generated_data/segmentation_features.csv "
            "exists in your project folder."
        )

