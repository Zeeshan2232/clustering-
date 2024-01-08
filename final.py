import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def plot(file):
    # Read the data
    df = pd.read_csv(file)

    # Select features for clustering
    features_for_clustering = df[['Oil', 'Gas', 'Coal']]

    # Normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_for_clustering)

    # Elbow Method to find Optimal k
    inertia = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_features)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig('Elbow Method.png', dpi=300)
    plt.show()

    # Applying k-means clustering
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(normalized_features)

    # 3-D Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Oil'], df['Coal'], df['Gas'], c=df['Cluster'], cmap='viridis', s=10)
    ax.set_xlabel('Oil Consumption')
    ax.set_ylabel('Coal Consumption')
    ax.set_zlabel('Gas Consumption')
    ax.set_title('KMeans Clustering Results - 3D Visualization')
    plt.savefig('3-D Visualization of Clustering.png', dpi=300)
    plt.show()

    # Model Fitting
    selected_country = 'United Kingdom'
    country_data = df[df['Country'] == selected_country]

    years = country_data['Year']
    hydro_values = country_data['Oil']

    def exponential_growth(t, a, b):
        return a * np.exp(b * (t - years.min()))

    initial_guess = [1.0, 0.01]
    params, covariance = curve_fit(exponential_growth, years, hydro_values, p0=initial_guess)

    predicted_values = exponential_growth(years, *params)

    plt.scatter(years, hydro_values, label='Actual Data')
    plt.plot(years, predicted_values, color='red', label='Fitted Curve')
    plt.xlabel('Year')
    plt.ylabel('Coal Consumption')
    plt.title(f'Exponential Growth Model Fitting for {selected_country}')
    plt.legend()
    plt.savefig('Exponential Growth.png', dpi=300)
    plt.show()

    print("Fitted Parameters (a, b):", params)
    err_ranges = np.sqrt(np.diag(covariance))
    lower_bound = params - 1.96 * err_ranges
    upper_bound = params + 1.96 * err_ranges

    print("Confidence Intervals (95%):")
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)

    # Comparative Analysis
    features_for_clustering = ['Oil', 'Gas', 'Coal']

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features_for_clustering])

    selected_countries = pd.DataFrame(columns=df.columns)
    selected_set = set()

    for cluster in df['Cluster'].unique():
        available_countries = df[(df['Cluster'] == cluster) & (~df['Country'].isin(selected_set))]
        if not available_countries.empty:
            selected_country = available_countries.iloc[0]
            selected_set.add(selected_country['Country'])
            selected_countries = selected_countries.append(selected_country)

    fig, axes = plt.subplots(nrows=1, ncols=len(selected_countries), figsize=(15, 5), sharey=True)

    def exponential_growth(t, a, b, t_min):
        return a * np.exp(b * (t - t_min))

    for ax, (index, country_row) in zip(axes, selected_countries.iterrows()):
        selected_country = country_row['Country']
        cluster = country_row['Cluster']

        country_data = df[df['Country'] == selected_country]
        years = country_data['Year']
        oil_values = country_data['Oil']

        params, _ = curve_fit(exponential_growth, years, oil_values, p0=[1.0, 0.01, years.min()])
        predicted_values = exponential_growth(years, *params)

        ax.plot(years, oil_values, 'o-', label='Actual Data')
        ax.plot(years, predicted_values, label=f'Fitted Curve - {selected_country}')
        ax.set_xlabel('Year')
        ax.set_title(f'Cluster {cluster}')

    axes[0].set_ylabel('Oil Consumption')
    fig.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig('Comparative Analysis.png', dpi=300)
    plt.show()

    # Prediction For Future Years
    features_for_clustering = df[['Oil', 'Gas', 'Coal']]

    # Analyze Cluster Centers
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Centers:")
    print(cluster_centers)

    # Visualization: 3D Scatter Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['Gas'], df['Oil'], df['Coal'], c=df['Cluster'], cmap='viridis', s=10, label='Data Points')
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', marker='X', s=200, label='Cluster Centers')

    ax.set_xlabel('Gas Consumption')
    ax.set_ylabel('Oil Consumption')
    ax.set_zlabel('Coal Consumption')
    ax.set_title('3D Scatter Plot with Cluster Centers')
    ax.legend()

    plt.savefig('Cluster Center.png', dpi=300)
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[features_for_clustering])

    selected_countries = pd.DataFrame(columns=df.columns)
    selected_set = set()

    for cluster in df['Cluster'].unique():
        available_countries = df[(df['Cluster'] == cluster) & (~df['Country'].isin(selected_set))]
        if not available_countries.empty:
            selected_country = available_countries.iloc[0]
            selected_set.add(selected_country['Country'])
            selected_countries = selected_countries.append(selected_country)

    fig, axes = plt.subplots(nrows=1, ncols=len(selected_countries), figsize=(15, 5), sharey=True)

    for ax, (index, country_row) in zip(axes, selected_countries.iterrows()):
        selected_country = country_row['Country']
        cluster = country_row['Cluster']
        country_data = df[df['Country'] == selected_country]
        years = country_data['Year']
        oil_values = country_data['Oil']

        params, _ = curve_fit(exponential_growth, years, oil_values, p0=[1.0, 0.01, years.min()])
        future_years = np.arange(years.min(), years.max() + 10)
        future_predictions = exponential_growth(future_years, *params)

        ax.plot(years, oil_values, 'o-', label='Actual Data')
        ax.plot(years, exponential_growth(years, *params), label=f'Fitted Curve - {selected_country}')
        ax.plot(future_years, future_predictions, label=f'Predictions - {selected_country}', linestyle='--')
        ax.set_xlabel('Year')
        ax.set_title(f'Cluster {cluster}')

    axes[0].set_ylabel('Oil Consumption')
    fig.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig('Prediction for Future Years.png', dpi=300)
    plt.show()


plot('final_1.csv')
