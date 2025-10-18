import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

def load_and_combine_data():
    """Load both CSV files and combine them."""
    data_dir = Path("data")
    
    # Load both datasets
    # df_2013 = pd.read_csv(data_dir / "lichess_db_standard_rated_2013-01_metrics.csv")
    df_2025_1 = pd.read_csv(data_dir / "lichess_db_standard_rated_2025-08_metrics.csv")
    df_2025_2 = pd.read_csv(data_dir / "lichess_db_standard_rated_2025-06_metrics.csv")
    # Add year column to distinguish datasets
    # df_2013['Year'] = 2013
    df_2025_1['Year'] = 2025
    df_2025_2['Year'] = 2025
    # Combine datasets
    # combined_df = pd.concat([df_2013, df_2020], ignore_index=True)
    
    # print(f"Combined dataset shape: {combined_df.shape}")
    # print(f"2013 data: {df_2013.shape[0]} rows")
    print(f"2025_1 data: {df_2025_1.shape[0]} rows")
    print(f"2025_2 data: {df_2025_2.shape[0]} rows")
    return pd.concat([df_2025_1, df_2025_2], ignore_index=True)

def filter_and_group_time_controls(df):
    """Filter for time controls with format 'xx+0' (no increment) and group them."""
    # Filter for time controls ending with '+0'
    filtered_df = df[df['TimeControl'].str.endswith('+0')].copy()
    
    # Convert time control to numeric
    filtered_df['TimeSeconds'] = filtered_df['TimeControl'].str.split('+').str[0].astype(int)
    
    # Create time control groups
    def categorize_time_control(seconds):
        if seconds <= 180:  # 0-3 minutes
            return "0-3 minutes"
        elif seconds <= 540:  # 3-9 minutes
            return "3-9 minutes"
        elif seconds <= 900:  # 9-15 minutes
            return "9-15 minutes"
        else:  # 15+ minutes
            return "15+ minutes"
    
    filtered_df['TimeControlGroup'] = filtered_df['TimeSeconds'].apply(categorize_time_control)
    
    print(f"After filtering for '+0' time controls: {filtered_df.shape[0]} rows")
    print(f"Unique time controls: {sorted(filtered_df['TimeControl'].unique())}")
    print(f"Time control groups: {sorted(filtered_df['TimeControlGroup'].unique())}")
    
    return filtered_df

def create_rating_buckets(df):
    """Create rating buckets of 200 points each."""
    # Create rating buckets
    df['RatingBucket'] = (df['Rating'] // 200) * 200
    
    # Create readable bucket labels
    df['RatingBucketLabel'] = df['RatingBucket'].astype(str) + '-' + (df['RatingBucket'] + 199).astype(str)
    
    print(f"Rating range: {df['Rating'].min()} - {df['Rating'].max()}")
    print(f"Rating buckets: {sorted(df['RatingBucket'].unique())}")
    
    return df

def calculate_averages(df):
    """Calculate average ACPL_midgame for each time control group and rating bucket."""
    # Group by time control group and rating bucket, calculate mean ACPL
    averages = df.groupby(['TimeControlGroup', 'RatingBucket', 'RatingBucketLabel'])['ACPL_midgame'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    # Rename columns for clarity
    averages.columns = ['TimeControlGroup', 'RatingBucket', 'RatingBucketLabel', 'MeanACPL', 'StdACPL', 'Count']
    
    # Create order for time control groups
    time_group_order = ["0-3 minutes", "3-9 minutes", "9-15 minutes", "15+ minutes"]
    averages['TimeGroupOrder'] = averages['TimeControlGroup'].map({group: i for i, group in enumerate(time_group_order)})
    
    print(f"Calculated averages for {len(averages)} combinations")
    print(f"Time control groups: {sorted(averages['TimeControlGroup'].unique())}")
    
    return averages

def create_visualization(averages_df):
    """Create line plot showing ACPL vs time controls by rating bucket."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get unique rating buckets and sort them
    rating_buckets = sorted(averages_df['RatingBucket'].unique())
    
    # Define colors for different rating buckets
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    # Plot lines for each rating bucket
    for i, bucket in enumerate(rating_buckets):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('TimeGroupOrder')
        
        if len(bucket_data) > 0:
            # Get the label for this bucket
            label = bucket_data['RatingBucketLabel'].iloc[0]
            
            # Create x-axis positions for time control groups
            x_positions = range(len(bucket_data))
            
            # Plot the data points
            ax.plot(x_positions, bucket_data['MeanACPL'], 
                   marker='o', linewidth=2, markersize=6, 
                   color=colors[i], label=f'Rating {label}', alpha=0.7)
            
            # Add linear fit if we have enough data points
            if len(bucket_data) >= 2:
                # Prepare data for linear regression
                X = x_positions
                y = bucket_data['MeanACPL'].values
                
                # Fit linear regression
                reg = LinearRegression().fit(np.array(X).reshape(-1, 1), y)
                y_pred = reg.predict(np.array(X).reshape(-1, 1))
                
                # Calculate R-squared
                r_squared = reg.score(np.array(X).reshape(-1, 1), y)
                
                # Calculate correlation coefficient
                correlation, p_value = stats.pearsonr(X, y)
                
                # Plot the linear fit line
                ax.plot(x_positions, y_pred, 
                       '--', linewidth=2, color=colors[i], alpha=0.8)
                
                # Add R-squared to legend if correlation is significant
                if p_value < 0.05:  # Significant correlation
                    ax.text(0.02, 0.98 - i*0.08, f'{label}: R²={r_squared:.3f}', 
                           transform=ax.transAxes, fontsize=8, color=colors[i],
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel('Time Control Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Centipawn Loss (ACPL) - Midgame', fontsize=12, fontweight='bold')
    ax.set_title('ACPL vs Time Control Groups by Player Rating Buckets\n(No Increment Games Only, Only Midgame Moves)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set x-axis to show time control groups
    time_groups = ["0-3 minutes", "3-9 minutes", "9-15 minutes", "15+ minutes"]
    ax.set_xticks(range(len(time_groups)))
    ax.set_xticklabels(time_groups, rotation=45, ha='right')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    return fig

def print_summary_statistics(averages_df):
    """Print summary statistics about the analysis."""
    print("\n" + "="*60)
    
    print(f"\nTotal combinations analyzed: {len(averages_df)}")
    print(f"Time control groups: {sorted(averages_df['TimeControlGroup'].unique())}")
    print(f"Rating buckets: {len(averages_df['RatingBucket'].unique())}")
    
    print(f"\nACPL Range: {averages_df['MeanACPL'].min():.2f} - {averages_df['MeanACPL'].max():.2f}")
    
    # Show data for each time control group
    print(f"\nAverage ACPL by Time Control Group:")
    time_control_summary = averages_df.groupby('TimeControlGroup')['MeanACPL'].agg(['mean', 'std', 'count'])
    for time_group in ["0-3 minutes", "3-9 minutes", "9-15 minutes", "15+ minutes"]:
        if time_group in time_control_summary.index:
            mean_acpl = time_control_summary.loc[time_group, 'mean']
            std_acpl = time_control_summary.loc[time_group, 'std']
            count = time_control_summary.loc[time_group, 'count']
            print(f"  {time_group}: {mean_acpl:.2f} ± {std_acpl:.2f} (n={count})")
    
    # Show data for each rating bucket
    print(f"\nAverage ACPL by Rating Bucket:")
    rating_summary = averages_df.groupby('RatingBucketLabel')['MeanACPL'].agg(['mean', 'std', 'count'])
    for rating_bucket in sorted(rating_summary.index):
        mean_acpl = rating_summary.loc[rating_bucket, 'mean']
        std_acpl = rating_summary.loc[rating_bucket, 'std']
        count = rating_summary.loc[rating_bucket, 'count']
        print(f"  {rating_bucket}: {mean_acpl:.2f} ± {std_acpl:.2f} (n={count})")
    
    # Calculate and display linear regression statistics for each rating bucket
    print(f"\nLinear Regression Analysis (ACPL vs Time Control Groups):")
    print("-" * 60)
    for bucket in sorted(averages_df['RatingBucket'].unique()):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('TimeGroupOrder')
        if len(bucket_data) >= 2:
            # Calculate linear regression
            X = range(len(bucket_data))
            y = bucket_data['MeanACPL'].values
            
            reg = LinearRegression().fit(np.array(X).reshape(-1, 1), y)
            r_squared = reg.score(np.array(X).reshape(-1, 1), y)
            correlation, p_value = stats.pearsonr(X, y)
            
            # Get slope and intercept
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            label = bucket_data['RatingBucketLabel'].iloc[0]
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  {label}:")
            print(f"    Equation: ACPL = {slope:.4f} × TimeGroup + {intercept:.2f}")
            print(f"    R² = {r_squared:.4f}, r = {correlation:.4f}, p = {p_value:.4f} {significance}")
            print(f"    Interpretation: {'Negative' if slope < 0 else 'Positive'} relationship")
            if p_value < 0.05:
                print(f"    → {'Longer time groups' if slope < 0 else 'Shorter time groups'} associated with {'lower' if slope < 0 else 'higher'} ACPL")
            else:
                print(f"    → No significant relationship between time groups and ACPL")
            print()

def main():
    """Main analysis function."""
    print("Chess ACPL vs Time Control Analysis")
    print("="*50)
    
    # Load and combine data
    df = load_and_combine_data()
    
    # Filter for no-increment time controls and group them
    df_filtered = filter_and_group_time_controls(df)
    
    # Create rating buckets
    df_bucketed = create_rating_buckets(df_filtered)
    
    # Calculate averages
    averages = calculate_averages(df_bucketed)
    
    # Print summary statistics
    print_summary_statistics(averages)
    
    # Create visualization
    fig = create_visualization(averages)
    
    # Save the plot
    output_path = "acpl_vs_timecontrol_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    return averages

if __name__ == "__main__":
    results = main()
