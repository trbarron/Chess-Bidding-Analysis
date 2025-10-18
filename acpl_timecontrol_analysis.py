import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

def load_and_combine_data():
    """Load all CSV files from the data directory and combine them."""
    data_dir = Path("data")
    
    csv_files = list(data_dir.glob("**/*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"Found {len(csv_files)} CSV file(s):")
    
    dataframes = []
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def filter_and_group_time_controls(df):
    """Filter for time controls with format 'xx+0' (no increment) and group them."""
    filtered_df = df[df['TimeControl'].str.endswith('+0')].copy()
    
    filtered_df['TimeSeconds'] = filtered_df['TimeControl'].str.split('+').str[0].astype(int)
    
    def categorize_time_control(seconds):
        if seconds < 30:
            return "0-30s"
        elif seconds < 60:
            return "30s-1m"
        elif seconds < 180:
            return "1m-3m"
        elif seconds < 300:
            return "3m-5m"
        elif seconds < 600:
            return "5m-10m"
        elif seconds < 900:
            return "10m-15m"
        else:
            return "15m+"
    
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
    """Calculate weighted average ACPL_midgame for each time control group and rating bucket."""
    def weighted_avg_and_std(group):
        weights = group['NumMoves_midgame']
        values = group['ACPL_midgame']
        
        weighted_mean = np.average(values, weights=weights)
        variance = np.average((values - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(variance)
        
        return pd.Series({
            'MeanACPL': weighted_mean,
            'StdACPL': weighted_std,
            'Count': len(group),
            'TotalMoves': weights.sum()
        })
    
    averages = df.groupby(['TimeControlGroup', 'RatingBucket', 'RatingBucketLabel']).apply(
        weighted_avg_and_std, include_groups=False
    ).reset_index()
    
    bucket_order = ["0-30s", "30s-1m", "1m-3m", "3m-5m", "5m-10m", "10m-15m", "15m+"]
    time_controls = [tc for tc in bucket_order if tc in averages['TimeControlGroup'].unique()]
    averages['TimeGroupOrder'] = averages['TimeControlGroup'].map({group: i for i, group in enumerate(time_controls)})
    
    print(f"Calculated weighted averages for {len(averages)} combinations")
    print(f"Time control groups: {time_controls}")
    
    return averages

def create_visualization(averages_df):
    """Create line plot showing ACPL vs time controls by rating bucket."""
    plt.style.use('seaborn-v0_8')
    
    bucket_order = ["0-30s", "30s-1m", "1m-3m", "3m-5m", "5m-10m", "10m-15m", "15m+"]
    time_controls = [tc for tc in bucket_order if tc in averages_df['TimeControlGroup'].unique()]
    num_time_controls = len(time_controls)
    
    fig_width = max(14, 6 + num_time_controls * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    
    rating_buckets = sorted(averages_df['RatingBucket'].unique())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    for i, bucket in enumerate(rating_buckets):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('TimeGroupOrder')
        
        if len(bucket_data) > 0:
            label = bucket_data['RatingBucketLabel'].iloc[0]
            
            x_positions = bucket_data['TimeGroupOrder'].values
            
            ax.plot(x_positions, bucket_data['MeanACPL'], 
                   marker='o', linewidth=2, markersize=6, 
                   color=colors[i], label=f'Rating {label}', alpha=0.7)
            
            if len(bucket_data) >= 2:
                X = x_positions
                y = bucket_data['MeanACPL'].values
                
                reg = LinearRegression().fit(np.array(X).reshape(-1, 1), y)
                y_pred = reg.predict(np.array(X).reshape(-1, 1))
                
                r_squared = reg.score(np.array(X).reshape(-1, 1), y)
                
                correlation, p_value = stats.pearsonr(X, y)
                
                ax.plot(x_positions, y_pred, 
                       '--', linewidth=2, color=colors[i], alpha=0.8)
                
                if p_value < 0.05:
                    ax.text(0.02, 0.98 - i*0.08, f'{label}: R²={r_squared:.3f}', 
                           transform=ax.transAxes, fontsize=8, color=colors[i],
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Time Control', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Centipawn Loss (ACPL) - Midgame (Weighted)', fontsize=12, fontweight='bold')
    ax.set_title('ACPL vs Time Control by Player Rating Buckets\n(No Increment Games Only, Weighted by Number of Midgame Moves)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    ax.set_xticks(range(len(time_controls)))
    ax.set_xticklabels(time_controls, rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def print_summary_statistics(averages_df):
    """Print summary statistics about the analysis."""
    print("\n" + "="*60)
    
    print(f"\nTotal combinations analyzed: {len(averages_df)}")
    bucket_order = ["0-30s", "30s-1m", "1m-3m", "3m-5m", "5m-10m", "10m-15m", "15m+"]
    time_controls = [tc for tc in bucket_order if tc in averages_df['TimeControlGroup'].unique()]
    print(f"Time controls: {time_controls}")
    print(f"Rating buckets: {len(averages_df['RatingBucket'].unique())}")
    
    print(f"\nACPL Range: {averages_df['MeanACPL'].min():.2f} - {averages_df['MeanACPL'].max():.2f}")
    
    print(f"\nWeighted Average ACPL by Time Control:")
    time_control_summary = averages_df.groupby('TimeControlGroup').agg({
        'MeanACPL': ['mean', 'std'],
        'Count': 'sum',
        'TotalMoves': 'sum'
    })
    for time_group in time_controls:
        if time_group in time_control_summary.index:
            mean_acpl = time_control_summary.loc[time_group, ('MeanACPL', 'mean')]
            std_acpl = time_control_summary.loc[time_group, ('MeanACPL', 'std')]
            count = time_control_summary.loc[time_group, ('Count', 'sum')]
            total_moves = time_control_summary.loc[time_group, ('TotalMoves', 'sum')]
            print(f"  {time_group}: {mean_acpl:.2f} ± {std_acpl:.2f} (n={count}, moves={total_moves:.0f})")
    
    print(f"\nWeighted Average ACPL by Rating Bucket:")
    rating_summary = averages_df.groupby('RatingBucketLabel').agg({
        'MeanACPL': ['mean', 'std'],
        'Count': 'sum',
        'TotalMoves': 'sum'
    })
    for rating_bucket in sorted(rating_summary.index):
        mean_acpl = rating_summary.loc[rating_bucket, ('MeanACPL', 'mean')]
        std_acpl = rating_summary.loc[rating_bucket, ('MeanACPL', 'std')]
        count = rating_summary.loc[rating_bucket, ('Count', 'sum')]
        total_moves = rating_summary.loc[rating_bucket, ('TotalMoves', 'sum')]
        print(f"  {rating_bucket}: {mean_acpl:.2f} ± {std_acpl:.2f} (n={count}, moves={total_moves:.0f})")
    
    print(f"\nLinear Regression Analysis (ACPL vs Time Control):")
    print("-" * 60)
    for bucket in sorted(averages_df['RatingBucket'].unique()):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('TimeGroupOrder')
        if len(bucket_data) >= 2:
            X = bucket_data['TimeGroupOrder'].values
            y = bucket_data['MeanACPL'].values
            
            reg = LinearRegression().fit(np.array(X).reshape(-1, 1), y)
            r_squared = reg.score(np.array(X).reshape(-1, 1), y)
            correlation, p_value = stats.pearsonr(X, y)
            
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            label = bucket_data['RatingBucketLabel'].iloc[0]
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  {label}:")
            print(f"    Equation: ACPL = {slope:.4f} × TimeControl + {intercept:.2f}")
            print(f"    R² = {r_squared:.4f}, r = {correlation:.4f}, p = {p_value:.4f} {significance}")
            print(f"    Interpretation: {'Negative' if slope < 0 else 'Positive'} relationship")
            if p_value < 0.05:
                print(f"    → {'Longer time controls' if slope < 0 else 'Shorter time controls'} associated with {'lower' if slope < 0 else 'higher'} ACPL")
            else:
                print(f"    → No significant relationship between time control and ACPL")
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
    output_path = "./media/analysis_2.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    return averages

if __name__ == "__main__":
    results = main()
