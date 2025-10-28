import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    """Filter and identify specific time controls with sufficient data (no increment, >= 60 seconds)."""
    # Parse time control format (e.g., "300+0", "600+5", "1800+0")
    time_control_parts = df['TimeControl'].str.split('+')
    
    # Handle edge cases where time control might be '-' or invalid
    df['StartingTime'] = pd.to_numeric(time_control_parts.str[0], errors='coerce')
    df['Increment'] = pd.to_numeric(time_control_parts.str[1].fillna('0'), errors='coerce')
    
    # Filter out rows with invalid time controls
    valid_time_controls = df['StartingTime'].notna() & df['Increment'].notna()
    df = df[valid_time_controls].copy()
    
    print(f"Total games before filtering: {df.shape[0]:,} rows")
    total_valid = df.shape[0]
    
    # Filter to only games with 0 increment
    df = df[df['Increment'] == 0].copy()
    
    print(f"Games with 0 increment: {df.shape[0]:,} rows ({df.shape[0]/total_valid*100:.1f}% of valid games)")
    
    # Filter to only games with starting time >= 60 seconds (1 minute)
    df = df[df['StartingTime'] >= 60].copy()
    
    print(f"Games with time control >= 60+0: {df.shape[0]:,} rows ({df.shape[0]/total_valid*100:.1f}% of valid games)")
    
    # For no-increment games, time in seconds equals starting time
    df['TotalTimeSeconds'] = df['StartingTime']
    df['TotalTimeMinutes'] = df['TotalTimeSeconds'] / 60
    
    # Group all time controls >= 30 minutes as "Classical"
    df['TimeControlSpecific'] = df.apply(
        lambda row: 'Classical (≥30min)' if row['TotalTimeMinutes'] >= 30 
        else f"{int(row['StartingTime'])}+0", 
        axis=1
    )
    
    print(f"Total games analyzed: {df.shape[0]:,} rows")
    
    return df

def create_rating_buckets(df):
    """Create rating buckets of 200 points each."""
    # Create rating buckets
    df['RatingBucket'] = (df['Rating'] // 200) * 200
    
    # Create readable bucket labels
    df['RatingBucketLabel'] = df['RatingBucket'].astype(str) + '-' + (df['RatingBucket'] + 199).astype(str)
    
    return df

def filter_time_controls_by_threshold(df, min_games_per_bucket=100000):
    """Filter to only include time controls with sufficient games per rating bucket."""
    # Count games per time control per rating bucket
    counts = df.groupby(['TimeControlSpecific', 'RatingBucket']).size().reset_index(name='GameCount')
    
    # For each time control, check if it meets the threshold in at least one rating bucket
    time_control_valid = counts.groupby('TimeControlSpecific')['GameCount'].max() >= min_games_per_bucket
    valid_time_controls = time_control_valid[time_control_valid].index.tolist()
    
    print(f"\nTime controls with at least {min_games_per_bucket:,} games in at least one rating bucket:")
    
    # Get detailed stats for valid time controls
    for tc in sorted(valid_time_controls, key=lambda x: (x == 'Classical (≥30min)', 
                                                          float(x.split('+')[0]) if '+' in x else float('inf'))):
        tc_data = df[df['TimeControlSpecific'] == tc]
        total_games = len(tc_data)
        avg_time = tc_data['TotalTimeMinutes'].median()
        
        # Show count per rating bucket
        bucket_counts = counts[counts['TimeControlSpecific'] == tc].sort_values('RatingBucket')
        print(f"\n  {tc} (median: {avg_time:.1f}min, total: {total_games:,} games)")
        for _, row in bucket_counts.iterrows():
            bucket_label = f"{int(row['RatingBucket'])}-{int(row['RatingBucket'])+199}"
            status = "✓" if row['GameCount'] >= min_games_per_bucket else "✗"
            print(f"    {bucket_label}: {row['GameCount']:,} games {status}")
    
    # Filter dataframe to only include valid time controls
    df_filtered = df[df['TimeControlSpecific'].isin(valid_time_controls)].copy()
    
    print(f"\nFiltered to {len(df_filtered):,} games across {len(valid_time_controls)} time controls")
    
    return df_filtered

def calculate_averages(df):
    """Calculate weighted average ACPL_midgame for each specific time control and rating bucket."""
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
    
    averages = df.groupby(['TimeControlSpecific', 'RatingBucket', 'RatingBucketLabel']).apply(
        weighted_avg_and_std, include_groups=False
    ).reset_index()
    
    # Add median time for each time control for sorting and display
    time_control_medians = df.groupby('TimeControlSpecific')['TotalTimeMinutes'].median().to_dict()
    averages['MedianTimeMinutes'] = averages['TimeControlSpecific'].map(time_control_medians)
    
    # Sort time controls by median time
    averages = averages.sort_values('MedianTimeMinutes')
    
    # Create ordering for x-axis
    unique_tcs = averages.sort_values('MedianTimeMinutes')['TimeControlSpecific'].unique()
    tc_order_map = {tc: i for i, tc in enumerate(unique_tcs)}
    averages['TimeControlOrder'] = averages['TimeControlSpecific'].map(tc_order_map)
    
    return averages

def create_visualization(averages_df):
    """Create line plot showing ACPL vs specific time controls by rating bucket."""
    plt.style.use('seaborn-v0_8')
    
    # Get unique time controls sorted by median time
    time_controls = averages_df.sort_values('MedianTimeMinutes')['TimeControlSpecific'].unique()
    num_time_controls = len(time_controls)
    
    fig_width = max(14, 4 + num_time_controls * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    
    rating_buckets = sorted(averages_df['RatingBucket'].unique())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    # Get median times for each time control
    time_control_medians = averages_df.groupby('TimeControlSpecific')['MedianTimeMinutes'].first().to_dict()
    
    for i, bucket in enumerate(rating_buckets):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('TimeControlOrder')
        
        if len(bucket_data) > 0:
            label = bucket_data['RatingBucketLabel'].iloc[0]
            
            # Use median time values for x-axis positioning
            x_positions = bucket_data['MedianTimeMinutes'].values
            
            ax.plot(x_positions, bucket_data['MeanACPL'], 
                   marker='o', linewidth=2, markersize=8, 
                   color=colors[i], label=f'Rating {label}', alpha=0.75)
            
            # Add logarithmic regression line if we have enough data points
            if len(bucket_data) >= 2:
                X = x_positions
                y = bucket_data['MeanACPL'].values
                
                # Use log transformation for x values
                X_log = np.log(X).reshape(-1, 1)
                
                reg = LinearRegression().fit(X_log, y)
                y_pred = reg.predict(X_log)
                
                r_squared = reg.score(X_log, y)
                
                # Calculate correlation between log(X) and y
                correlation, p_value = stats.pearsonr(np.log(X), y)
                
                ax.plot(x_positions, y_pred, 
                       '--', linewidth=1.5, color=colors[i], alpha=0.5)
                
                # Display R² for significant correlations
                if p_value < 0.05:
                    ax.text(0.02, 0.98 - i*0.06, f'{label}: R²={r_squared:.3f}', 
                           transform=ax.transAxes, fontsize=8, color=colors[i],
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Time Control (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Centipawn Loss (ACPL) - Midgame (Weighted)', fontsize=12, fontweight='bold')
    ax.set_title('ACPL vs Time Controls (0 Increment, >= 60+0) by Player Rating', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Create x-axis labels
    x_ticks = []
    x_labels = []
    for tc in time_controls:
        median_time = time_control_medians[tc]
        x_ticks.append(median_time)
        
        # Format label based on time control type
        if tc == 'Classical (≥30min)':
            x_labels.append(f'Classical\n(≥30min)')
        else:
            # Extract base time and increment from format like "180+0"
            x_labels.append(f'{tc}\n({median_time:.1f}min)')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 150)
    plt.tight_layout()
    
    return fig

def print_summary_statistics(averages_df):
    """Print summary statistics about the analysis."""
    print("\n" + "="*60)
    
    print(f"\nTotal combinations analyzed: {len(averages_df)}")
    print(f"ACPL Range: {averages_df['MeanACPL'].min():.2f} - {averages_df['MeanACPL'].max():.2f}")
    
    print(f"\nWeighted Average ACPL by Time Control:")
    time_control_summary = averages_df.groupby('TimeControlSpecific').agg({
        'MeanACPL': ['mean', 'std'],
        'Count': 'sum',
        'TotalMoves': 'sum',
        'MedianTimeMinutes': 'first'
    }).sort_values(('MedianTimeMinutes', 'first'))
    
    for time_control in time_control_summary.index:
        mean_acpl = time_control_summary.loc[time_control, ('MeanACPL', 'mean')]
        std_acpl = time_control_summary.loc[time_control, ('MeanACPL', 'std')]
        count = time_control_summary.loc[time_control, ('Count', 'sum')]
        total_moves = time_control_summary.loc[time_control, ('TotalMoves', 'sum')]
        median_time = time_control_summary.loc[time_control, ('MedianTimeMinutes', 'first')]
        print(f"  {time_control} ({median_time:.1f}min): {mean_acpl:.2f} ± {std_acpl:.2f} (n={count:,}, moves={total_moves:.0f})")
    
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
        print(f"  {rating_bucket}: {mean_acpl:.2f} ± {std_acpl:.2f} (n={count:,}, moves={total_moves:.0f})")
    
    print(f"\nLogarithmic Regression Analysis (Time vs ACPL):")
    print("-" * 40)
    for bucket in sorted(averages_df['RatingBucket'].unique()):
        bucket_data = averages_df[averages_df['RatingBucket'] == bucket].sort_values('MedianTimeMinutes')
        if len(bucket_data) >= 2:
            X = bucket_data['MedianTimeMinutes'].values
            y = bucket_data['MeanACPL'].values
            
            # Use log transformation for x values
            X_log = np.log(X).reshape(-1, 1)
            
            reg = LinearRegression().fit(X_log, y)
            r_squared = reg.score(X_log, y)
            correlation, p_value = stats.pearsonr(np.log(X), y)
            
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            label = bucket_data['RatingBucketLabel'].iloc[0]
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"  {label}: R²={r_squared:.3f}, p={p_value:.3f} {significance}")
            if p_value < 0.05:
                direction = "decreases" if slope < 0 else "increases" if slope > 0 else "no trend"
                print(f"    → Log(time) {direction} ACPL (slope={slope:.3f} ACPL/log(min))")
            print()

def main(min_games_per_bucket=100000):
    """Main analysis function."""
    print("Chess ACPL vs Specific Time Control Analysis (0 Increment, >= 60+0)")
    print("="*60)
        
    # Load and combine data
    df = load_and_combine_data()
    
    # Parse and categorize time controls
    df_parsed = filter_and_group_time_controls(df)
    
    # Create rating buckets
    df_bucketed = create_rating_buckets(df_parsed)
    
    # Filter to only include time controls with sufficient data per rating bucket
    df_filtered = filter_time_controls_by_threshold(df_bucketed, min_games_per_bucket)
    
    # Calculate averages
    averages = calculate_averages(df_filtered)
    
    # Print summary statistics
    print_summary_statistics(averages)
    
    # Create visualization
    fig = create_visualization(averages)
    
    # Save the plot
    output_path = "./media/acpl_timecontrol_analysis_all_combined.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    return averages

if __name__ == "__main__":
    results = main()
