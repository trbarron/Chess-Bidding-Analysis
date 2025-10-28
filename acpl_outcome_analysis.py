import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import argparse

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

def pair_games(df):
    """Pair up White and Black rows for each game."""
    print("\nPairing games (White and Black rows)...")
    
    # Separate White and Black rows
    white_df = df[df['Color'] == 'White'].copy()
    black_df = df[df['Color'] == 'Black'].copy()
    
    print(f"White rows: {len(white_df):,}")
    print(f"Black rows: {len(black_df):,}")
    
    # Merge on GameID
    paired_df = white_df.merge(
        black_df,
        on='GameID',
        suffixes=('_white', '_black')
    )
    
    print(f"Successfully paired games: {len(paired_df):,}")
    
    return paired_df

def categorize_time_control(time_control_str):
    """Categorize time control into Lichess categories."""
    try:
        parts = time_control_str.split('+')
        base_time = int(parts[0])
        increment = int(parts[1]) if len(parts) > 1 else 0
        
        # Total estimated time = base + 40 moves * increment
        total_time = base_time + (40 * increment)
        
        # Lichess categories
        if total_time < 180:  # < 3 minutes
            return 'Bullet'
        elif total_time < 480:  # < 8 minutes
            return 'Blitz'
        elif total_time < 1500:  # < 25 minutes
            return 'Rapid'
        else:  # >= 25 minutes
            return 'Classical'
    except:
        return 'Unknown'

def calculate_game_features(df):
    """Calculate average rating, ACPL difference, outcome, and time control category."""
    print("\nCalculating game features...")
    
    # Average rating of both players
    df['AvgRating'] = (df['Rating_white'] + df['Rating_black']) / 2
    
    # ACPL difference (White ACPL - Black ACPL)
    # Positive means White played worse (higher ACPL)
    # Negative means Black played worse
    df['ACPL_Diff'] = df['ACPL_midgame_white'] - df['ACPL_midgame_black']
    
    # Determine outcome from White's perspective
    # 1 = White win, 0 = Draw, -1 = Black win
    df['Outcome'] = df['Winner_white'].apply(
        lambda x: 1 if x == 'White' else (-1 if x == 'Black' else 0)
    )
    
    # Create outcome labels
    df['OutcomeLabel'] = df['Outcome'].map({
        1: 'White Win',
        0: 'Draw',
        -1: 'Black Win'
    })
    
    # Categorize time control
    df['TimeControlCategory'] = df['TimeControl_white'].apply(categorize_time_control)
    
    # Extract specific time control (for zero-increment analysis)
    df['SpecificTimeControl'] = df['TimeControl_white']
    
    # Parse time control to get base time and increment
    time_control_parts = df['TimeControl_white'].str.split('+')
    df['BaseTime'] = pd.to_numeric(time_control_parts.str[0], errors='coerce')
    df['Increment'] = pd.to_numeric(time_control_parts.str[1].fillna('0'), errors='coerce')
    
    print(f"Average rating range: {df['AvgRating'].min():.0f} - {df['AvgRating'].max():.0f}")
    print(f"ACPL difference range: {df['ACPL_Diff'].min():.1f} - {df['ACPL_Diff'].max():.1f}")
    
    # Count time control categories
    tc_counts = df['TimeControlCategory'].value_counts()
    print(f"\nTime control distribution:")
    for tc, count in tc_counts.items():
        print(f"  {tc}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Count outcomes
    outcome_counts = df['OutcomeLabel'].value_counts()
    print(f"\nOutcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df

def create_rating_buckets(df, bucket_size=200):
    """Create rating buckets."""
    df['RatingBucket'] = (df['AvgRating'] // bucket_size) * bucket_size
    df['RatingBucketLabel'] = df['RatingBucket'].astype(int).astype(str) + '-' + (df['RatingBucket'] + bucket_size - 1).astype(int).astype(str)
    
    return df

def create_acpl_buckets(df, bucket_size=20):
    """Create ACPL difference buckets."""
    # Round to nearest bucket
    df['ACPL_Diff_Bucket'] = (df['ACPL_Diff'] // bucket_size) * bucket_size
    
    return df

def calculate_outcome_rates(df, min_games=100):
    """Calculate win/loss/draw rates for each time control, rating bucket, and ACPL difference bucket."""
    print(f"\nCalculating outcome rates (minimum {min_games} games per bucket)...")
    
    # Group by time control category, rating bucket, and ACPL difference bucket
    grouped = df.groupby(['TimeControlCategory', 'RatingBucket', 'ACPL_Diff_Bucket'])
    
    results = []
    
    for (tc_category, rating_bucket, acpl_bucket), group in grouped:
        n_games = len(group)
        
        if n_games >= min_games:
            # Calculate rates
            white_wins = (group['Outcome'] == 1).sum()
            black_wins = (group['Outcome'] == -1).sum()
            draws = (group['Outcome'] == 0).sum()
            
            white_win_rate = white_wins / n_games * 100
            black_win_rate = black_wins / n_games * 100
            draw_rate = draws / n_games * 100
            
            # Calculate draw odds rates (draws count as wins for Black)
            black_with_draw_odds_rate = (black_wins + draws) / n_games * 100
            white_with_draw_odds_rate = white_wins / n_games * 100  # White loses on draws
            
            results.append({
                'TimeControlCategory': tc_category,
                'RatingBucket': rating_bucket,
                'RatingBucketLabel': f"{int(rating_bucket)}-{int(rating_bucket)+199}",
                'ACPL_Diff_Bucket': acpl_bucket,
                'NumGames': n_games,
                'WhiteWinRate': white_win_rate,
                'BlackWinRate': black_win_rate,
                'DrawRate': draw_rate,
                'BlackDrawOddsRate': black_with_draw_odds_rate,
                'WhiteDrawOddsRate': white_with_draw_odds_rate,
                'WhiteWins': white_wins,
                'BlackWins': black_wins,
                'Draws': draws
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"Created {len(results_df)} bucket combinations with sufficient data")
    
    # Print breakdown by time control
    if len(results_df) > 0:
        print("\nBreakdown by time control:")
        for tc in sorted(results_df['TimeControlCategory'].unique()):
            tc_count = len(results_df[results_df['TimeControlCategory'] == tc])
            print(f"  {tc}: {tc_count} bucket combinations")
    
    return results_df

def calculate_outcome_rates_by_specific_tc(df, min_games=1000):
    """Calculate win/loss/draw rates for specific time controls (0 increment only) by rating bucket and ACPL difference."""
    print(f"\nCalculating outcome rates by specific time control (minimum {min_games} games per bucket)...")
    
    # Filter to zero increment only
    df_zero_inc = df[df['Increment'] == 0].copy()
    print(f"Games with 0 increment: {len(df_zero_inc):,}")
    
    # Group by specific time control, rating bucket, and ACPL difference bucket
    grouped = df_zero_inc.groupby(['SpecificTimeControl', 'RatingBucket', 'ACPL_Diff_Bucket'])
    
    results = []
    
    for (tc, rating_bucket, acpl_bucket), group in grouped:
        n_games = len(group)
        
        if n_games >= min_games:
            # Calculate rates
            white_wins = (group['Outcome'] == 1).sum()
            black_wins = (group['Outcome'] == -1).sum()
            draws = (group['Outcome'] == 0).sum()
            
            white_win_rate = white_wins / n_games * 100
            black_win_rate = black_wins / n_games * 100
            draw_rate = draws / n_games * 100
            
            # Calculate draw odds rates (draws count as wins for Black)
            black_with_draw_odds_rate = (black_wins + draws) / n_games * 100
            white_with_draw_odds_rate = white_wins / n_games * 100
            
            results.append({
                'SpecificTimeControl': tc,
                'RatingBucket': rating_bucket,
                'RatingBucketLabel': f"{int(rating_bucket)}-{int(rating_bucket)+199}",
                'ACPL_Diff_Bucket': acpl_bucket,
                'NumGames': n_games,
                'WhiteWinRate': white_win_rate,
                'BlackWinRate': black_win_rate,
                'DrawRate': draw_rate,
                'BlackDrawOddsRate': black_with_draw_odds_rate,
                'WhiteDrawOddsRate': white_with_draw_odds_rate,
                'WhiteWins': white_wins,
                'BlackWins': black_wins,
                'Draws': draws
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"Created {len(results_df)} bucket combinations with sufficient data")
    
    # Print breakdown by specific time control
    if len(results_df) > 0:
        print("\nTime controls with sufficient data:")
        for tc in sorted(results_df['SpecificTimeControl'].unique()):
            tc_count = len(results_df[results_df['SpecificTimeControl'] == tc])
            tc_games = df_zero_inc[df_zero_inc['SpecificTimeControl'] == tc]
            print(f"  {tc}: {len(tc_games):,} games, {tc_count} bucket combinations")
    
    return results_df

def create_heatmap_visualization(results_df, rating_bucket, metric='WhiteWinRate'):
    """Create a heatmap showing outcome rates vs ACPL difference for a specific rating bucket."""
    bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
    
    if len(bucket_data) == 0:
        return None
    
    # Sort by ACPL difference bucket
    bucket_data = bucket_data.sort_values('ACPL_Diff_Bucket')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bar plot
    colors = ['green' if x < 0 else 'red' if x > 0 else 'gray' 
              for x in bucket_data['ACPL_Diff_Bucket']]
    
    # Calculate appropriate bar width based on bucket spacing
    if len(bucket_data) > 1:
        bucket_diff = bucket_data['ACPL_Diff_Bucket'].iloc[1] - bucket_data['ACPL_Diff_Bucket'].iloc[0]
        bar_width = bucket_diff * 0.9
    else:
        bar_width = 10
    
    bars = ax.bar(bucket_data['ACPL_Diff_Bucket'], bucket_data[metric], width=bar_width, 
                   color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at 50%
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% line')
    
    # Add vertical line at 0
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Equal ACPL')
    
    rating_label = bucket_data['RatingBucketLabel'].iloc[0]
    metric_label = metric.replace('Rate', ' Rate').replace('Win', 'Win')
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_label} (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_label} vs ACPL Difference\nRating Range: {rating_label}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add text annotations
    ax.text(0.02, 0.98, 'Green: White played better (lower ACPL)\nRed: Black played better (lower ACPL)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig

def create_multi_rating_visualization(results_df, rating_buckets=None, time_control_label="All"):
    """Create a plot showing White win rate vs ACPL difference for multiple rating buckets."""
    if rating_buckets is None:
        rating_buckets = sorted(results_df['RatingBucket'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set background color for the entire plot
    ax.set_facecolor('#f0f0f0')  # Light grey base
    
    # Get x-axis range
    x_min = results_df['ACPL_Diff_Bucket'].min()
    x_max = results_df['ACPL_Diff_Bucket'].max()
    
    # Add shaded regions to indicate which player is doing better
    # Left side (White playing better) - white background
    ax.axvspan(x_min, 0, alpha=0.6, color='white', label='White Playing Better', zorder=0)
    # Right side (Black playing better) - grey background
    ax.axvspan(0, x_max, alpha=0.6, color='#d0d0d0', label='Black Playing Better', zorder=0)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    for i, rating_bucket in enumerate(rating_buckets):
        bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
        bucket_data = bucket_data.sort_values('ACPL_Diff_Bucket')
        
        if len(bucket_data) > 0:
            rating_label = bucket_data['RatingBucketLabel'].iloc[0]
            
            ax.plot(bucket_data['ACPL_Diff_Bucket'], bucket_data['WhiteWinRate'],
                   marker='o', linewidth=2.5, markersize=7, 
                   color=colors[i], label=f'Rating {rating_label}', alpha=0.8, zorder=2)
    
    # Add reference line at 50%
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='50% Win Rate', zorder=1)
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('White Win Rate (%)', fontsize=13, fontweight='bold')
    
    title = f'White Win Rate vs ACPL Difference by Player Rating'
    if time_control_label != "All":
        title += f'\nTime Control: {time_control_label}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Improve X axis clarity
    ax.grid(True, alpha=0.3, linewidth=0.8, zorder=1)
    ax.grid(True, which='major', axis='x', alpha=0.5, linewidth=1, zorder=1)
    
    # Set more readable x-axis ticks
    x_ticks = np.arange(np.floor(x_min/50)*50, np.ceil(x_max/50)*50 + 1, 50)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    return fig

def create_draw_rate_visualization(results_df, rating_buckets=None, time_control_label="All"):
    """Create a plot showing Draw rate vs ACPL difference for multiple rating buckets."""
    if rating_buckets is None:
        rating_buckets = sorted(results_df['RatingBucket'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set background color for the entire plot
    ax.set_facecolor('#f0f0f0')  # Light grey base
    
    # Get x-axis range
    x_min = results_df['ACPL_Diff_Bucket'].min()
    x_max = results_df['ACPL_Diff_Bucket'].max()
    
    # Add shaded regions to indicate which player is doing better
    # Left side (White playing better) - white background
    ax.axvspan(x_min, 0, alpha=0.6, color='white', label='White Playing Better', zorder=0)
    # Right side (Black playing better) - grey background
    ax.axvspan(0, x_max, alpha=0.6, color='#d0d0d0', label='Black Playing Better', zorder=0)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    for i, rating_bucket in enumerate(rating_buckets):
        bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
        bucket_data = bucket_data.sort_values('ACPL_Diff_Bucket')
        
        if len(bucket_data) > 0:
            rating_label = bucket_data['RatingBucketLabel'].iloc[0]
            
            ax.plot(bucket_data['ACPL_Diff_Bucket'], bucket_data['DrawRate'],
                   marker='o', linewidth=2.5, markersize=7, 
                   color=colors[i], label=f'Rating {rating_label}', alpha=0.8, zorder=2)
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Draw Rate (%)', fontsize=13, fontweight='bold')
    
    title = f'Draw Rate vs ACPL Difference by Player Rating'
    if time_control_label != "All":
        title += f'\nTime Control: {time_control_label}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Improve X axis clarity
    ax.grid(True, alpha=0.3, linewidth=0.8, zorder=1)
    ax.grid(True, which='major', axis='x', alpha=0.5, linewidth=1, zorder=1)
    
    # Set more readable x-axis ticks
    x_ticks = np.arange(np.floor(x_min/50)*50, np.ceil(x_max/50)*50 + 1, 50)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    return fig

def create_time_control_comparison_visualization(results_df, rating_bucket, metric='WhiteWinRate'):
    """Create a plot showing win/draw rates vs ACPL difference for different time controls within a rating bucket."""
    bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
    
    if len(bucket_data) == 0:
        return None
    
    # Get unique time controls for this rating bucket
    time_controls = sorted(bucket_data['SpecificTimeControl'].unique(), 
                          key=lambda x: int(x.split('+')[0]) if '+' in x else 0)
    
    if len(time_controls) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set background color for the entire plot
    ax.set_facecolor('#f0f0f0')  # Light grey base
    
    # Get x-axis range
    x_min = bucket_data['ACPL_Diff_Bucket'].min()
    x_max = bucket_data['ACPL_Diff_Bucket'].max()
    
    # Add shaded regions to indicate which player is doing better
    ax.axvspan(x_min, 0, alpha=0.6, color='white', label='White Playing Better', zorder=0)
    ax.axvspan(0, x_max, alpha=0.6, color='#d0d0d0', label='Black Playing Better', zorder=0)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_controls)))
    
    for i, tc in enumerate(time_controls):
        tc_data = bucket_data[bucket_data['SpecificTimeControl'] == tc].copy()
        tc_data = tc_data.sort_values('ACPL_Diff_Bucket')
        
        if len(tc_data) > 0:
            ax.plot(tc_data['ACPL_Diff_Bucket'], tc_data[metric],
                   marker='o', linewidth=2.5, markersize=7, 
                   color=colors[i], label=f'{tc}', alpha=0.8, zorder=2)
    
    # Add reference line
    if metric == 'WhiteWinRate':
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='50% Win Rate', zorder=1)
    
    rating_label = bucket_data['RatingBucketLabel'].iloc[0]
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=13, fontweight='bold')
    
    if metric == 'WhiteWinRate':
        ax.set_ylabel('White Win Rate (%)', fontsize=13, fontweight='bold')
        title = f'White Win Rate vs ACPL Difference by Time Control\nRating Range: {rating_label}'
    else:
        ax.set_ylabel('Draw Rate (%)', fontsize=13, fontweight='bold')
        title = f'Draw Rate vs ACPL Difference by Time Control\nRating Range: {rating_label}'
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Improve X axis clarity
    ax.grid(True, alpha=0.3, linewidth=0.8, zorder=1)
    ax.grid(True, which='major', axis='x', alpha=0.5, linewidth=1, zorder=1)
    
    # Set more readable x-axis ticks
    x_ticks = np.arange(np.floor(x_min/50)*50, np.ceil(x_max/50)*50 + 1, 50)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title='Time Control')
    
    plt.tight_layout()
    
    return fig

def create_draw_odds_visualization(results_df, rating_buckets=None, time_control_label="All"):
    """Create a plot showing White's win rate when Black has draw odds vs ACPL difference for multiple rating buckets."""
    if rating_buckets is None:
        rating_buckets = sorted(results_df['RatingBucket'].unique())
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set background color for the entire plot
    ax.set_facecolor('#f0f0f0')  # Light grey base
    
    # Get x-axis range
    x_min = results_df['ACPL_Diff_Bucket'].min()
    x_max = results_df['ACPL_Diff_Bucket'].max()
    
    # Add shaded regions to indicate which player is doing better
    # Left side (White playing better) - white background
    ax.axvspan(x_min, 0, alpha=0.6, color='white', label='White Playing Better', zorder=0)
    # Right side (Black playing better) - grey background
    ax.axvspan(0, x_max, alpha=0.6, color='#d0d0d0', label='Black Playing Better', zorder=0)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_buckets)))
    
    for i, rating_bucket in enumerate(rating_buckets):
        bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
        bucket_data = bucket_data.sort_values('ACPL_Diff_Bucket')
        
        if len(bucket_data) > 0:
            rating_label = bucket_data['RatingBucketLabel'].iloc[0]
            
            # Plot White's win rate when Black has draw odds (higher = White winning, lower = Black winning/drawing)
            ax.plot(bucket_data['ACPL_Diff_Bucket'], bucket_data['WhiteDrawOddsRate'],
                   marker='o', linewidth=2.5, markersize=7, 
                   color=colors[i], label=f'Rating {rating_label}', alpha=0.8, zorder=2)
    
    # Add reference line at 50%
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='50% Win Rate', zorder=1)
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('White Win Rate (Black has Draw Odds) (%)', fontsize=13, fontweight='bold')
    
    title = f'White Win Rate vs ACPL Difference by Player Rating (Black has Draw Odds)'
    if time_control_label != "All":
        title += f'\nTime Control: {time_control_label}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Improve X axis clarity
    ax.grid(True, alpha=0.3, linewidth=0.8, zorder=1)
    ax.grid(True, which='major', axis='x', alpha=0.5, linewidth=1, zorder=1)
    
    # Set more readable x-axis ticks
    x_ticks = np.arange(np.floor(x_min/50)*50, np.ceil(x_max/50)*50 + 1, 50)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Add text annotation explaining draw odds
    ax.text(0.02, 0.02, 'Black has Draw Odds: Draws count as wins for Black\nHigher values = White must win, Lower values = Black wins or draws',
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    return fig

def create_outcome_stacked_chart(results_df, rating_buckets=None):
    """Create stacked area chart showing all three outcomes."""
    if rating_buckets is None:
        rating_buckets = sorted(results_df['RatingBucket'].unique())
    
    # Focus on one rating bucket for clarity
    rating_bucket = rating_buckets[len(rating_buckets)//2]  # Middle rating bucket
    
    bucket_data = results_df[results_df['RatingBucket'] == rating_bucket].copy()
    bucket_data = bucket_data.sort_values('ACPL_Diff_Bucket')
    
    if len(bucket_data) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = bucket_data['ACPL_Diff_Bucket'].values
    white_wins = bucket_data['WhiteWinRate'].values
    draws = bucket_data['DrawRate'].values
    black_wins = bucket_data['BlackWinRate'].values
    
    # Create stacked area chart
    ax.fill_between(x, 0, white_wins, alpha=0.7, color='lightgreen', label='White Wins')
    ax.fill_between(x, white_wins, white_wins + draws, alpha=0.7, color='gray', label='Draws')
    ax.fill_between(x, white_wins + draws, 100, alpha=0.7, color='lightcoral', label='Black Wins')
    
    # Add reference line at ACPL = 0
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Equal ACPL')
    
    rating_label = bucket_data['RatingBucketLabel'].iloc[0]
    
    ax.set_xlabel('ACPL Difference (White ACPL - Black ACPL)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Outcome Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Game Outcomes vs ACPL Difference\nRating Range: {rating_label}', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    return fig

def print_summary_statistics(df, results_df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal games analyzed: {len(df):,}")
    print(f"Rating range: {df['AvgRating'].min():.0f} - {df['AvgRating'].max():.0f}")
    print(f"ACPL difference range: {df['ACPL_Diff'].min():.1f} - {df['ACPL_Diff'].max():.1f}")
    
    print(f"\nOverall outcome distribution:")
    outcome_counts = df['OutcomeLabel'].value_counts()
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\nACPL difference vs outcomes:")
    for outcome in ['White Win', 'Draw', 'Black Win']:
        outcome_data = df[df['OutcomeLabel'] == outcome]
        if len(outcome_data) > 0:
            mean_diff = outcome_data['ACPL_Diff'].mean()
            median_diff = outcome_data['ACPL_Diff'].median()
            print(f"  {outcome}: mean ACPL diff = {mean_diff:.2f}, median = {median_diff:.2f}")
    
    print(f"\nBucket combinations with sufficient data: {len(results_df)}")
    print(f"Rating buckets represented: {results_df['RatingBucket'].nunique()}")
    print(f"ACPL difference buckets represented: {results_df['ACPL_Diff_Bucket'].nunique()}")
    
    # Show correlation between ACPL difference and white win rate
    print(f"\nCorrelation between ACPL difference and White win rate:")
    for rating_bucket in sorted(results_df['RatingBucket'].unique()):
        bucket_data = results_df[results_df['RatingBucket'] == rating_bucket]
        if len(bucket_data) >= 3:
            correlation = bucket_data['ACPL_Diff_Bucket'].corr(bucket_data['WhiteWinRate'])
            rating_label = bucket_data['RatingBucketLabel'].iloc[0]
            print(f"  Rating {rating_label}: r = {correlation:.3f}")

def main(min_games_per_bucket=100, acpl_bucket_size=20, 
         generate_time_control_graphs=True, generate_rating_bucket_graphs=True, 
         generate_combined_graphs=True):
    """Main analysis function.
    
    Args:
        min_games_per_bucket: Minimum games required per bucket
        acpl_bucket_size: Size of ACPL difference buckets
        generate_time_control_graphs: Whether to generate per-time-control graphs (Bullet, Blitz, etc.)
        generate_rating_bucket_graphs: Whether to generate per-rating-bucket graphs (comparing time controls)
        generate_combined_graphs: Whether to generate combined graphs (all ratings, all time controls)
    """
    print("Chess ACPL Difference vs Game Outcomes Analysis")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Generate time control graphs: {generate_time_control_graphs}")
    print(f"  - Generate rating bucket graphs: {generate_rating_bucket_graphs}")
    print(f"  - Generate combined graphs: {generate_combined_graphs}")
    print()
    
    # Load and combine data
    df = load_and_combine_data()
    
    # Pair up White and Black rows for each game
    paired_df = pair_games(df)
    
    # Calculate game features
    game_df = calculate_game_features(paired_df)
    
    # Create rating buckets
    game_df = create_rating_buckets(game_df, bucket_size=200)
    
    # Create ACPL difference buckets
    game_df = create_acpl_buckets(game_df, bucket_size=acpl_bucket_size)
    
    # Calculate outcome rates
    results_df = calculate_outcome_rates(game_df, min_games=min_games_per_bucket)
    
    # Print summary statistics
    print_summary_statistics(game_df, results_df)
    
    # Calculate outcome rates for specific time controls (0 increment only) - only if needed
    results_specific_tc = None
    if generate_rating_bucket_graphs:
        results_specific_tc = calculate_outcome_rates_by_specific_tc(game_df, min_games=1000)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # === Part 1: Per-time-category graphs (existing functionality) ===
    if generate_time_control_graphs:
        print("\n=== Generating graphs by time control category ===")
        time_control_categories = ['Bullet', 'Blitz', 'Rapid', 'Classical']
        
        for tc_category in time_control_categories:
            # Filter results for this time control category
            tc_results = results_df[results_df['TimeControlCategory'] == tc_category].copy()
            
            if len(tc_results) == 0:
                print(f"  Skipping {tc_category} - no data")
                continue
            
            print(f"\n  Processing {tc_category}...")
            
            # Get rating buckets for this time control
            rating_buckets = sorted(tc_results['RatingBucket'].unique())
            
            if len(rating_buckets) == 0:
                print(f"  Skipping {tc_category} - no rating buckets with sufficient data")
                continue
            
            # 1. White win rate plot
            fig1 = create_multi_rating_visualization(tc_results, rating_buckets, tc_category)
            output_path1 = f"./media/acpl_outcome_winrate_{tc_category.lower()}.png"
            fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path1}")
            plt.close(fig1)
            
            # 2. Draw rate plot
            fig2 = create_draw_rate_visualization(tc_results, rating_buckets, tc_category)
            output_path2 = f"./media/acpl_outcome_drawrate_{tc_category.lower()}.png"
            fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path2}")
            plt.close(fig2)
            
            # 3. Draw odds plot (Black gets draw odds)
            fig3 = create_draw_odds_visualization(tc_results, rating_buckets, tc_category)
            output_path3 = f"./media/acpl_outcome_drawodds_{tc_category.lower()}.png"
            fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path3}")
            plt.close(fig3)
    
    # === Part 2: Per-rating-bucket graphs showing different time controls ===
    if generate_rating_bucket_graphs and results_specific_tc is not None and len(results_specific_tc) > 0:
        print("\n=== Generating graphs by rating bucket (0 increment time controls) ===")
        
        # Get rating buckets that have data
        rating_buckets = sorted(results_specific_tc['RatingBucket'].unique())
        
        for rating_bucket in rating_buckets:
            rating_label = f"{int(rating_bucket)}-{int(rating_bucket)+199}"
            
            # Check if this rating bucket has multiple time controls
            bucket_tcs = results_specific_tc[results_specific_tc['RatingBucket'] == rating_bucket]['SpecificTimeControl'].unique()
            
            if len(bucket_tcs) < 2:
                print(f"  Skipping rating {rating_label} - only {len(bucket_tcs)} time control(s)")
                continue
            
            print(f"\n  Processing rating bucket {rating_label}...")
            
            # 1. White win rate plot
            fig1 = create_time_control_comparison_visualization(results_specific_tc, rating_bucket, metric='WhiteWinRate')
            if fig1:
                output_path1 = f"./media/acpl_outcome_tc_comparison_winrate_{rating_label}.png"
                fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
                print(f"    Saved: {output_path1}")
                plt.close(fig1)
            
            # 2. Draw rate plot
            fig2 = create_time_control_comparison_visualization(results_specific_tc, rating_bucket, metric='DrawRate')
            if fig2:
                output_path2 = f"./media/acpl_outcome_tc_comparison_drawrate_{rating_label}.png"
                fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
                print(f"    Saved: {output_path2}")
                plt.close(fig2)
    
    # === Part 3: Combined graph with all rating buckets (all time controls combined) ===
    if generate_combined_graphs:
        print("\n=== Generating combined graph (all rating buckets, all time controls) ===")
        
        # Use the original results_df which has all time controls
        all_rating_buckets = sorted(results_df['RatingBucket'].unique())
        
        if len(all_rating_buckets) > 0:
            # Aggregate across all time controls for each rating bucket
            aggregated_results = []
            
            for rating_bucket in all_rating_buckets:
                for acpl_bucket in sorted(results_df['ACPL_Diff_Bucket'].unique()):
                    bucket_data = results_df[
                        (results_df['RatingBucket'] == rating_bucket) & 
                        (results_df['ACPL_Diff_Bucket'] == acpl_bucket)
                    ]
                    
                    if len(bucket_data) > 0:
                        # Weighted average by number of games
                        total_games = bucket_data['NumGames'].sum()
                        white_wins = bucket_data['WhiteWins'].sum()
                        black_wins = bucket_data['BlackWins'].sum()
                        draws = bucket_data['Draws'].sum()
                        
                        white_win_rate = (white_wins / total_games * 100) if total_games > 0 else 0
                        draw_rate = (draws / total_games * 100) if total_games > 0 else 0
                        black_draw_odds_rate = ((black_wins + draws) / total_games * 100) if total_games > 0 else 0
                        white_draw_odds_rate = (white_wins / total_games * 100) if total_games > 0 else 0  # Same as white_win_rate
                        
                        aggregated_results.append({
                            'RatingBucket': rating_bucket,
                            'RatingBucketLabel': f"{int(rating_bucket)}-{int(rating_bucket)+199}",
                            'ACPL_Diff_Bucket': acpl_bucket,
                            'WhiteWinRate': white_win_rate,
                            'DrawRate': draw_rate,
                            'BlackDrawOddsRate': black_draw_odds_rate,
                            'WhiteDrawOddsRate': white_draw_odds_rate,
                            'NumGames': total_games
                        })
            
            agg_results_df = pd.DataFrame(aggregated_results)
            
            if len(agg_results_df) > 0:
                # White win rate plot
                fig1 = create_multi_rating_visualization(agg_results_df, all_rating_buckets, "All Time Controls")
                output_path1 = "./media/acpl_outcome_winrate_all_combined.png"
                fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path1}")
                plt.close(fig1)
                
                # Draw rate plot
                fig2 = create_draw_rate_visualization(agg_results_df, all_rating_buckets, "All Time Controls")
                output_path2 = "./media/acpl_outcome_drawrate_all_combined.png"
                fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path2}")
                plt.close(fig2)
                
                # Draw odds plot (Black gets draw odds)
                fig3 = create_draw_odds_visualization(agg_results_df, all_rating_buckets, "All Time Controls")
                output_path3 = "./media/acpl_outcome_drawodds_all_combined.png"
                fig3.savefig(output_path3, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_path3}")
                plt.close(fig3)
    
    print("\nAnalysis complete!")
    
    return game_df, results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze chess game outcomes based on ACPL differences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

        """
    )
    
    parser.add_argument('--no-time-controls', action='store_true',
                       help='Skip generating per-time-control graphs (Bullet, Blitz, etc.)')
    parser.add_argument('--no-rating-buckets', action='store_true',
                       help='Skip generating per-rating-bucket graphs (comparing time controls)')
    parser.add_argument('--no-combined', action='store_true',
                       help='Skip generating combined graphs (all ratings, all time controls)')
    
    # Convenience flags
    parser.add_argument('--combined-only', action='store_true',
                       help='Generate only combined graphs (shortcut for --no-time-controls --no-rating-buckets)')
    parser.add_argument('--time-controls-only', action='store_true',
                       help='Generate only time control graphs (shortcut for --no-rating-buckets --no-combined)')
    
    args = parser.parse_args()
    
    # Handle convenience flags
    if args.combined_only:
        args.no_time_controls = True
        args.no_rating_buckets = True
        args.no_combined = False
    
    if args.time_controls_only:
        args.no_time_controls = False
        args.no_rating_buckets = True
        args.no_combined = True
    
    # Determine which graphs to generate
    generate_tc = not args.no_time_controls
    generate_rb = not args.no_rating_buckets
    generate_combined = not args.no_combined
    
    game_data, results = main(
        generate_time_control_graphs=generate_tc,
        generate_rating_bucket_graphs=generate_rb,
        generate_combined_graphs=generate_combined
    )

