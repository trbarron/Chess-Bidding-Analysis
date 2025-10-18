"""
Chess Game Analyzer - Lichess Database Processor
"""

import chess
import chess.pgn
import csv
import sys
import re
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from contextlib import contextmanager


class Profiler:
    """Thread-safe profiler for tracking time spent in different operations"""
    
    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation: str):
        """Context manager to profile a code block"""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self._timings[operation] = self._timings.get(operation, 0.0) + elapsed
                self._counts[operation] = self._counts.get(operation, 0) + 1
    
    def print_summary(self):
        """Print profiling summary showing time distribution"""
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        total_time = sum(self._timings.values())
        
        sorted_operations = sorted(self._timings.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Operation':<40} {'Time (s)':<12} {'Percent':<10} {'Count':<10} {'Avg (ms)'}")
        print("-" * 80)
        
        for operation, elapsed in sorted_operations:
            count = self._counts[operation]
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            avg_ms = (elapsed / count * 1000) if count > 0 else 0
            print(f"{operation:<40} {elapsed:>11.2f} {percentage:>9.1f}% {count:>9} {avg_ms:>10.2f}")
        
        print("-" * 80)
        print(f"{'TOTAL':<40} {total_time:>11.2f}")
        print("="*80)


@dataclass
class GameMetrics:
    time_control: str
    rating: int
    color: str
    acpl_midgame: float
    inaccuracies_midgame: int
    mistakes_midgame: int
    blunders_midgame: int
    num_moves_midgame: int


class GameDivider:
    """Reimplementation of Lichess's Divider algorithm"""
    
    @staticmethod
    def divide_game(boards: List[chess.Board]) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (midgame_start, endgame_start) as move indices (0-based)
        Returns (None, None) if no clear phases
        """
        midgame_start = None
        endgame_start = None
        
        for idx, board in enumerate(boards):
            if midgame_start is None:
                piece_count = GameDivider.majors_and_minors(board)
                if piece_count <= 10:
                    midgame_start = idx
                elif GameDivider.backrank_sparse(board):
                    midgame_start = idx
                elif GameDivider.mixedness(board) > 150:
                    midgame_start = idx
            
            if midgame_start is not None and endgame_start is None:
                if GameDivider.majors_and_minors(board) <= 6:
                    endgame_start = idx
                    break
        
        return midgame_start, endgame_start
    
    @staticmethod
    def majors_and_minors(board: chess.Board) -> int:
        """Count pieces excluding kings and pawns"""
        return (chess.popcount(board.knights) + 
                chess.popcount(board.bishops) + 
                chess.popcount(board.rooks) + 
                chess.popcount(board.queens))
    
    @staticmethod
    def backrank_sparse(board: chess.Board) -> bool:
        """Check if back ranks are sparse (<4 pieces on 1st or 8th rank)"""
        white_backrank = 0
        black_backrank = 0
        
        for file in range(8):
            piece = board.piece_at(chess.square(file, 0))
            if piece is not None and piece.color == chess.WHITE:
                white_backrank += 1
            
            piece = board.piece_at(chess.square(file, 7))
            if piece is not None and piece.color == chess.BLACK:
                black_backrank += 1
        
        return white_backrank < 4 or black_backrank < 4
    
    @staticmethod
    def mixedness(board: chess.Board) -> int:
        """Calculate mixedness score based on piece distribution"""
        total_score = 0
        piece_map = board.piece_map()
        
        for y in range(7):
            for x in range(7):
                white_count = 0
                black_count = 0
                
                for dy in range(2):
                    for dx in range(2):
                        square = chess.square(x + dx, y + dy)
                        piece = piece_map.get(square)
                        
                        if piece is not None:
                            if piece.color == chess.WHITE:
                                white_count += 1
                            else:
                                black_count += 1
                
                total_score += GameDivider.score(y + 1, white_count, black_count)
                if total_score > 150:
                    return total_score
        
        return total_score
    
    @staticmethod
    def score(y: int, white: int, black: int) -> int:
        """Score function from Lichess's implementation"""
        if white == 0:
            if black == 0:
                return 0
            if black == 1:
                return 1 + y
            if black == 2:
                return 2 + (6 - y) if y < 6 else 0
            if black == 3:
                return 3 + (7 - y) if y < 7 else 0
            if black == 4:
                return 3 + (7 - y) if y < 7 else 0
        elif black == 0:
            if white == 1:
                return 1 + (8 - y)
            if white == 2:
                return 2 + (y - 2) if y > 2 else 0
            if white == 3 or white == 4:
                return 3 + (y - 1) if y > 1 else 0
        else:
            if white == 1 and black == 1:
                return 5 + abs(4 - y)
            if white == 2 and black == 1:
                return 4 + (y - 1)
            if white == 3 and black == 1:
                return 5 + (y - 1)
            if white == 1 and black == 2:
                return 4 + (7 - y)
            if white == 2 and black == 2:
                return 7
            if white == 1 and black == 3:
                return 5 + (7 - y)
        
        return 0


class ChessAnalyzer:
    
    def __init__(self, input_file: str, output_file: str, max_workers: int = None, limit: int = None):
        self.input_file = input_file
        self.output_file = output_file
        self.games_processed = 0
        self.games_skipped = 0
        self.skipped_no_time_control = 0
        self.skipped_no_eval = 0
        self.skipped_no_rating = 0
        self.skipped_insufficient_eval = 0
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.limit = limit
        self._lock = threading.Lock()
        self._last_progress_time = time.time()
        self._start_time = time.time()
        self._progress_interval = 5.0
        self.profiler = Profiler()
        self._stop_processing = False
    
    def process_file(self, batch_size: int = 100):
        """Process the entire PGN file and write results to CSV using multithreading"""
        print(f"Processing: {self.input_file}")
        print(f"Output to: {self.output_file}")
        print(f"Using {self.max_workers} threads with batch size {batch_size}")
        if self.limit:
            print(f"Limiting to {self.limit} successfully processed games")
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'TimeControl', 'Rating', 'Color', 'ACPL_midgame',
                'Inaccuracies_midgame', 'Mistakes_midgame', 'Blunders_midgame',
                'NumMoves_midgame'
            ])
            
            # Process file in streaming batches with parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                self._process_file_streaming(writer, batch_size, executor)
        
        total_read = self.games_processed + self.games_skipped
        elapsed = time.time() - self._start_time
        rate = total_read / elapsed if elapsed > 0 else 0
        process_pct = (self.games_processed / total_read * 100) if total_read > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"COMPLETE!")
        print(f"{'='*80}")
        print(f"Total games read:     {total_read:,}")
        print(f"Games processed:      {self.games_processed:,} ({process_pct:.1f}% of read)")
        print(f"Games skipped:        {self.games_skipped:,} ({100-process_pct:.1f}% of read)")
        print(f"Total time:           {elapsed:.1f}s")
        print(f"Processing rate:      {rate:.1f} games/sec")
        print(f"\nSkip reasons:")
        print(f"  - No time control:           {self.skipped_no_time_control:,}")
        print(f"  - No evaluation data:        {self.skipped_no_eval:,}")
        print(f"  - No/invalid ratings:        {self.skipped_no_rating:,}")
        print(f"  - Insufficient evaluation:   {self.skipped_insufficient_eval:,}")
        
        self.profiler.print_summary()
    
    def _process_file_streaming(self, writer, batch_size: int, executor):
        """Process the PGN file in streaming batches with parallel processing"""
        from queue import Queue
        import threading
        
        # Queue to hold batches ready for processing
        batch_queue = Queue(maxsize=self.max_workers * 2)  # Buffer for 2x workers
        results_queue = Queue()
        
        def game_reader():
            current_batch = []
            
            with open(self.input_file, 'r', encoding='utf-8') as pgn_file:
                while True:
                    if self._stop_processing:
                        break
                    
                    with self.profiler.profile('chess.pgn.read_game'):
                        game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    current_batch.append(game)
                    
                    if len(current_batch) >= batch_size:
                        batch_queue.put(current_batch)
                        current_batch = []
                
                if current_batch:
                    batch_queue.put(current_batch)
            
            batch_queue.put(None)
        
        # Thread for writing results to CSV
        def result_writer():
            while True:
                try:
                    result = results_queue.get(timeout=1.0)
                    if result is None:  # End signal
                        break
                    
                    batch_results, batch_id = result
                    self._write_batch_results(writer, batch_results)
                    results_queue.task_done()
                except:
                    # Timeout or other error, continue checking
                    continue
        
        # Start the threads
        reader_thread = threading.Thread(target=game_reader)
        writer_thread = threading.Thread(target=result_writer)
        reader_thread.start()
        writer_thread.start()
        
        # Submit batches for processing as they become available
        active_futures = {}
        batch_id = 0
        
        try:
            while True:
                # Get next batch from queue
                batch = batch_queue.get()
                if batch is None:  # End signal
                    break
                
                # Submit batch for processing
                future = executor.submit(self._process_game_batch, batch, batch_id)
                active_futures[future] = batch_id
                batch_id += 1
                
                # Process completed batches
                completed_futures = []
                for future in list(active_futures.keys()):
                    if future.done():
                        try:
                            batch_results = future.result()
                            results_queue.put((batch_results, active_futures[future]))
                        except Exception as exc:
                            print(f'Batch {active_futures[future]} generated an exception: {exc}')
                        completed_futures.append(future)
                
                # Remove completed futures
                for future in completed_futures:
                    del active_futures[future]
            
            # Wait for all remaining batches to complete
            for future in active_futures:
                try:
                    batch_results = future.result()
                    results_queue.put((batch_results, active_futures[future]))
                except Exception as exc:
                    print(f'Batch {active_futures[future]} generated an exception: {exc}')
            
            # Signal writer thread to finish
            results_queue.put(None)
        
        finally:
            # Wait for threads to finish
            reader_thread.join()
            writer_thread.join()
    
    def _process_game_batch(self, games: List[chess.pgn.Game], batch_id: int = 0) -> List[Tuple[GameMetrics, GameMetrics]]:
        """Process a batch of games and return their metrics"""
        batch_results = []
        batch_processed = 0
        batch_skipped = 0
        
        for game in games:
            with self.profiler.profile('process_game (total)'):
                metrics = self.process_game(game)
            if metrics is not None:
                batch_results.append(metrics)
                batch_processed += 1
            else:
                batch_skipped += 1
        
        with self._lock:
            self.games_processed += batch_processed
            self.games_skipped += batch_skipped
            
            if self.limit and self.games_processed >= self.limit:
                self._stop_processing = True
            
            current_time = time.time()
            if current_time - self._last_progress_time >= self._progress_interval:
                total_read = self.games_processed + self.games_skipped
                elapsed = current_time - self._start_time
                rate = total_read / elapsed if elapsed > 0 else 0
                process_pct = (self.games_processed / total_read * 100) if total_read > 0 else 0
                
                print(f"Read: {total_read:,} games | "
                      f"Processed: {self.games_processed:,} ({process_pct:.1f}%) | "
                      f"Skipped: {self.games_skipped:,} | "
                      f"Rate: {rate:.1f} games/sec")
                self._last_progress_time = current_time
        
        return batch_results
    
    def _write_batch_results(self, writer, batch_results: List[Tuple[GameMetrics, GameMetrics]]):
        """Write a batch of results to the CSV file"""
        for white_metrics, black_metrics in batch_results:
            self.write_metrics(writer, white_metrics)
            self.write_metrics(writer, black_metrics)
    
    def has_evaluation_data(self, game: chess.pgn.Game) -> bool:
        """Check if the game has %eval annotations on its moves"""
        if not game.variations:
            return False
            
        node = game
        move_count = 0
        
        while node.variations and move_count < 5:
            next_node = node.variation(0)
            if '%eval' in next_node.comment:
                return True
            node = next_node
            move_count += 1
        
        return False

    def process_game(self, game: chess.pgn.Game) -> Optional[Tuple[GameMetrics, GameMetrics]]:
        """Process a single game and return metrics for both players"""
        headers = game.headers
        
        time_control = headers.get('TimeControl', '')
        if not time_control:
            return None
            
        with self.profiler.profile('has_evaluation_data'):
            has_eval = self.has_evaluation_data(game)
        if not has_eval:
            return None
        
        # Extract metadata
        white_elo = headers.get('WhiteElo', '0')
        black_elo = headers.get('BlackElo', '0')
        
        try:
            white_elo = int(white_elo)
            black_elo = int(black_elo)
        except ValueError:
            return None
        
        if white_elo == 0 or black_elo == 0:
            return None
        
        with self.profiler.profile('replay_game'):
            boards = []
            moves = []
            evals = []
            node = game
            board = game.board()
            
            while node.variations:
                next_node = node.variation(0)
                boards.append(board.copy())
                moves.append(next_node)
                
                eval_value = self.extract_eval(next_node.comment)
                evals.append(eval_value)
                
                board.push(next_node.move)
                node = next_node
        
        with self.profiler.profile('GameDivider.divide_game'):
            midgame_start, endgame_start = GameDivider.divide_game(boards)
        
        if midgame_start is None:
            return None
        
        midgame_end = endgame_start if endgame_start is not None else len(boards)
        
        with self.profiler.profile('calculate_player_metrics'):
            white_metrics = self.calculate_player_metrics(
                moves, evals, midgame_start, midgame_end,
                'White', time_control, white_elo
            )
            
            black_metrics = self.calculate_player_metrics(
                moves, evals, midgame_start, midgame_end,
                'Black', time_control, black_elo
            )
        
        return white_metrics, black_metrics
    
    def extract_eval(self, comment: str) -> Optional[float]:
        """Extract evaluation from comment like '[%eval 0.25]'"""
        match = re.search(r'\[%eval ([+-]?\d+\.?\d*)\]', comment)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None
    
    def calculate_player_metrics(
        self,
        moves: List[chess.pgn.ChildNode],
        evals: List[Optional[float]],
        midgame_start: int,
        midgame_end: int,
        color: str,
        time_control: str,
        rating: int
    ) -> GameMetrics:
        """Calculate metrics for a single player during the middlegame"""
        
        # Determine which moves belong to this player
        color_offset = 0 if color == 'White' else 1
        
        total_centipawn_loss = 0.0
        move_count = 0
        inaccuracies = 0
        mistakes = 0
        blunders = 0
        
        for idx in range(midgame_start, midgame_end):
            # Skip if not this player's move
            if idx % 2 != color_offset:
                continue
            
            move_count += 1
            
            # Calculate centipawn loss
            if idx > 0 and evals[idx] is not None and evals[idx - 1] is not None:
                eval_before = evals[idx - 1]
                eval_after = evals[idx]
                
                # Flip sign for black (eval is from white's perspective)
                if color == 'Black':
                    eval_before = -eval_before
                    eval_after = -eval_after
                
                # Centipawn loss is how much worse the position got
                cp_loss = max(0, (eval_before - eval_after) * 100)
                total_centipawn_loss += cp_loss
            
            # Count errors from NAGs (Numeric Annotation Glyphs)
            move_node = moves[idx]
            for nag in move_node.nags:
                if nag == 6:  # ?!
                    inaccuracies += 1
                elif nag == 2:  # ?
                    mistakes += 1
                elif nag == 4:  # ??
                    blunders += 1
        
        acpl = total_centipawn_loss / move_count if move_count > 0 else 0.0
        
        return GameMetrics(
            time_control=time_control,
            rating=rating,
            color=color,
            acpl_midgame=acpl,
            inaccuracies_midgame=inaccuracies,
            mistakes_midgame=mistakes,
            blunders_midgame=blunders,
            num_moves_midgame=move_count
        )
    
    def write_metrics(self, writer, metrics: GameMetrics):
        """Write a single row to the CSV"""
        writer.writerow([
            metrics.time_control,
            metrics.rating,
            metrics.color,
            f"{metrics.acpl_midgame:.2f}",
            metrics.inaccuracies_midgame,
            metrics.mistakes_midgame,
            metrics.blunders_midgame,
            metrics.num_moves_midgame
        ])


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process chess PGN files with multithreading')
    parser.add_argument('--input', '-i', default="data/lichess_db_standard_rated_2025-09.pgn",
                       help='Input PGN file path')
    parser.add_argument('--output', '-o', default="data/lichess_db_standard_rated_2025-09_metrics.csv",
                       help='Output CSV file path')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker threads (default: auto-detect)')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='Batch size for processing games (default: 1000)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='Limit number of successfully processed games (default: no limit)')
    
    args = parser.parse_args()
    
    analyzer = ChessAnalyzer(args.input, args.output, args.workers, args.limit)
    analyzer.process_file(args.batch_size)


if __name__ == '__main__':
    main()