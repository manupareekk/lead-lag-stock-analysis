import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)

@dataclass
class FetchMetrics:
    """Metrics for a single fetch operation"""
    symbols: List[str]
    fetch_time: float
    cache_hits: int
    cache_misses: int
    method_used: str
    timestamp: datetime
    data_points: int
    
class PerformanceMonitor:
    """Monitor and track performance of data fetching operations"""
    
    def __init__(self):
        self.metrics: List[FetchMetrics] = []
        self.session_start = datetime.now()
    
    def record_fetch(self, 
                    symbols: List[str],
                    fetch_time: float,
                    cache_hits: int = 0,
                    cache_misses: int = 0,
                    method_used: str = "unknown",
                    data_points: int = 0):
        """Record metrics for a fetch operation"""
        metric = FetchMetrics(
            symbols=symbols,
            fetch_time=fetch_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            method_used=method_used,
            timestamp=datetime.now(),
            data_points=data_points
        )
        self.metrics.append(metric)
        
        # Log performance info
        symbols_str = ", ".join(symbols[:3]) + ("..." if len(symbols) > 3 else "")
        logger.info(f"Fetch completed: {len(symbols)} symbols ({symbols_str}) in {fetch_time:.2f}s using {method_used}")
    
    def get_performance_summary(self, last_n_minutes: Optional[int] = None) -> Dict:
        """Get performance summary statistics"""
        if not self.metrics:
            return {"message": "No fetch operations recorded yet"}
        
        # Filter metrics by time if specified
        filtered_metrics = self.metrics
        if last_n_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
            filtered_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not filtered_metrics:
            return {"message": f"No fetch operations in the last {last_n_minutes} minutes"}
        
        # Calculate statistics
        fetch_times = [m.fetch_time for m in filtered_metrics]
        total_symbols = sum(len(m.symbols) for m in filtered_metrics)
        total_cache_hits = sum(m.cache_hits for m in filtered_metrics)
        total_cache_misses = sum(m.cache_misses for m in filtered_metrics)
        
        # Method usage statistics
        method_counts = {}
        for metric in filtered_metrics:
            method_counts[metric.method_used] = method_counts.get(metric.method_used, 0) + 1
        
        return {
            "total_operations": len(filtered_metrics),
            "total_symbols_fetched": total_symbols,
            "avg_fetch_time": round(statistics.mean(fetch_times), 3),
            "min_fetch_time": round(min(fetch_times), 3),
            "max_fetch_time": round(max(fetch_times), 3),
            "median_fetch_time": round(statistics.median(fetch_times), 3),
            "total_fetch_time": round(sum(fetch_times), 3),
            "cache_hit_rate": round(total_cache_hits / (total_cache_hits + total_cache_misses) * 100, 1) if (total_cache_hits + total_cache_misses) > 0 else 0,
            "method_usage": method_counts,
            "avg_symbols_per_operation": round(total_symbols / len(filtered_metrics), 1),
            "operations_per_minute": round(len(filtered_metrics) / max(1, (datetime.now() - filtered_metrics[0].timestamp).total_seconds() / 60), 2)
        }
    
    def get_speed_comparison(self) -> Dict:
        """Compare performance between different fetching methods"""
        if not self.metrics:
            return {"message": "No data available for comparison"}
        
        method_stats = {}
        
        for metric in self.metrics:
            method = metric.method_used
            if method not in method_stats:
                method_stats[method] = {
                    "times": [],
                    "symbol_counts": [],
                    "operations": 0
                }
            
            method_stats[method]["times"].append(metric.fetch_time)
            method_stats[method]["symbol_counts"].append(len(metric.symbols))
            method_stats[method]["operations"] += 1
        
        comparison = {}
        for method, stats in method_stats.items():
            if stats["operations"] > 0:
                comparison[method] = {
                    "operations": stats["operations"],
                    "avg_time": round(statistics.mean(stats["times"]), 3),
                    "avg_symbols_per_op": round(statistics.mean(stats["symbol_counts"]), 1),
                    "time_per_symbol": round(statistics.mean(stats["times"]) / max(1, statistics.mean(stats["symbol_counts"])), 4)
                }
        
        return comparison
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        self.metrics.clear()
        self.session_start = datetime.now()
        logger.info("Performance metrics cleared")
    
    def export_metrics_csv(self, filename: str = "fetch_performance.csv"):
        """Export metrics to CSV file"""
        import csv
        
        if not self.metrics:
            logger.warning("No metrics to export")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'method_used', 'symbol_count', 'symbols', 'fetch_time', 
                         'cache_hits', 'cache_misses', 'data_points']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow({
                    'timestamp': metric.timestamp.isoformat(),
                    'method_used': metric.method_used,
                    'symbol_count': len(metric.symbols),
                    'symbols': '|'.join(metric.symbols),
                    'fetch_time': metric.fetch_time,
                    'cache_hits': metric.cache_hits,
                    'cache_misses': metric.cache_misses,
                    'data_points': metric.data_points
                })
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filename}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()