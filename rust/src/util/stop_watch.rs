use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct StopWatch {
    begin: Instant,
    laps: Vec<(Duration, String)>,
    end: Instant,
    total_ticks: i32,
    last_lap_time: Instant,

}

impl StopWatch {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            begin: now,
            laps: Vec::new(),
            end: now,
            total_ticks: 0,
            last_lap_time: now,
        }
    }

    pub fn start(&mut self) {
        let now = Instant::now();
        self.begin = now;
        self.last_lap_time = now;
    }

    pub fn lap(&mut self, label: &str) {
        let now = Instant::now();
        let duration = now - self.last_lap_time;
        self.last_lap_time = now;
        self.laps.push((duration, label.to_string()));
    }

    pub fn tick(&mut self) {
        self.total_ticks += 1;
        self.end = Instant::now();
    }

    pub fn stop(&mut self) {
        self.end = Instant::now();
    }

    pub fn calc_stats(&self) -> Option<(Duration, Duration, Duration)> {
        if self.laps.is_empty() {
            return None;
        }

        let mut min_duration = Duration::MAX;
        let mut max_duration = Duration::new(0, 0);
        let mut avg_duration = Duration::new(0, 0);

        for lap in &self.laps {
            min_duration = min_duration.min(lap.0);
            max_duration = max_duration.max(lap.0);
            avg_duration = avg_duration + lap.0;
        }
        avg_duration = avg_duration / self.laps.len() as u32;

        Some((
            min_duration,
            max_duration,
            avg_duration,
        ))
    }

    pub fn calc_detailed_stats(&self) -> (Option<(Duration, Duration, Duration)>, HashMap<String, (Duration, Duration, Duration)>) {
        let total_stats = self.calc_stats();

        let mut lap_groups: HashMap<String, Vec<Duration>> = HashMap::new();
        for (duration, label) in &self.laps {
            lap_groups.entry(label.clone())
                .or_insert_with(Vec::new)
                .push(*duration);
        }

        let mut label_stats: HashMap<String, (Duration, Duration, Duration)> = HashMap::new();

        for (label, durations) in lap_groups {
            let min = durations.iter().min().copied().unwrap_or(Duration::MAX);
            let max = durations.iter().max().copied().unwrap_or(Duration::ZERO);
            let sum: Duration = durations.iter().sum();
            let avg = sum / durations.len() as u32;

            label_stats.insert(label, (min, max, avg));
        }

        (total_stats, label_stats)
    }

    pub fn to_string(&self) -> String {
        let stats = self.calc_stats().unwrap();
        let outstr = format!("\
            Min: {} | {:.2} fps\n\
            Max: {} | {:.2} fps\n\
            Avg: {} | {:.2} fps\n",
            StopWatch::format_duration_ms(stats.0),
            1.0 / stats.0.as_secs_f32(),
            StopWatch::format_duration_ms(stats.1),
            1.0 / stats.1.as_secs_f32(),
            StopWatch::format_duration_ms(stats.2),
            1.0 / stats.2.as_secs_f32(),
        );
        outstr
    }

    pub fn format_duration_ms(duration: Duration) -> String {
        let ns = duration.as_nanos();
        if ns < 1_000 {  // less than 1 microsecond
            format!("{} ns", ns)
        } else if ns < 1_000_000 {  // less than 1 millisecond
            format!("{:.3} µs", ns as f64 / 1_000.0)
        } else {
            format!("{:.3} ms", ns as f64 / 1_000_000.0)
        }
    }


    pub fn to_string_detailed(&self) -> String {
        let (total_stats, label_stats) = self.calc_detailed_stats();
        let mut outstr = String::new();

        if let Some(stats) = total_stats {
            outstr.push_str("[ Total ]\n");
            outstr.push_str(&format!("  Min: {} | {:.2} fps\n",
            StopWatch::format_duration_ms(stats.0), 1.0 / stats.0.as_secs_f32()));
            outstr.push_str(&format!("  Max: {} | {:.2} fps\n",
            StopWatch::format_duration_ms(stats.1), 1.0 / stats.1.as_secs_f32()));
            outstr.push_str(&format!("  Avg: {} | {:.2} fps\n",
            StopWatch::format_duration_ms(stats.2), 1.0 / stats.2.as_secs_f32()));
        }

        if !label_stats.is_empty() {
            outstr.push_str("\n[ Laps ]\n");
            for (label, (min, max, avg)) in label_stats {
                outstr.push_str(&format!("{}:\n", label));
                outstr.push_str(&format!("  Min: {} | {:.2} fps\n", StopWatch::format_duration_ms(min), 1.0 / min.as_secs_f32()));
                outstr.push_str(&format!("  Max: {} | {:.2} fps\n", StopWatch::format_duration_ms(max), 1.0 / min.as_secs_f32()));
                outstr.push_str(&format!("  Avg: {} | {:.2} fps\n", StopWatch::format_duration_ms(avg), 1.0 / min.as_secs_f32()));
                outstr.push_str("\n");
            }
        }

        outstr
    }
}