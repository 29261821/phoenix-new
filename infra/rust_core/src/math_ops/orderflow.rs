pub fn calculate_ofi(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .zip(close.iter())
        .zip(volume.iter())
        .map(|(((h, l), c), v)| {
            let spread = (h - l).abs().max(1e-12);
            let close_location = ((c - l) - (h - c)) / spread;
            close_location * v
        })
        .collect()
}

