pub fn rolling_volatility(values: &[f64], window: usize) -> Vec<f64> {
    if window < 2 {
        return vec![0.0; values.len()];
    }

    let mut output = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let start = (index + 1).saturating_sub(window);
        let sample = &values[start..=index];
        if sample.len() < 2 {
            output.push(0.0);
            continue;
        }

        let mean = sample.iter().sum::<f64>() / sample.len() as f64;
        let variance = sample
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / (sample.len() - 1) as f64;
        output.push(variance.sqrt());
    }
    output
}

