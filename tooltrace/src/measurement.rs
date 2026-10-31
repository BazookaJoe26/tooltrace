//! Independent detection of the optional printed measurement rulers.
//! No detected ruler point is used to fit or rescale the calibration.
use crate::calibration::PIXELS_PER_MM as PPM;
use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use opencv::{
    core::{BORDER_DEFAULT, Mat, Size},
    imgproc,
    prelude::*,
};
use tooltrace_common::PaperSize;

pub const TOLERANCE_MM: f64 = 0.5;

#[derive(Debug)]
pub struct AxisMeasurement {
    pub measured_mm: f64,
    pub endpoints: [(f64, f64); 2],
    pub ticks_found: usize,
    pub ticks_expected: usize,
    pub tick_rms_mm: f64,
}

#[derive(Debug)]
pub struct AxisCheck {
    pub axis: char,
    pub expected_mm: f64,
    pub measurement: Option<AxisMeasurement>,
    pub unavailable_reason: String,
}

struct Layout {
    axis: char,
    fixed: f64,
    start: f64,
    end: f64,
    ticks: Vec<f64>,
}

fn layouts(paper: PaperSize, tag_size: f64) -> Vec<Layout> {
    let (w, h) = paper.dimensions_mm();
    let make = |axis, fixed: f64, start: f64, end: f64| {
        let length = end - start;
        let mut ticks: Vec<f64> = (0..=(length.max(0.0) / 10.0).floor() as usize)
            .map(|i| i as f64 * 10.0)
            .collect();
        if (ticks.last().copied().unwrap_or(0.0) - length).abs() > 0.001 {
            ticks.push(length);
        }
        // Y labels increase upward from the lower endpoint on the paper.
        if axis == 'Y' {
            ticks = ticks.into_iter().rev().map(|p| length - p).collect();
        }
        Layout {
            axis,
            fixed: fixed * PPM,
            start: start * PPM,
            end: end * PPM,
            ticks: ticks.into_iter().map(|p| (start + p) * PPM).collect(),
        }
    };
    vec![
        make(
            'X',
            h - 15.0 - tag_size / 2.0,
            25.0 + tag_size,
            w - 25.0 - tag_size,
        ),
        make(
            'Y',
            15.0 + tag_size / 2.0,
            25.0 + tag_size,
            h - 25.0 - tag_size,
        ),
    ]
}

fn regression(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let variance = points.iter().map(|p| (p.0 - x).powi(2)).sum::<f64>();
    if variance < 1e-6 {
        return None;
    }
    let slope = points.iter().map(|p| (p.0 - x) * (p.1 - y)).sum::<f64>() / variance;
    Some((y - slope * x, slope))
}

fn detect_axis(
    contrast: &[f32],
    width: usize,
    height: usize,
    layout: &Layout,
) -> std::result::Result<AxisMeasurement, String> {
    let Layout {
        axis,
        fixed,
        start,
        end,
        ticks,
    } = layout;
    if end - start < 20.0 * PPM {
        return Err("not enough space for a ruler".into());
    }
    let sample = |along: f64, across: f64| -> f32 {
        let (x, y) = if *axis == 'X' {
            (along.round() as i32, across.round() as i32)
        } else {
            (across.round() as i32, along.round() as i32)
        };
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return 0.0;
        }
        contrast[y as usize * width + x as usize]
    };
    let search = (3.0 * PPM).ceil() as i32;
    // A long stroke is necessary but not sufficient: the cross ticks are
    // subsequently matched too, so unrelated straight edges do not pass.
    let mut best = (0usize, *fixed);
    for offset in -search..=search {
        let across = fixed + offset as f64;
        let score = ((*start as i32 + 12)..(*end as i32 - 12))
            .step_by(3)
            .filter(|&p| (-2..=2).any(|d| sample(p as f64, across + d as f64) > 20.0))
            .count();
        if score > best.0 {
            best = (score, across);
        }
    }
    let mut ridge = Vec::new();
    for p in ((*start as i32 + 12)..(*end as i32 - 12)).step_by(3) {
        // Cross ticks would bias the fitted direction; use only the axis stroke.
        if ticks.iter().any(|t| (p as f64 - t).abs() < 5.0) {
            continue;
        }
        let strongest = (-10..=10)
            .map(|d| (sample(p as f64, best.1 + d as f64), best.1 + d as f64))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap();
        if strongest.0 > 20.0 {
            ridge.push((p as f64, strongest.1));
        }
    }
    let (mut intercept, mut slope) = regression(&ridge).ok_or("axis stroke not detected")?;
    for _ in 0..3 {
        let retained: Vec<_> = ridge
            .iter()
            .copied()
            .filter(|p| (p.1 - intercept - slope * p.0).abs() < 0.5 * PPM)
            .collect();
        (intercept, slope) = regression(&retained).ok_or("axis stroke inconsistent")?;
    }
    if slope.abs() > 0.06 {
        return Err("axis direction inconsistent".into());
    }
    let probes: Vec<_> = ((*start as i32 + 12)..(*end as i32 - 12))
        .step_by(3)
        .collect();
    let coverage = probes
        .iter()
        .filter(|&&p| {
            (-4..=4).any(|d| sample(p as f64, intercept + slope * p as f64 + d as f64) > 20.0)
        })
        .count() as f64
        / probes.len() as f64;
    if coverage < 0.7 {
        return Err("axis absent, faint or occluded".into());
    }

    let first = (*start as i32 - search).max(0);
    let last = *end as i32 + search;
    let inner = (0.5 * PPM).ceil() as i32;
    let outer = (2.7 * PPM).ceil() as i32;
    let mut profile = Vec::new();
    let mut extents = Vec::new();
    for p in first..=last {
        let across = intercept + slope * p as f64;
        let mut sums = [0.0_f64; 2];
        let mut counts = [0usize; 2];
        for (side, sign) in [-1.0, 1.0].iter().enumerate() {
            for d in inner..=outer {
                let value = sample(p as f64, across + sign * d as f64) as f64;
                sums[side] += (value - 12.0).max(0.0);
                if value > 20.0 {
                    counts[side] += 1;
                }
            }
        }
        profile.push(if counts[0].min(counts[1]) >= 3 {
            sums[0].min(sums[1])
        } else {
            0.0
        });
        extents.push(counts[0].min(counts[1]));
    }
    // Group a tick's two or three dark columns into one subpixel location.
    let mut candidates = Vec::new();
    let mut i = 0;
    while i < profile.len() {
        if profile[i] < 60.0 {
            i += 1;
            continue;
        }
        let begin = i;
        while i < profile.len() && profile[i] >= 60.0 {
            i += 1;
        }
        if i - begin > (0.8 * PPM) as usize {
            continue;
        }
        let sum: f64 = profile[begin..i].iter().sum();
        let center = (begin..i)
            .map(|j| (first as f64 + j as f64) * profile[j])
            .sum::<f64>()
            / sum;
        let extent = extents[begin..i].iter().copied().max().unwrap();
        candidates.push((center, extent));
    }
    let mut used = vec![false; candidates.len()];
    let mut matches = Vec::new();
    let mut endpoints = [None, None];
    for (index, &expected) in ticks.iter().enumerate() {
        let endpoint = index == 0 || index + 1 == ticks.len();
        let candidate = candidates
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                !used[*i] && (p.0 - expected).abs() <= 3.0 * PPM && (!endpoint || p.1 >= 9)
            })
            .min_by(|a, b| {
                (a.1.0 - expected)
                    .abs()
                    .total_cmp(&(b.1.0 - expected).abs())
            });
        if let Some((i, &(measured, _))) = candidate {
            used[i] = true;
            matches.push((expected, measured));
            if index == 0 {
                endpoints[0] = Some(measured);
            }
            if index + 1 == ticks.len() {
                endpoints[1] = Some(measured);
            }
        }
    }
    let (Some(a), Some(b)) = (endpoints[0], endpoints[1]) else {
        return Err("both endpoint ticks must be visible".into());
    };
    if matches.len() < 3.max((ticks.len() as f64 * 0.7).ceil() as usize) {
        return Err("too few matching cross ticks".into());
    }
    let (zero, scale) = regression(&matches).ok_or("invalid tick spacing")?;
    let rms = (matches
        .iter()
        .map(|p| (p.1 - zero - scale * p.0).powi(2))
        .sum::<f64>()
        / matches.len() as f64)
        .sqrt()
        / PPM;
    if rms > 0.4 || !(0.94..1.06).contains(&scale) {
        return Err("cross tick spacing inconsistent".into());
    }
    let point = |p| {
        if *axis == 'X' {
            (p, intercept + slope * p)
        } else {
            (intercept + slope * p, p)
        }
    };
    Ok(AxisMeasurement {
        measured_mm: (b - a) * (1.0 + slope * slope).sqrt() / PPM,
        endpoints: [point(a), point(b)],
        ticks_found: matches.len(),
        ticks_expected: ticks.len(),
        tick_rms_mm: rms,
    })
}

pub fn check_measurement_axes(
    image: &RgbImage,
    paper: PaperSize,
    tag_size: f64,
) -> Result<Vec<AxisCheck>> {
    let gray = image::imageops::grayscale(image);
    let raw = Mat::from_slice(gray.as_raw())?;
    let mat = raw.reshape(1, gray.height() as i32)?;
    let mut background = Mat::default();
    imgproc::gaussian_blur(
        &mat,
        &mut background,
        Size::new(0, 0),
        0.6 * PPM,
        0.6 * PPM,
        BORDER_DEFAULT,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let contrast: Vec<f32> = background
        .data_bytes()?
        .iter()
        .zip(gray.as_raw())
        .map(|(&b, &a)| b as f32 - a as f32)
        .collect();
    Ok(layouts(paper, tag_size)
        .iter()
        .map(|layout| {
            let (measurement, unavailable_reason) = match detect_axis(
                &contrast,
                image.width() as usize,
                image.height() as usize,
                layout,
            ) {
                Ok(value) => (Some(value), String::new()),
                Err(reason) => (None, reason),
            };
            AxisCheck {
                axis: layout.axis,
                expected_mm: (layout.end - layout.start) / PPM,
                measurement,
                unavailable_reason,
            }
        })
        .collect())
}

pub fn report(checks: &[AxisCheck]) -> String {
    let mut text =
        format!("Independent printed ruler check (tolerance +/-{TOLERANCE_MM:.2} mm):\n");
    for check in checks {
        if let Some(value) = &check.measurement {
            let error = value.measured_mm - check.expected_mm;
            let status = if error.abs() > TOLERANCE_MM {
                "OUTSIDE TOLERANCE"
            } else {
                "within tolerance"
            };
            text.push_str(&format!("  {}: expected {:.2} mm, measured {:.2} mm, deviation {:+.2} mm ({:+.2}%), {status}\n    ticks {}/{}, spacing residual RMS {:.3} mm\n",
                check.axis, check.expected_mm, value.measured_mm, error, 100.0 * error / check.expected_mm,
                value.ticks_found, value.ticks_expected, value.tick_rms_mm));
        } else {
            text.push_str(&format!(
                "  {}: not verified ({})\n",
                check.axis, check.unavailable_reason
            ));
        }
    }
    text.push_str("Rulers validate the paper plane only; object height and physical print scaling are not verified.\n");
    text
}

pub fn save_overlay(image: &RgbImage, checks: &[AxisCheck], path: &str) -> Result<()> {
    let font = FontRef::try_from_slice(include_bytes!("../../fonts/NotoSans-Regular.ttf"))?;
    let mut overlay = image.clone();
    for (i, check) in checks.iter().enumerate() {
        let label = if let Some(value) = &check.measurement {
            let error = value.measured_mm - check.expected_mm;
            let color = if error.abs() > TOLERANCE_MM {
                Rgb([255, 60, 0])
            } else {
                Rgb([0, 190, 50])
            };
            let [a, b] = value.endpoints;
            draw_line_segment_mut(
                &mut overlay,
                (a.0 as f32, a.1 as f32),
                (b.0 as f32, b.1 as f32),
                color,
            );
            for p in [a, b] {
                draw_filled_circle_mut(&mut overlay, (p.0 as i32, p.1 as i32), 6, color);
            }
            format!(
                "{}: {:.2} / {:.2} mm ({:+.2} mm)",
                check.axis, value.measured_mm, check.expected_mm, error
            )
        } else {
            format!("{}: not verified", check.axis)
        };
        draw_text_mut(
            &mut overlay,
            Rgb([220, 0, 0]),
            25,
            25 + i as i32 * 32,
            PxScale::from(26.0),
            &font,
            &label,
        );
    }
    overlay
        .save(path)
        .with_context(|| format!("Could not save ruler overlay: {path}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paper_image(paper: PaperSize) -> RgbImage {
        let (w, h) = paper.dimensions_mm();
        RgbImage::from_pixel(
            (w * PPM).ceil() as u32,
            (h * PPM).ceil() as u32,
            Rgb([210, 210, 210]),
        )
    }

    fn draw_ruler(image: &mut RgbImage, layout: &Layout, scale: f64, missing_endpoint: bool) {
        // Independent renderer: known geometry, with a small translation/tilt.
        let coordinate = |along: f64, across: f64| {
            let p = layout.start + (along - layout.start) * scale + 2.0;
            let q = layout.fixed + across + 0.008 * (p - layout.start);
            if layout.axis == 'X' {
                (p as f32, q as f32)
            } else {
                (q as f32, p as f32)
            }
        };
        let black = Rgb([35, 35, 35]);
        for width in [-0.7_f64, 0.0, 0.7] {
            draw_line_segment_mut(
                image,
                coordinate(layout.start, width),
                coordinate(layout.end, width),
                black,
            );
        }
        for (i, &p) in layout.ticks.iter().enumerate() {
            if missing_endpoint && i == 0 {
                continue;
            }
            let major = i == 0 || i + 1 == layout.ticks.len() || i % 5 == 0;
            let length = if major { 2.0 } else { 1.0 } * PPM;
            for width in [-0.7, 0.0, 0.7] {
                draw_line_segment_mut(
                    image,
                    coordinate(p + width, -length),
                    coordinate(p + width, length),
                    black,
                );
            }
        }
    }

    #[test]
    fn measures_independent_axis_scale_errors_instead_of_forcing_nominal_lengths() {
        let mut image = paper_image(PaperSize::A4);
        let layout = layouts(PaperSize::A4, 40.0);
        draw_ruler(&mut image, &layout[0], 1.012, false);
        draw_ruler(&mut image, &layout[1], 0.996, false);
        let checks = check_measurement_axes(&image, PaperSize::A4, 40.0).unwrap();
        for (check, scale) in checks.iter().zip([1.012_f64, 0.996]) {
            let measured = check
                .measurement
                .as_ref()
                .unwrap_or_else(|| panic!("{check:?}"));
            assert!(
                (measured.measured_mm - check.expected_mm * scale).abs() < 0.15,
                "{check:?}"
            );
            assert_eq!(measured.ticks_found, measured.ticks_expected);
        }
        assert!(report(&checks).contains("OUTSIDE TOLERANCE"));
    }

    #[test]
    fn absent_axes_and_a_missing_end_tick_are_not_reported_as_verified() {
        let mut image = paper_image(PaperSize::A4);
        assert!(
            check_measurement_axes(&image, PaperSize::A4, 40.0)
                .unwrap()
                .iter()
                .all(|c| c.measurement.is_none())
        );
        let layout = layouts(PaperSize::A4, 40.0);
        draw_ruler(&mut image, &layout[0], 1.0, true);
        let checks = check_measurement_axes(&image, PaperSize::A4, 40.0).unwrap();
        assert!(checks[0].measurement.is_none(), "{checks:?}");
        assert!(checks[1].measurement.is_none());
        // A nearby parallel object edge also has no cross ticks.
        draw_line_segment_mut(
            &mut image,
            (layout[1].fixed as f32, layout[1].start as f32),
            (layout[1].fixed as f32, layout[1].end as f32),
            Rgb([0, 0, 0]),
        );
        assert!(
            check_measurement_axes(&image, PaperSize::A4, 40.0).unwrap()[1]
                .measurement
                .is_none()
        );
    }

    #[test]
    fn ruler_layout_follows_paper_and_tag_size() {
        for (paper, size) in [(PaperSize::Letter, 50.0), (PaperSize::A3, 40.0)] {
            let mut image = paper_image(paper);
            for layout in layouts(paper, size) {
                draw_ruler(&mut image, &layout, 1.0, false);
            }
            let checks = check_measurement_axes(&image, paper, size).unwrap();
            assert!(
                checks.iter().all(|c| c
                    .measurement
                    .as_ref()
                    .is_some_and(|m| (m.measured_mm - c.expected_mm).abs() < 0.15)),
                "{checks:?}"
            );
        }
    }
}
